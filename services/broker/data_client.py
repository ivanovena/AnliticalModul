# data_client.py
import os
import json
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Union
import requests
from datetime import datetime, timedelta
import aiohttp
from redis_cache import get_redis_cache

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataClient")

class DataClient:
    """Cliente para obtener datos del servicio de ingestion y streaming con caché en Redis"""
    
    def __init__(self):
        """Inicializa el cliente de datos"""
        self.ingestion_url = os.getenv("INGESTION_SERVICE_URL", "http://ingestion:8080")
        self.streaming_url = os.getenv("STREAMING_SERVICE_URL", "http://streaming:8090")
        self.fmp_api_key = os.getenv("FMP_API_KEY", "")
        
        # Flag para usar directamente FMP API cuando ingestion falla
        self.direct_fmp_mode = False
        self.ingestion_error_count = 0
        self.max_ingestion_errors = 5  # Cambiar a modo FMP directo después de 5 errores
        
        # Timeouts y reintentos
        self.request_timeout = 10  # Aumentado de 5 a 10 segundos
        self.max_retries = 3
        self.retry_delay = 1  # segundos
        
        # TTLs de caché (tiempo de vida en segundos)
        self.cache_ttl = {
            "market_data": 300,    # 5 minutos
            "historical": 3600,    # 1 hora
            "prediction": 600,     # 10 minutos
            "symbol_list": 3600,   # 1 hora
            "news": 1800,          # 30 minutos (nuevo)
            "profile": 86400,      # 24 horas (nuevo)
            "ratios": 86400,       # 24 horas (nuevo)
            "earnings": 86400,     # 24 horas (nuevo)
            "sentiment": 3600      # 1 hora (nuevo)
        }
        
        # Obtener instancia de Redis
        try:
            self.redis = get_redis_cache()
        except Exception as e:
            logger.error(f"Error inicializando Redis, se usará un diccionario en memoria: {e}")
            self.redis = self._get_memory_cache()
            
        self.session = None
    
    def _get_memory_cache(self):
        """Proporciona un fallback de caché en memoria cuando Redis no está disponible"""
        class MemoryCache:
            def __init__(self):
                self.cache = {}
                self.ttl_values = {}
            
            def set(self, key, value, ttl=None):
                self.cache[key] = value
                if ttl:
                    self.ttl_values[key] = time.time() + ttl
                return True
            
            def get(self, key, default=None):
                if key in self.cache:
                    # Comprobar TTL
                    if key in self.ttl_values and time.time() > self.ttl_values[key]:
                        del self.cache[key]
                        del self.ttl_values[key]
                        return default
                    return self.cache.get(key, default)
                return default
                
            def delete(self, key):
                if key in self.cache:
                    del self.cache[key]
                if key in self.ttl_values:
                    del self.ttl_values[key]
                return True
                
            def exists(self, key):
                return key in self.cache
                
            def keys(self, pattern="*"):
                import fnmatch
                return [k for k in self.cache.keys() if fnmatch.fnmatch(k, pattern)]
                
        logger.warning("Usando caché en memoria como fallback")
        return MemoryCache()
    
    async def _get_aiohttp_session(self):
        """Obtiene una sesión aiohttp reutilizable"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.request_timeout)
            )
        return self.session
    
    async def close(self):
        """Cierra la sesión HTTP"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Obtiene datos de mercado para un símbolo utilizando Redis como caché
        
        Args:
            symbol: Símbolo del activo
        
        Returns:
            Datos de mercado para el símbolo
        """
        # Construir clave para Redis
        cache_key = f"market_data:{symbol}"
        
        # Intentar obtener desde caché
        cached_data = self.redis.get(cache_key)
        if cached_data:
            logger.debug(f"Datos de mercado para {symbol} obtenidos desde caché")
            return cached_data
        
        try:
            # Si estamos en modo FMP directo, saltamos el intento de usar ingestion
            if self.direct_fmp_mode and self.fmp_api_key:
                fallback_data = await self._get_market_data_fallback(symbol)
                if fallback_data:
                    self.redis.set(cache_key, fallback_data, ttl=self.cache_ttl["market_data"] // 2)
                    return fallback_data
                return {}
                
            # Intentar obtener desde el servicio de ingestion usando el endpoint REST
            url = f"{self.ingestion_url}/api/market-data/{symbol}"
            session = await self._get_aiohttp_session()
            
            # Realizar solicitud con reintentos
            for retry in range(self.max_retries):
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Reiniciar contador de errores
                            self.ingestion_error_count = 0
                            
                            # Guardar en caché
                            self.redis.set(cache_key, data, ttl=self.cache_ttl["market_data"])
                            
                            return data
                        elif response.status == 426:
                            # Error específico: Upgrade Required - significa que estamos intentando usar HTTP donde se espera WebSocket
                            logger.warning(f"Error 426: El endpoint de ingestion requiere actualización de protocolo. Verificar URL: {url}")
                            self.ingestion_error_count += 1
                            break  # Salir del bucle para pasar directo al fallback
                        elif response.status == 404:
                            # El API endpoint podría no ser correcto
                            logger.warning(f"Error 404: Endpoint no encontrado. Verificar URL: {url}")
                            # Intentar con una URL alternativa sin el prefijo /api
                            url = f"{self.ingestion_url}/market-data/{symbol}"
                            # Continuamos para probar con la URL alternativa
                        else:
                            logger.warning(f"Error obteniendo datos de mercado para {symbol} desde ingestion: {response.status}")
                            self.ingestion_error_count += 1
                            
                            # Si no es el último intento, esperar antes de reintentar
                            if retry < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay * (retry + 1))
                            
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"Error de conexión al obtener datos de mercado para {symbol}: {e}")
                    self.ingestion_error_count += 1
                    
                    # Si no es el último intento, esperar antes de reintentar
                    if retry < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (retry + 1))
            
            # Si acumulamos demasiados errores, cambiar a modo FMP directo
            if self.ingestion_error_count >= self.max_ingestion_errors:
                logger.warning(f"Cambiando a modo FMP directo después de {self.ingestion_error_count} errores con ingestion")
                self.direct_fmp_mode = True
            
            # Intentar una verificación de salud del servicio para diagnosticar problemas
            try:
                health_url = f"{self.ingestion_url}/api/health"
                async with session.get(health_url) as health_response:
                    if health_response.status == 200:
                        health_data = await health_response.json()
                        logger.info(f"Estado del servicio de ingestion: {health_data.get('status', 'unknown')}, versión: {health_data.get('version', 'unknown')}")
                    else:
                        logger.warning(f"No se pudo obtener el estado del servicio de ingestion: {health_response.status}")
            except Exception as e:
                logger.warning(f"Error al verificar la salud del servicio de ingestion: {e}")
            
            # Intentar fallback a FMP API si hay una clave disponible
            if self.fmp_api_key:
                fallback_data = await self._get_market_data_fallback(symbol)
                if fallback_data:
                    # Guardar en caché con TTL más corto (datos de fallback)
                    self.redis.set(cache_key, fallback_data, ttl=self.cache_ttl["market_data"] // 2)
                    return fallback_data
            
            # Si todo falló, devolver un diccionario vacío
            return {}
                    
        except Exception as e:
            logger.error(f"Error inesperado al obtener datos de mercado para {symbol}: {e}")
            return {}
    
    async def _get_market_data_fallback(self, symbol: str) -> Dict[str, Any]:
        """
        Obtiene datos de mercado desde FMP API como fallback
        
        Args:
            symbol: Símbolo del activo
        
        Returns:
            Datos de mercado para el símbolo
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={self.fmp_api_key}"
            session = await self._get_aiohttp_session()
            
            async with session.get(url) as response:
                if response.status == 200:
                    quotes = await response.json()
                    if quotes and len(quotes) > 0:
                        quote = quotes[0]
                        return {
                            "symbol": symbol,
                            "price": quote.get("price", 0),
                            "change": quote.get("change", 0),
                            "volume": quote.get("volume", 0),
                            "timestamp": datetime.now().isoformat(),
                            "source": "fmp_fallback"
                        }
            
            # Si no hay datos, devolver un diccionario vacío
            return {}
                
        except Exception as e:
            logger.error(f"Error obteniendo datos de mercado fallback para {symbol}: {e}")
            return {}
    
    async def get_historical_data(self, symbol: str, timeframe: str = "1d", limit: int = 30) -> List[Dict[str, Any]]:
        """
        Obtiene datos históricos para un símbolo
        
        Args:
            symbol: Símbolo del activo
            timeframe: Marco temporal (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            limit: Número máximo de registros
        
        Returns:
            Lista de datos históricos
        """
        # Construir clave para Redis
        cache_key = f"historical:{symbol}:{timeframe}:{limit}"
        
        # Intentar obtener desde caché
        cached_data = self.redis.get(cache_key)
        if cached_data:
            logger.debug(f"Datos históricos para {symbol} obtenidos desde caché")
            return cached_data
        
        try:
            # Intentar obtener desde el servicio de ingestion
            url = f"{self.ingestion_url}/api/historical/{symbol}?timeframe={timeframe}&limit={limit}"
            session = await self._get_aiohttp_session()
            
            # Realizar solicitud con reintentos
            for retry in range(self.max_retries):
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Guardar en caché
                            self.redis.set(cache_key, data, ttl=self.cache_ttl["historical"])
                            
                            return data
                        elif response.status == 404:
                            # El API endpoint podría no ser correcto
                            logger.warning(f"Error 404: Endpoint no encontrado. Verificar URL: {url}")
                            # Intentar con una URL alternativa sin el prefijo /api
                            url = f"{self.ingestion_url}/historical/{symbol}?timeframe={timeframe}&limit={limit}"
                            continue  # Probar con la URL alternativa
                        else:
                            logger.warning(f"Error obteniendo datos históricos para {symbol} desde ingestion: {response.status}")
                            
                            # Si no es el último intento, esperar antes de reintentar
                            if retry < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay * (retry + 1))
                            
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"Error de conexión al obtener datos históricos para {symbol}: {e}")
                    
                    # Si no es el último intento, esperar antes de reintentar
                    if retry < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (retry + 1))
            
            # Si llegamos aquí, fallaron todos los intentos
            # Intentar fallback a FMP API si hay una clave disponible
            if self.fmp_api_key:
                fallback_data = await self._get_historical_data_fallback(symbol, timeframe, limit)
                if fallback_data:
                    # Guardar en caché con TTL más corto (datos de fallback)
                    self.redis.set(cache_key, fallback_data, ttl=self.cache_ttl["historical"] // 2)
                    return fallback_data
            
            # Si todo falló, devolver una lista vacía
            return []
                    
        except Exception as e:
            logger.error(f"Error inesperado al obtener datos históricos para {symbol}: {e}")
            return []
    
    async def _get_historical_data_fallback(self, symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        """
        Obtiene datos históricos desde FMP API como fallback
        
        Args:
            symbol: Símbolo del activo
            timeframe: Marco temporal (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            limit: Número máximo de registros
        
        Returns:
            Lista de datos históricos
        """
        try:
            # Mapear timeframe a formato FMP API
            fmp_timeframe = "1min" if timeframe == "1m" else \
                           "5min" if timeframe == "5m" else \
                           "15min" if timeframe == "15m" else \
                           "30min" if timeframe == "30m" else \
                           "1hour" if timeframe == "1h" else \
                           "4hour" if timeframe == "4h" else "1day"
            
            url = f"https://financialmodelingprep.com/api/v3/historical-chart/{fmp_timeframe}/{symbol}?apikey={self.fmp_api_key}&limit={limit}"
            session = await self._get_aiohttp_session()
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Formatear datos para que coincidan con lo que espera el frontend
                    formatted_data = []
                    for item in data:
                        formatted_data.append({
                            "date": item.get("date"),
                            "open": float(item.get("open", 0)),
                            "high": float(item.get("high", 0)),
                            "low": float(item.get("low", 0)),
                            "close": float(item.get("close", 0)),
                            "volume": int(item.get("volume", 0)),
                            "source": "fmp_fallback"
                        })
                    
                    return formatted_data
            
            # Si no hay datos, devolver una lista vacía
            return []
                
        except Exception as e:
            logger.error(f"Error obteniendo datos históricos fallback para {symbol}: {e}")
            return []
    
    async def get_streaming_prediction(self, symbol: str) -> Dict[str, Any]:
        """
        Obtiene predicción en tiempo real para un símbolo
        
        Args:
            symbol: Símbolo del activo
        
        Returns:
            Predicción del modelo
        """
        # Construir clave para Redis
        cache_key = f"prediction:{symbol}"
        
        # Intentar obtener desde caché
        cached_data = self.redis.get(cache_key)
        if cached_data:
            # Verificar si la predicción está fresca (menos de 5 minutos)
            if "timestamp" in cached_data:
                cached_time = datetime.fromisoformat(cached_data["timestamp"])
                if (datetime.now() - cached_time) < timedelta(minutes=5):
                    logger.debug(f"Predicción para {symbol} obtenida desde caché")
                    return cached_data
        
        try:
            # Intentar obtener desde el servicio de streaming
            url = f"{self.streaming_url}/prediction/{symbol}"
            session = await self._get_aiohttp_session()
            
            # Realizar solicitud con reintentos
            for retry in range(self.max_retries):
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Añadir timestamp si no lo tiene
                            if "timestamp" not in data:
                                data["timestamp"] = datetime.now().isoformat()
                            
                            # Guardar en caché
                            self.redis.set(cache_key, data, ttl=self.cache_ttl["prediction"])
                            
                            return data
                        else:
                            logger.warning(f"Error obteniendo predicción para {symbol} desde streaming: {response.status}")
                            
                            # Si no es el último intento, esperar antes de reintentar
                            if retry < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay * (retry + 1))
                            
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"Error de conexión al obtener predicción para {symbol}: {e}")
                    
                    # Si no es el último intento, esperar antes de reintentar
                    if retry < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (retry + 1))
            
            # Si llegamos aquí, fallaron todos los intentos
            # Devolver una predicción por defecto
            default_prediction = {
                "symbol": symbol,
                "prediction": 0,
                "confidence": 0,
                "direction": "neutral",
                "timestamp": datetime.now().isoformat(),
                "source": "default"
            }
            
            return default_prediction
                    
        except Exception as e:
            logger.error(f"Error inesperado al obtener predicción para {symbol}: {e}")
            return {
                "symbol": symbol,
                "prediction": 0,
                "confidence": 0,
                "direction": "neutral",
                "timestamp": datetime.now().isoformat(),
                "source": "error"
            }
    
    async def get_symbols(self) -> List[str]:
        """
        Obtiene lista de símbolos disponibles
        
        Returns:
            Lista de símbolos
        """
        # Construir clave para Redis
        cache_key = "symbols:list"
        
        # Intentar obtener desde caché
        cached_data = self.redis.get(cache_key)
        if cached_data:
            logger.debug("Lista de símbolos obtenida desde caché")
            return cached_data
        
        try:
            # Intentar obtener desde el servicio de ingestion
            url = f"{self.ingestion_url}/symbols"
            session = await self._get_aiohttp_session()
            
            # Realizar solicitud con reintentos
            for retry in range(self.max_retries):
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            symbols = data.get("symbols", [])
                            
                            # Guardar en caché
                            self.redis.set(cache_key, symbols, ttl=self.cache_ttl["symbol_list"])
                            
                            return symbols
                        else:
                            logger.warning(f"Error obteniendo lista de símbolos desde ingestion: {response.status}")
                            
                            # Si no es el último intento, esperar antes de reintentar
                            if retry < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay * (retry + 1))
                            
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"Error de conexión al obtener lista de símbolos: {e}")
                    
                    # Si no es el último intento, esperar antes de reintentar
                    if retry < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (retry + 1))
            
            # Si llegamos aquí, fallaron todos los intentos
            # Devolver una lista predeterminada de símbolos populares
            default_symbols = [
                "AAPL", "MSFT", "GOOG", "AMZN", "TSLA", 
                "META", "NVDA", "NFLX", "INTC", "AMD",
                "IAG.MC", "PHM.MC", "AENA.MC", "BA", 
                "CAR", "DLTR", "SASA.IS"
            ]
            
            return default_symbols
                    
        except Exception as e:
            logger.error(f"Error inesperado al obtener lista de símbolos: {e}")
            # Devolver una lista predeterminada de símbolos populares
            return [
                "AAPL", "MSFT", "GOOG", "AMZN", "TSLA", 
                "META", "NVDA", "NFLX", "INTC", "AMD"
            ]

    async def get_company_news(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtiene noticias recientes relacionadas con un símbolo utilizando la API de FMP
        
        Args:
            symbol: Símbolo del activo
            limit: Número máximo de noticias a obtener
            
        Returns:
            Lista de noticias
        """
        # Construir clave para Redis
        cache_key = f"news:{symbol}:{limit}"
        
        # Intentar obtener desde caché
        cached_data = self.redis.get(cache_key)
        if cached_data:
            logger.debug(f"Noticias para {symbol} obtenidas desde caché")
            return cached_data
        
        try:
            # Intentar obtener desde FMP API
            if not self.fmp_api_key:
                return []
                
            url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={symbol}&limit={limit}&apikey={self.fmp_api_key}"
            session = await self._get_aiohttp_session()
            
            async with session.get(url) as response:
                if response.status == 200:
                    news_data = await response.json()
                    
                    # Formatear datos para uniformidad
                    formatted_news = []
                    for item in news_data:
                        formatted_news.append({
                            "symbol": item.get("symbol", symbol),
                            "title": item.get("title", ""),
                            "text": item.get("text", ""),
                            "source": item.get("site", ""),
                            "url": item.get("url", ""),
                            "image": item.get("image", ""),
                            "date": item.get("publishedDate", ""),
                            "sentiment": self._calculate_news_sentiment(item.get("title", "") + " " + item.get("text", ""))
                        })
                    
                    # Guardar en caché
                    self.redis.set(cache_key, formatted_news, ttl=self.cache_ttl["news"])
                    
                    return formatted_news
                else:
                    logger.warning(f"Error obteniendo noticias para {symbol} desde FMP: {response.status}")
                    return []
                
        except Exception as e:
            logger.error(f"Error inesperado al obtener noticias para {symbol}: {e}")
            return []
    
    def _calculate_news_sentiment(self, text: str) -> Dict[str, float]:
        """
        Calcula un sentimiento básico basado en palabras clave en el texto
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con puntuaciones de sentimiento
        """
        # Lista de palabras positivas y negativas comunes en finanzas
        positive_words = [
            "up", "rise", "growth", "profit", "gain", "positive", "success", "beat", "exceed", 
            "strong", "improve", "opportunity", "bullish", "outperform", "boost", "optimistic",
            "recovery", "overcome", "advantage", "momentum", "promising", "surge", "rally"
        ]
        
        negative_words = [
            "down", "fall", "drop", "loss", "negative", "fail", "miss", "weak", "decline", 
            "decrease", "bearish", "underperform", "concern", "risk", "warning", "volatility",
            "caution", "pressure", "struggle", "challenge", "uncertain", "threat", "crash"
        ]
        
        text = text.lower()
        pos_count = sum(1 for word in positive_words if f" {word} " in f" {text} ")
        neg_count = sum(1 for word in negative_words if f" {word} " in f" {text} ")
        
        total = pos_count + neg_count
        if total == 0:
            return {"positive": 0.5, "negative": 0.5, "score": 0}
        
        positive_ratio = pos_count / total
        negative_ratio = neg_count / total
        score = (positive_ratio - negative_ratio) * 100  # De -100 a 100
        
        return {
            "positive": round(positive_ratio, 2),
            "negative": round(negative_ratio, 2),
            "score": round(score, 2)
        }
    
    async def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """
        Obtiene el perfil de la empresa para un símbolo usando la API de FMP
        
        Args:
            symbol: Símbolo del activo
            
        Returns:
            Datos del perfil de la empresa
        """
        # Construir clave para Redis
        cache_key = f"profile:{symbol}"
        
        # Intentar obtener desde caché
        cached_data = self.redis.get(cache_key)
        if cached_data:
            logger.debug(f"Perfil para {symbol} obtenido desde caché")
            return cached_data
        
        try:
            # Intentar obtener desde FMP API
            if not self.fmp_api_key:
                return {}
                
            url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={self.fmp_api_key}"
            session = await self._get_aiohttp_session()
            
            async with session.get(url) as response:
                if response.status == 200:
                    profiles = await response.json()
                    if profiles and len(profiles) > 0:
                        profile = profiles[0]
                        
                        # Formatear datos para uniformidad
                        formatted_profile = {
                            "symbol": profile.get("symbol", symbol),
                            "name": profile.get("companyName", ""),
                            "exchange": profile.get("exchange", ""),
                            "industry": profile.get("industry", ""),
                            "sector": profile.get("sector", ""),
                            "description": profile.get("description", ""),
                            "employees": profile.get("fullTimeEmployees", 0),
                            "ceo": profile.get("ceo", ""),
                            "website": profile.get("website", ""),
                            "marketCap": profile.get("mktCap", 0),
                            "price": profile.get("price", 0),
                            "changes": profile.get("changes", 0),
                            "currency": profile.get("currency", "USD"),
                            "country": profile.get("country", ""),
                            "ipoDate": profile.get("ipoDate", ""),
                            "logo": profile.get("image", ""),
                            "isActive": profile.get("isActivelyTrading", True)
                        }
                        
                        # Guardar en caché
                        self.redis.set(cache_key, formatted_profile, ttl=self.cache_ttl["profile"])
                        
                        return formatted_profile
                
                logger.warning(f"Error obteniendo perfil para {symbol} desde FMP: {response.status}")
                return {}
                
        except Exception as e:
            logger.error(f"Error inesperado al obtener perfil para {symbol}: {e}")
            return {}
    
    async def get_key_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Obtiene métricas clave financieras para un símbolo usando la API de FMP
        
        Args:
            symbol: Símbolo del activo
            
        Returns:
            Métricas financieras clave
        """
        # Construir clave para Redis
        cache_key = f"ratios:{symbol}"
        
        # Intentar obtener desde caché
        cached_data = self.redis.get(cache_key)
        if cached_data:
            logger.debug(f"Métricas para {symbol} obtenidas desde caché")
            return cached_data
        
        try:
            # Intentar obtener desde FMP API
            if not self.fmp_api_key:
                return {}
                
            url = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}?limit=1&apikey={self.fmp_api_key}"
            session = await self._get_aiohttp_session()
            
            async with session.get(url) as response:
                if response.status == 200:
                    metrics_data = await response.json()
                    if metrics_data and len(metrics_data) > 0:
                        metrics = metrics_data[0]
                        
                        # Formatear datos para uniformidad y seleccionar métricas relevantes
                        formatted_metrics = {
                            "symbol": symbol,
                            "date": metrics.get("date", ""),
                            "pe_ratio": metrics.get("priceEarningsRatio", 0),
                            "price_to_book": metrics.get("priceToBookRatio", 0),
                            "dividend_yield": metrics.get("dividendYield", 0),
                            "debt_to_equity": metrics.get("debtToEquity", 0),
                            "roe": metrics.get("returnOnEquity", 0),
                            "roa": metrics.get("returnOnAssets", 0),
                            "current_ratio": metrics.get("currentRatio", 0),
                            "quick_ratio": metrics.get("quickRatio", 0),
                            "gross_margin": metrics.get("grossProfitMargin", 0),
                            "operating_margin": metrics.get("operatingProfitMargin", 0),
                            "net_margin": metrics.get("netProfitMargin", 0),
                            "eps": metrics.get("ebitPerShare", 0)
                        }
                        
                        # Guardar en caché
                        self.redis.set(cache_key, formatted_metrics, ttl=self.cache_ttl["ratios"])
                        
                        return formatted_metrics
                
                logger.warning(f"Error obteniendo métricas para {symbol} desde FMP: {response.status}")
                return {}
                
        except Exception as e:
            logger.error(f"Error inesperado al obtener métricas para {symbol}: {e}")
            return {}
    
    async def get_earnings_calendar(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Obtiene calendario de ganancias usando la API de FMP
        
        Args:
            symbol: Símbolo del activo (opcional)
            
        Returns:
            Lista de eventos de calendario de ganancias
        """
        # Construir clave para Redis
        cache_key = f"earnings:{symbol if symbol else 'all'}"
        
        # Intentar obtener desde caché
        cached_data = self.redis.get(cache_key)
        if cached_data:
            logger.debug(f"Calendario de ganancias obtenido desde caché")
            return cached_data
        
        try:
            # Intentar obtener desde FMP API
            if not self.fmp_api_key:
                return []
            
            # URL para un símbolo específico o todos los símbolos
            if symbol:
                url = f"https://financialmodelingprep.com/api/v3/earnings-calendar/{symbol}?apikey={self.fmp_api_key}"
            else:
                url = f"https://financialmodelingprep.com/api/v3/earnings-calendar?apikey={self.fmp_api_key}"
                
            session = await self._get_aiohttp_session()
            
            async with session.get(url) as response:
                if response.status == 200:
                    earnings_data = await response.json()
                    
                    # Guardar en caché
                    self.redis.set(cache_key, earnings_data, ttl=self.cache_ttl["earnings"])
                    
                    return earnings_data
                
                logger.warning(f"Error obteniendo calendario de ganancias desde FMP: {response.status}")
                return []
                
        except Exception as e:
            logger.error(f"Error inesperado al obtener calendario de ganancias: {e}")
            return []
    
    async def get_market_sentiment(self, symbol: str = None) -> Dict[str, Any]:
        """
        Analiza el sentimiento de mercado basado en noticias y datos técnicos
        
        Args:
            symbol: Símbolo del activo (opcional)
            
        Returns:
            Análisis de sentimiento del mercado
        """
        # Si se proporciona un símbolo, analizamos el sentimiento específico
        if (symbol):
            # Construir clave para Redis
            cache_key = f"sentiment:{symbol}"
            
            # Intentar obtener desde caché
            cached_data = self.redis.get(cache_key)
            if cached_data:
                logger.debug(f"Sentimiento para {symbol} obtenido desde caché")
                return cached_data
            
            try:
                # Recopilar datos para análisis de sentimiento
                news = await self.get_company_news(symbol, limit=5)
                market_data = await self.get_market_data(symbol)
                historical_data = await self.get_historical_data(symbol, timeframe="1d", limit=10)
                
                # Inicializar resultado
                sentiment = {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "overall_score": 0,
                    "news_sentiment": 0,
                    "technical_sentiment": 0,
                    "volume_sentiment": 0,
                    "recommendation": "NEUTRAL",
                    "details": {}
                }
                
                # Analizar sentimiento de noticias
                if news:
                    news_scores = [item.get("sentiment", {}).get("score", 0) for item in news if "sentiment" in item]
                    if news_scores:
                        sentiment["news_sentiment"] = round(sum(news_scores) / len(news_scores), 2)
                
                # Analizar datos técnicos (tendencia reciente)
                if historical_data and len(historical_data) > 1:
                    # Calcular tendencia de precios
                    oldest_price = historical_data[-1].get("close", 0)
                    newest_price = historical_data[0].get("close", 0)
                    if oldest_price > 0:
                        price_change_pct = ((newest_price - oldest_price) / oldest_price) * 100
                        sentiment["technical_sentiment"] = round(price_change_pct * 5, 2)  # Escalar para que tenga más influencia
                        
                        # Añadir detalles
                        sentiment["details"]["price_change_pct"] = round(price_change_pct, 2)
                
                # Analizar volumen
                if market_data and "volume" in market_data and historical_data and len(historical_data) > 1:
                    # Comparar volumen actual con promedio histórico
                    hist_volumes = [item.get("volume", 0) for item in historical_data]
                    avg_volume = sum(hist_volumes) / len(hist_volumes) if hist_volumes else 0
                    current_volume = float(market_data.get("volume", 0))
                    
                    if avg_volume > 0:
                        volume_ratio = current_volume / avg_volume
                        # Si volumen es mayor que el promedio y precio subió, es positivo
                        # Si volumen es mayor y precio bajó, es negativo
                        if sentiment["technical_sentiment"] > 0:
                            volume_sentiment = (volume_ratio - 1) * 30  # Factor de escala
                        else:
                            volume_sentiment = (1 - volume_ratio) * 30
                            
                        sentiment["volume_sentiment"] = round(max(min(volume_sentiment, 100), -100), 2)
                        
                        # Añadir detalles
                        sentiment["details"]["volume_ratio"] = round(volume_ratio, 2)
                
                # Calcular puntuación general (ponderada)
                weights = {
                    "news": 0.3,       # 30% basado en noticias
                    "technical": 0.5,   # 50% basado en tendencia de precios
                    "volume": 0.2       # 20% basado en volumen
                }
                
                overall_score = (
                    sentiment["news_sentiment"] * weights["news"] +
                    sentiment["technical_sentiment"] * weights["technical"] +
                    sentiment["volume_sentiment"] * weights["volume"]
                )
                
                sentiment["overall_score"] = round(overall_score, 2)
                
                # Determinar recomendación
                if sentiment["overall_score"] > 30:
                    sentiment["recommendation"] = "STRONG_BUY"
                elif sentiment["overall_score"] > 10:
                    sentiment["recommendation"] = "BUY"
                elif sentiment["overall_score"] < -30:
                    sentiment["recommendation"] = "STRONG_SELL"
                elif sentiment["overall_score"] < -10:
                    sentiment["recommendation"] = "SELL"
                else:
                    sentiment["recommendation"] = "NEUTRAL"
                
                # Guardar en caché
                self.redis.set(cache_key, sentiment, ttl=self.cache_ttl["sentiment"])
                
                return sentiment
                
            except Exception as e:
                logger.error(f"Error inesperado al analizar sentimiento para {symbol}: {e}")
                return {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "overall_score": 0,
                    "recommendation": "NEUTRAL",
                    "error": str(e)
                }
        else:
            # Análisis general del mercado
            # Aquí podríamos implementar un análisis de índices principales
            # o de múltiples símbolos populares
            return {
                "timestamp": datetime.now().isoformat(),
                "message": "El análisis de sentimiento general del mercado no está implementado aún."
            }

# Instancia global del cliente de datos
_data_client_instance = None

def get_data_client() -> DataClient:
    """
    Obtiene una instancia única del cliente de datos
    
    Returns:
        Instancia de DataClient
    """
    global _data_client_instance
    
    if _data_client_instance is None:
        _data_client_instance = DataClient()
    
    return _data_client_instance

async def close_data_client():
    """Cierra el cliente de datos"""
    global _data_client_instance
    
    if _data_client_instance:
        await _data_client_instance.close()
        _data_client_instance = None