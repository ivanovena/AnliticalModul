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
        
        # Timeouts y reintentos
        self.request_timeout = 5  # segundos
        self.max_retries = 3
        self.retry_delay = 1  # segundos
        
        # TTLs de caché (tiempo de vida en segundos)
        self.cache_ttl = {
            "market_data": 300,  # 5 minutos
            "historical": 3600,  # 1 hora
            "prediction": 600,   # 10 minutos
            "symbol_list": 3600  # 1 hora
        }
        
        # Obtener instancia de Redis
        self.redis = get_redis_cache()
        self.session = None
    
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
            # Intentar obtener desde el servicio de ingestion
            url = f"{self.ingestion_url}/market-data/{symbol}"
            session = await self._get_aiohttp_session()
            
            # Realizar solicitud con reintentos
            for retry in range(self.max_retries):
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Guardar en caché
                            self.redis.set(cache_key, data, ttl=self.cache_ttl["market_data"])
                            
                            return data
                        else:
                            logger.warning(f"Error obteniendo datos de mercado para {symbol} desde ingestion: {response.status}")
                            
                            # Si no es el último intento, esperar antes de reintentar
                            if retry < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delay * (retry + 1))
                            
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"Error de conexión al obtener datos de mercado para {symbol}: {e}")
                    
                    # Si no es el último intento, esperar antes de reintentar
                    if retry < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (retry + 1))
            
            # Si llegamos aquí, fallaron todos los intentos
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
            url = f"{self.ingestion_url}/historical/{symbol}?timeframe={timeframe}&limit={limit}"
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