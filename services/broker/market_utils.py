"""
Utilidades para análisis de mercado y procesamiento de datos financieros
"""
import numpy as np
import logging
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("market_utils.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MarketUtils")

class MarketAnalyzer:
    """Clase para análisis técnico y fundamental de datos de mercado"""
    
    def __init__(self, api_key: str = None):
        """
        Inicializar analizador de mercado
        
        Args:
            api_key: API key para acceder a datos de mercado
        """
        self.api_key = api_key or "h5JPnHPAdjxBAXAGwTOL3Acs3W5zaByx"  # Default FMP API key
        self.indicators_cache = {}  # Cache para indicadores técnicos
        self.fundamental_cache = {}  # Cache para datos fundamentales
        
    def get_historical_prices(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Obtener precios históricos para un símbolo
        
        Args:
            symbol: Símbolo a consultar
            days: Número de días de historia
            
        Returns:
            Lista de precios históricos ordenados por fecha
        """
        try:
            # Calcular fecha de inicio
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Formatear fechas
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Consultar API
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_str}&to={end_str}&apikey={self.api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "historical" in data:
                    # Ordenar por fecha
                    historical = sorted(data["historical"], key=lambda x: x["date"])
                    return historical
                else:
                    logger.warning(f"No hay datos históricos para {symbol}")
                    return []
            else:
                logger.warning(f"Error obteniendo datos históricos: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error en get_historical_prices: {e}")
            return []
    
    def calculate_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """
        Calcular indicadores técnicos para un símbolo
        
        Args:
            symbol: Símbolo a analizar
            
        Returns:
            Diccionario con indicadores técnicos
        """
        # Verificar si hay datos en caché y son recientes
        cache_key = f"{symbol}_tech"
        if cache_key in self.indicators_cache:
            cache_time, indicators = self.indicators_cache[cache_key]
            # Usar caché si es menos de 1 hora
            if datetime.now() - cache_time < timedelta(hours=1):
                return indicators
        
        # Obtener datos históricos
        historical = self.get_historical_prices(symbol, days=30)
        
        if not historical:
            logger.warning(f"No hay suficientes datos para calcular indicadores de {symbol}")
            return {}
        
        try:
            # Extraer precios
            prices = np.array([float(item["close"]) for item in historical])
            volumes = np.array([float(item["volume"]) for item in historical])
            dates = [item["date"] for item in historical]
            
            # Calcular indicadores
            indicators = {}
            
            # SMA (Simple Moving Average) para 7 y 20 días
            if len(prices) >= 7:
                indicators["sma_7"] = np.mean(prices[-7:])
            if len(prices) >= 20:
                indicators["sma_20"] = np.mean(prices[-20:])
            
            # EMA (Exponential Moving Average)
            if len(prices) >= 14:
                # Simplificación de EMA - en producción usar pandas o similar
                alpha = 2 / (14 + 1)
                ema = prices[-14]
                for price in prices[-13:]:
                    ema = alpha * price + (1 - alpha) * ema
                indicators["ema_14"] = ema
            
            # RSI (Relative Strength Index)
            if len(prices) >= 15:
                # Calcular cambios en precio
                changes = np.diff(prices[-15:])
                gains = np.sum(np.maximum(changes, 0))
                losses = np.sum(np.abs(np.minimum(changes, 0)))
                
                # Calcular RS y RSI
                if losses > 0:
                    rs = gains / losses
                    indicators["rsi_14"] = 100 - (100 / (1 + rs))
                else:
                    indicators["rsi_14"] = 100  # Solo ganancias
            
            # Volumen promedio
            indicators["avg_volume"] = np.mean(volumes[-10:]) if len(volumes) >= 10 else 0
            
            # Tendencia (simple)
            if len(prices) >= 5:
                # Último precio vs promedio de 5 días previos
                last_price = prices[-1]
                prev_avg = np.mean(prices[-6:-1])
                indicators["trend"] = "up" if last_price > prev_avg else "down"
                indicators["trend_strength"] = abs(last_price - prev_avg) / prev_avg * 100
            
            # Señales simples
            if "sma_7" in indicators and "sma_20" in indicators:
                indicators["signal"] = "buy" if indicators["sma_7"] > indicators["sma_20"] else "sell"
            
            if "rsi_14" in indicators:
                # Sobrecompra/sobreventa
                if indicators["rsi_14"] > 70:
                    indicators["rsi_signal"] = "overbought"
                elif indicators["rsi_14"] < 30:
                    indicators["rsi_signal"] = "oversold"
                else:
                    indicators["rsi_signal"] = "neutral"
            
            # Almacenar en caché
            self.indicators_cache[cache_key] = (datetime.now(), indicators)
            
            return indicators
        except Exception as e:
            logger.error(f"Error calculando indicadores para {symbol}: {e}")
            return {}
    
    def get_price_prediction(self, symbol: str, indicators: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generar predicción simple basada en indicadores técnicos
        
        Args:
            symbol: Símbolo a analizar
            indicators: Indicadores técnicos pre-calculados (opcional)
            
        Returns:
            Predicción de precio
        """
        # Usar indicadores proporcionados o calcularlos
        if not indicators:
            indicators = self.calculate_technical_indicators(symbol)
        
        if not indicators:
            logger.warning(f"No hay indicadores disponibles para {symbol}")
            return {
                "prediction": 0,
                "confidence": 0,
                "direction": "neutral",
                "factors": []
            }
        
        try:
            # Factores que influyen en la predicción
            factors = []
            
            # Tendencia general
            trend_score = 0
            if "trend" in indicators:
                trend_factor = 0.5 if indicators["trend"] == "up" else -0.5
                trend_score += trend_factor
                factors.append({"name": "trend", "impact": trend_factor})
            
            # RSI
            rsi_score = 0
            if "rsi_14" in indicators:
                rsi = indicators["rsi_14"]
                if rsi > 70:
                    rsi_score = -0.3  # Sobrecompra - señal de venta
                    factors.append({"name": "rsi_overbought", "impact": -0.3})
                elif rsi < 30:
                    rsi_score = 0.3  # Sobreventa - señal de compra
                    factors.append({"name": "rsi_oversold", "impact": 0.3})
            
            # Medias móviles
            ma_score = 0
            if "sma_7" in indicators and "sma_20" in indicators:
                sma_7 = indicators["sma_7"]
                sma_20 = indicators["sma_20"]
                
                if sma_7 > sma_20:
                    # Cruce alcista
                    ma_score = 0.4
                    factors.append({"name": "golden_cross", "impact": 0.4})
                else:
                    # Cruce bajista
                    ma_score = -0.4
                    factors.append({"name": "death_cross", "impact": -0.4})
            
            # Combinar puntuaciones
            total_score = trend_score + rsi_score + ma_score
            
            # Normalizar a un rango +/- 5%
            prediction_pct = total_score * 5  # Escalar para que esté entre -5% y +5%
            
            # Dirección y confianza
            direction = "up" if prediction_pct > 0 else "down" if prediction_pct < 0 else "neutral"
            confidence = min(abs(total_score) + 0.3, 0.9)  # Confianza entre 0.3 y 0.9
            
            return {
                "prediction": prediction_pct,
                "confidence": confidence,
                "direction": direction,
                "factors": factors
            }
        except Exception as e:
            logger.error(f"Error generando predicción para {symbol}: {e}")
            return {
                "prediction": 0,
                "confidence": 0,
                "direction": "neutral",
                "factors": []
            }
    
    def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """
        Obtener datos fundamentales para un símbolo
        
        Args:
            symbol: Símbolo a consultar
            
        Returns:
            Datos fundamentales
        """
        # Verificar si hay datos en caché y son recientes
        cache_key = f"{symbol}_fund"
        if cache_key in self.fundamental_cache:
            cache_time, data = self.fundamental_cache[cache_key]
            # Usar caché si es menos de 24 horas
            if datetime.now() - cache_time < timedelta(hours=24):
                return data
        
        try:
            # Obtener perfil de la empresa
            url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={self.api_key}"
            response = requests.get(url, timeout=10)
            
            profile_data = {}
            if response.status_code == 200:
                profiles = response.json()
                if profiles:
                    profile = profiles[0]
                    profile_data = {
                        "name": profile.get("companyName", ""),
                        "sector": profile.get("sector", ""),
                        "industry": profile.get("industry", ""),
                        "market_cap": profile.get("mktCap", 0),
                        "beta": profile.get("beta", 0),
                        "price": profile.get("price", 0),
                        "change": profile.get("changes", 0),
                        "exchange": profile.get("exchange", "")
                    }
            
            # Obtener métricas financieras
            url = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}?limit=1&apikey={self.api_key}"
            response = requests.get(url, timeout=10)
            
            ratios_data = {}
            if response.status_code == 200:
                ratios = response.json()
                if ratios:
                    ratio = ratios[0]
                    ratios_data = {
                        "pe_ratio": ratio.get("priceEarningsRatio", 0),
                        "pb_ratio": ratio.get("priceToBookRatio", 0),
                        "dividend_yield": ratio.get("dividendYield", 0),
                        "roe": ratio.get("returnOnEquity", 0),
                        "current_ratio": ratio.get("currentRatio", 0)
                    }
            
            # Combinar datos
            fundamental_data = {**profile_data, **ratios_data}
            
            # Almacenar en caché
            self.fundamental_cache[cache_key] = (datetime.now(), fundamental_data)
            
            return fundamental_data
        except Exception as e:
            logger.error(f"Error obteniendo datos fundamentales para {symbol}: {e}")
            return {}
    
    def get_market_sentiment(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Obtener sentimiento general del mercado o para un conjunto de símbolos
        
        Args:
            symbols: Lista de símbolos (opcional)
            
        Returns:
            Datos de sentimiento de mercado
        """
        try:
            # Si no se especifican símbolos, usar índices principales
            if not symbols:
                symbols = ["SPY", "QQQ", "DIA"]
            
            # Obtener indicadores técnicos para cada símbolo
            sentiment_data = {}
            
            for symbol in symbols:
                indicators = self.calculate_technical_indicators(symbol)
                if indicators:
                    # Determinar sentimiento basado en RSI y tendencia
                    sentiment = "neutral"
                    if "rsi_14" in indicators:
                        if indicators["rsi_14"] > 70:
                            sentiment = "bearish"  # Sobrecompra - potencial caída
                        elif indicators["rsi_14"] < 30:
                            sentiment = "bullish"  # Sobreventa - potencial subida
                    
                    if "trend" in indicators:
                        # Si hay una tendencia fuerte, puede sobreescribir el RSI
                        if indicators["trend"] == "up" and indicators.get("trend_strength", 0) > 2:
                            sentiment = "bullish"
                        elif indicators["trend"] == "down" and indicators.get("trend_strength", 0) > 2:
                            sentiment = "bearish"
                    
                    sentiment_data[symbol] = {
                        "sentiment": sentiment,
                        "rsi": indicators.get("rsi_14", 0),
                        "trend": indicators.get("trend", "neutral"),
                        "signal": indicators.get("signal", "hold")
                    }
            
            # Calcular sentimiento general
            if sentiment_data:
                bullish_count = sum(1 for data in sentiment_data.values() if data["sentiment"] == "bullish")
                bearish_count = sum(1 for data in sentiment_data.values() if data["sentiment"] == "bearish")
                neutral_count = sum(1 for data in sentiment_data.values() if data["sentiment"] == "neutral")
                
                if bullish_count > bearish_count + neutral_count:
                    overall = "bullish"
                elif bearish_count > bullish_count + neutral_count:
                    overall = "bearish"
                else:
                    overall = "neutral"
                
                return {
                    "overall": overall,
                    "symbols": sentiment_data
                }
            else:
                return {
                    "overall": "neutral",
                    "symbols": {}
                }
        except Exception as e:
            logger.error(f"Error obteniendo sentimiento de mercado: {e}")
            return {
                "overall": "neutral",
                "symbols": {}
            }

# Singleton analyzer
market_analyzer = MarketAnalyzer()

def get_market_analyzer():
    """Obtener instancia singleton del analizador de mercado"""
    return market_analyzer
