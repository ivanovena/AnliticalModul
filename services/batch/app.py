import logging
import pandas as pd
import numpy as np
import datetime
import time
import json
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from kafka import KafkaProducer
import joblib
from config import DB_URI, KAFKA_BROKER, KAFKA_TOPIC, MODEL_OUTPUT_PATH

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("batch_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BatchService")

# Initialize SQLAlchemy engine with connection pooling
try:
    engine = create_engine(
        DB_URI, 
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_recycle=3600
    )
    logger.info("Database connection established")
except Exception as e:
    logger.critical(f"Failed to connect to database: {e}")
    raise

# Initialize Kafka producer with error handling
try:
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',
        retries=5,
        retry_backoff_ms=100
    )
    logger.info("Kafka producer initialized")
except Exception as e:
    logger.critical(f"Failed to initialize Kafka producer: {e}")
    raise

def load_data(symbol: str, max_records: int = 20000) -> pd.DataFrame:
    """
    Load market data for a symbol with enhanced error handling and data validation
    Limita la cantidad máxima de registros para evitar sobrecarga de memoria
    """
    query = f"""
    SELECT * FROM (
        SELECT * FROM market_data 
        WHERE symbol = :symbol 
        ORDER BY datetime DESC
        LIMIT :max_records
    ) as recent_data
    ORDER BY datetime ASC
    """
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(query), 
                conn, 
                params={"symbol": symbol, "max_records": max_records},
                parse_dates=["datetime"]
            )
        
        # Validate dataframe
        if df.empty:
            logger.error(f"No data found for {symbol}")
            return pd.DataFrame()
            
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values in {symbol} data: {missing_values[missing_values > 0]}")
            
            # Fill missing values appropriately
            df.fillna(method='ffill', inplace=True)  # Forward fill
            df.fillna(method='bfill', inplace=True)  # Then backward fill any remaining NAs
        
        # Ensure datetime is correctly parsed
        if not pd.api.types.is_datetime64_dtype(df['datetime']):
            try:
                df['datetime'] = pd.to_datetime(df['datetime'])
            except Exception as e:
                logger.error(f"Error converting datetime for {symbol}: {e}")
                return pd.DataFrame()
        
        # Sort by datetime to ensure chronological order
        df.sort_values('datetime', inplace=True)
        
        # Check for duplicate timestamps
        if df.duplicated('datetime').any():
            logger.warning(f"Duplicate timestamps found in {symbol} data. Removing duplicates.")
            df.drop_duplicates('datetime', keep='last', inplace=True)
        
        # Verify required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns for {symbol}. Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
            
        logger.info(f"Data loaded for {symbol}: {len(df)} records, date range: {df['datetime'].min()} to {df['datetime'].max()}")
        return df
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        return pd.DataFrame()

def engineer_features(df: pd.DataFrame, reduced_features: bool = True) -> pd.DataFrame:
    """
    Engineer advanced features for time series prediction
    Con opción para reducir la cantidad de features generadas y ahorrar memoria
    """
    if df.empty:
        return df
        
    try:
        # Copy to avoid modifying original
        feature_df = df.copy()
        
        # Technical indicators - versión reducida seleccionando solo las features más importantes
        
        # Price momentum (returns)
        feature_df['return_1d'] = feature_df['close'].pct_change(1)
        feature_df['return_5d'] = feature_df['close'].pct_change(5)
        
        if not reduced_features:
            feature_df['return_10d'] = feature_df['close'].pct_change(10)
        
        # Moving averages
        feature_df['ma_5'] = feature_df['close'].rolling(window=5).mean()
        feature_df['ma_10'] = feature_df['close'].rolling(window=10).mean()
        
        if not reduced_features:
            feature_df['ma_20'] = feature_df['close'].rolling(window=20).mean()
        
        # MA crossover signals
        feature_df['ma_5_10_crossover'] = (feature_df['ma_5'] > feature_df['ma_10']).astype(int)
        
        # Volatility indicators
        feature_df['volatility_5d'] = feature_df['return_1d'].rolling(window=5).std() * np.sqrt(5)
        
        if not reduced_features:
            feature_df['volatility_10d'] = feature_df['return_1d'].rolling(window=10).std() * np.sqrt(10)
        
        # Price channels - solo si no usamos versión reducida
        if not reduced_features:
            feature_df['high_5d'] = feature_df['high'].rolling(window=5).max()
            feature_df['low_5d'] = feature_df['low'].rolling(window=5).min()
            feature_df['channel_width_5d'] = (feature_df['high_5d'] - feature_df['low_5d']) / feature_df['close']
        
        # Volume indicators
        feature_df['volume_change'] = feature_df['volume'].pct_change(1)
        feature_df['volume_ma_5'] = feature_df['volume'].rolling(window=5).mean()
        
        if not reduced_features:
            feature_df['relative_volume'] = feature_df['volume'] / feature_df['volume_ma_5']
        
        # Bollinger Bands - versión simplificada
        feature_df['bb_middle'] = feature_df['close'].rolling(window=20).mean()
        feature_df['bb_std'] = feature_df['close'].rolling(window=20).std()
        
        if not reduced_features:
            feature_df['bb_upper'] = feature_df['bb_middle'] + 2 * feature_df['bb_std']
            feature_df['bb_lower'] = feature_df['bb_middle'] - 2 * feature_df['bb_std']
            feature_df['bb_width'] = (feature_df['bb_upper'] - feature_df['bb_lower']) / feature_df['bb_middle']
            feature_df['bb_position'] = (feature_df['close'] - feature_df['bb_lower']) / (feature_df['bb_upper'] - feature_df['bb_lower'])
        
        # RSI (simplified calculation)
        delta = feature_df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        feature_df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD - solo versión básica si usamos features reducidas
        feature_df['ema_12'] = feature_df['close'].ewm(span=12, adjust=False).mean()
        feature_df['ema_26'] = feature_df['close'].ewm(span=26, adjust=False).mean()
        feature_df['macd'] = feature_df['ema_12'] - feature_df['ema_26']
        
        if not reduced_features:
            feature_df['macd_signal'] = feature_df['macd'].ewm(span=9, adjust=False).mean()
            feature_df['macd_hist'] = feature_df['macd'] - feature_df['macd_signal']
        
        # OBV (On Balance Volume) - opcional para versión reducida
        if not reduced_features:
            feature_df['daily_ret'] = feature_df['close'].pct_change()
            feature_df['direction'] = np.where(feature_df['daily_ret'] > 0, 1, -1)
            feature_df['direction'] = np.where(feature_df['daily_ret'] == 0, 0, feature_df['direction'])
            feature_df['vol_adj'] = feature_df['volume'] * feature_df['direction']
            feature_df['obv'] = feature_df['vol_adj'].cumsum()
        
        # Candle patterns (simplified) - solo los más importantes
        feature_df['body_size'] = abs(feature_df['close'] - feature_df['open']) / feature_df['open']
        
        if not reduced_features:
            feature_df['upper_shadow'] = (feature_df['high'] - feature_df[['open', 'close']].max(axis=1)) / feature_df['open']
            feature_df['lower_shadow'] = (feature_df[['open', 'close']].min(axis=1) - feature_df['low']) / feature_df['open']
        
        feature_df['is_bullish'] = (feature_df['close'] > feature_df['open']).astype(int)
        
        # Time-based features - solo los más importantes
        feature_df['day_of_week'] = feature_df['datetime'].dt.dayofweek
        
        if not reduced_features:
            feature_df['month'] = feature_df['datetime'].dt.month
            feature_df['quarter'] = feature_df['datetime'].dt.quarter
        
        # Target variable - next day's close price (normalized as return)
        feature_df['target_next_return'] = feature_df['close'].pct_change(1).shift(-1)
        feature_df['target_next_close'] = feature_df['close'].shift(-1)
        
        # Manejar valores infinitos o muy grandes
        # Verificar columnas con valores extremos y limitarlos
        numeric_cols = feature_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            # Reemplazar infinitos con NaN
            mask = np.isinf(feature_df[col])
            if mask.any():
                feature_df.loc[mask, col] = np.nan
                
            # Calcular percentiles para identificar valores extremos
            if feature_df[col].count() > 0:  # Asegurarse de que hay suficientes datos
                q1 = feature_df[col].quantile(0.01)
                q3 = feature_df[col].quantile(0.99)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                
                # Limitar valores extremos
                feature_df[col] = feature_df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Drop rows with NaN values (from rolling windows)
        feature_df.dropna(inplace=True)
        
        # Liberar memoria eliminando columnas temporales
        del delta, gain, loss, rs
        if not reduced_features and 'daily_ret' in feature_df.columns:
            del feature_df['daily_ret'], feature_df['direction'], feature_df['vol_adj']
            
        # Forzar recolección de basura
        import gc
        gc.collect()
        
        logger.info(f"Engineered features created: {feature_df.shape[1]} features, {feature_df.shape[0]} rows")
        
        # Verificar que no haya valores infinitos antes de devolver
        # Verificar columna por columna para evitar problemas con tipos no numéricos
        for col in feature_df.select_dtypes(include=['float64', 'int64']).columns:
            mask = np.isinf(feature_df[col])
            if mask.any():
                logger.warning(f"Se encontraron valores infinitos en la columna {col}")
                feature_df.loc[mask, col] = np.nan
        
        # Eliminar filas con NaN después de reemplazar infinitos
        feature_df.dropna(inplace=True)
            
        return feature_df
    except Exception as e:
        logger.error(f"Error engineering features: {e}")
        return df

def train_models(df: pd.DataFrame, symbol: str) -> dict:
    """
    Train and evaluate models on the given dataset
    """
    if df.empty or df.shape[0] < 100:
        logger.error(f"No hay suficientes datos para entrenar modelos para {symbol}")
        return {"status": "error", "message": f"Datos insuficientes para {symbol}"}
        
    try:
        # Limpiar memoria no utilizada antes de empezar
        import gc
        gc.collect()
        
        # Preparar datos con reducción de memoria
        X = df.drop(['target_next_return', 'target_next_close', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'symbol'], axis=1, errors='ignore')
        y = df['target_next_close']
        
        # Verificar que no hay columnas categóricas o texto en el dataframe
        object_columns = X.select_dtypes(include=['object', 'string']).columns
        if not object_columns.empty:
            logger.warning(f"Eliminando columnas categóricas/texto: {list(object_columns)}")
            X = X.drop(columns=object_columns)
        
        # Verificar si hay valores NaN o infinitos y manejarlos
        X = X.replace([np.inf, -np.inf], np.nan)
        if X.isna().any().any():
            logger.warning(f"Reemplazando valores NaN con mediana para {symbol}")
            X = X.fillna(X.median())
        
        # Obtener solo las características más importantes usando Random Forest para reducir dimensionalidad
        if X.shape[1] > 30:
            logger.info(f"Reduciendo dimensionalidad de {X.shape[1]} características usando RandomForest")
            feature_selector = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            feature_selector.fit(X, y)
            
            # Obtener las 30 características más importantes
            importance = pd.DataFrame({'feature': X.columns, 'importance': feature_selector.feature_importances_})
            importance = importance.sort_values('importance', ascending=False)
            top_features = importance.head(30)['feature'].tolist()
            
            logger.info(f"Top 10 características seleccionadas: {', '.join(top_features[:10])}")
            X = X[top_features]
        
        # Split data into train and test sets with most recent 20% as test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Define time series cross validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Preparar todos los modelos
        models = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [10, 20, None],
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'model__n_estimators': [50, 100],
                    'model__learning_rate': [0.05, 0.1],
                    'model__max_depth': [3, 5],
                }
            },
            'elastic_net': {
                'model': ElasticNet(random_state=42),
                'params': {
                    'model__alpha': [0.1, 0.5, 1.0],
                    'model__l1_ratio': [0.1, 0.5, 0.9],
                }
            }
        }
        
        # Imprimir información de los datos
        logger.info(f"Entrenando modelos para {symbol} - Datos: {X.shape[0]} filas, {X.shape[1]} columnas")
        
        # Entrenamos un modelo a la vez para conservar memoria
        best_model = None
        best_score = float('-inf')
        best_model_name = None
        model_results = {}
        
        # Entrenar cada modelo individualmente
        for model_name, model_info in models.items():
            try:
                logger.info(f"Entrenando {model_name} para {symbol}...")
                
                # Crear pipeline
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model_info['model'])
                ])
                
                # Configurar GridSearchCV
                grid = GridSearchCV(
                    estimator=pipe,
                    param_grid=model_info['params'],
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )
                
                # Entrenar modelo
                grid.fit(X_train, y_train)
                
                # Evaluar en conjunto de prueba
                y_pred = grid.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # Guardar resultados
                model_results[model_name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'best_params': grid.best_params_
                }
                
                # Actualizar mejor modelo
                if r2 > best_score:
                    best_score = r2
                    best_model = grid.best_estimator_
                    best_model_name = model_name
                
                # Limpiar memoria
                del grid
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error entrenando {model_name} para {symbol}: {e}")
                model_results[model_name] = {'error': str(e)}
        
        # Si no se pudo entrenar ningún modelo
        if best_model is None:
            logger.error(f"No se pudo entrenar ningún modelo para {symbol}")
            return {"status": "error", "message": f"Fallo en entrenamiento para {symbol}"}
        
        # Guardar el mejor modelo
        model_filename = os.path.join(os.environ.get('MODEL_OUTPUT_PATH', '/models'), symbol)
        os.makedirs(model_filename, exist_ok=True)
        model_path = os.path.join(model_filename, f"{symbol}_model.pkl")
        joblib.dump(best_model, model_path)
        logger.info(f"Modelo {best_model_name} guardado para {symbol} en {model_path}")
        
        # Generar metadata
        metadata = {
            'symbol': symbol,
            'training_date': datetime.datetime.now().isoformat(),
            'data_points': len(df),
            'features': X.columns.tolist(),
            'best_model': best_model_name,
            'model_params': str(best_model.get_params()),
            'metrics': model_results
        }
        
        # Guardar metadata
        metadata_filename = os.path.join(model_filename, f"{symbol}_metadata.json")
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadata guardada para {symbol} en {metadata_filename}")
        
        # Feature importance plot solo para los modelos que lo soportan
        try:
            if hasattr(best_model[-1], 'feature_importances_'):
                plt.figure(figsize=(10, 6))
                features = X.columns
                importances = best_model[-1].feature_importances_
                indices = np.argsort(importances)[-20:]  # Top 20 features
                
                plt.title(f'Feature Importance for {symbol}')
                plt.barh(range(len(indices)), importances[indices], align='center')
                plt.yticks(range(len(indices)), [features[i] for i in indices])
                plt.xlabel('Relative Importance')
                
                # Guardar gráfico
                plt_filename = os.path.join(model_filename, f"{symbol}_feature_importance.png")
                plt.savefig(plt_filename)
                plt.close()
                logger.info(f"Gráfico de importancia de características guardado para {symbol}")
        except Exception as e:
            logger.warning(f"No se pudo generar gráfico de importancia para {symbol}: {e}")
        
        # Generar reporte HTML
        try:
            report_content = f"""
            <html>
            <head>
                <title>Model Report for {symbol}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    .metric {{ font-weight: bold; }}
                </style>
            </head>
            <body>
                <h1>Model Report for {symbol}</h1>
                <p>Training Date: {metadata['training_date']}</p>
                <p>Data Points: {metadata['data_points']}</p>
                
                <h2>Best Model: {best_model_name}</h2>
                <p>Parameters: {str(best_model.get_params())}</p>
                
                <h2>Performance Metrics</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>MAE</th>
                        <th>RMSE</th>
                        <th>R²</th>
                    </tr>
            """
            
            for model_name, results in model_results.items():
                if 'error' not in results:
                    report_content += f"""
                    <tr>
                        <td>{model_name}</td>
                        <td>{results['mae']:.4f}</td>
                        <td>{results['rmse']:.4f}</td>
                        <td>{results['r2']:.4f}</td>
                    </tr>
                    """
            
            report_content += """
                </table>
                
                <h2>Feature Importance</h2>
                <img src="{symbol}_feature_importance.png" alt="Feature Importance" style="max-width: 100%;" />
            </body>
            </html>
            """
            
            # Guardar reporte HTML
            report_filename = os.path.join(model_filename, f"{symbol}_report.html")
            with open(report_filename, 'w') as f:
                f.write(report_content)
            logger.info(f"Reporte HTML guardado para {symbol}")
        except Exception as e:
            logger.warning(f"No se pudo generar reporte HTML para {symbol}: {e}")
        
        return {
            "status": "success", 
            "symbol": symbol,
            "best_model": best_model_name,
            "metrics": {
                "mae": model_results[best_model_name]['mae'],
                "rmse": model_results[best_model_name]['rmse'],
                "r2": model_results[best_model_name]['r2']
            }
        }
    
    except Exception as e:
        logger.error(f"Error en train_models para {symbol}: {e}")
        return {"status": "error", "message": str(e)}

def save_model(model_data: Dict[str, Any], symbol: str) -> str:
    """
    Save trained model and metadata to disk
    """
    try:
        if not model_data or not model_data.get("best_model"):
            logger.warning(f"No best model found for {symbol}")
            return ""
        
        model_dir = os.path.join(MODEL_OUTPUT_PATH, symbol)
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{symbol}_model.pkl")
        
        if "model_object" in model_data and model_data["model_object"] is not None:
            # Save model file
            joblib.dump(model_data["model_object"], model_path)
            
            # Publish model update to Kafka
            result = publish_model_update(symbol, model_path, model_data)
            
            if result:
                return model_path
            else:
                logger.error(f"Failed to publish model update for {symbol}")
                return model_path
        else:
            logger.error(f"No model object available for {symbol}")
            return ""
    except Exception as e:
        logger.error(f"Failed to save model for {symbol}: {e}")
        return ""

def evaluate_model(model_path: str, symbol: str) -> Dict[str, Any]:
    """
    Load and evaluate a saved model on the most recent data
    """
    if not model_path or not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return {}
        
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Get recent data for evaluation
        recent_query = f"""
        SELECT * FROM market_data 
        WHERE symbol = :symbol 
        ORDER BY datetime DESC
        LIMIT 30
        """
        
        with engine.connect() as conn:
            recent_data = pd.read_sql(
                text(recent_query), 
                conn, 
                params={"symbol": symbol},
                parse_dates=["datetime"]
            )
        
        if recent_data.empty:
            logger.warning(f"No recent data found for {symbol}")
            return {}
        
        # Preprocess recent data
        recent_data = engineer_features(recent_data)
        
        if recent_data.empty:
            logger.warning(f"Failed to engineer features for recent data for {symbol}")
            return {}
        
        # Prepare features
        X_recent = recent_data.drop(['target_next_return', 'target_next_close', 'datetime', 'symbol'], axis=1, errors='ignore')
        
        # Verificar que tengamos todas las columnas necesarias para el modelo
        try:
            # Make predictions
            predictions = model.predict(X_recent)
        except Exception as e:
            logger.error(f"Error al predecir con el modelo para {symbol}: {e}")
            return {}
        
        # Calculate accuracy on known targets if available
        accuracy_metrics = {}
        if 'target_next_close' in recent_data.columns and not recent_data['target_next_close'].isna().all():
            y_true = recent_data['target_next_close'].dropna()
            
            # Verificar que las dimensiones coincidan
            if len(y_true) > 0 and len(predictions) > 0:
                # Asegurar que las longitudes sean iguales
                min_len = min(len(y_true), len(predictions))
                y_true = y_true.iloc[:min_len]
                y_pred = predictions[:min_len]
                
                if min_len > 0:
                    accuracy_metrics = {
                        'mae': mean_absolute_error(y_true, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                        'r2': r2_score(y_true, y_pred)
                    }
        
        # Generate latest prediction
        latest_prediction = float(predictions[0]) if len(predictions) > 0 else None
        latest_close = float(recent_data['close'].iloc[0]) if not recent_data.empty else None
        
        # Calculate prediction delta
        prediction_delta = None
        if latest_prediction is not None and latest_close is not None:
            prediction_delta = (latest_prediction - latest_close) / latest_close * 100
        
        evaluation = {
            'symbol': symbol,
            'model_path': model_path,
            'latest_prediction': latest_prediction,
            'latest_close': latest_close,
            'prediction_delta_pct': prediction_delta,
            'accuracy_metrics': accuracy_metrics,
            'prediction_timestamp': datetime.datetime.now().isoformat()
        }
        
        logger.info(f"Model evaluation for {symbol}: Prediction={latest_prediction:.4f if latest_prediction is not None else 'N/A'}, Delta={prediction_delta:.2f if prediction_delta is not None else 'N/A'}%")
        return evaluation
    except Exception as e:
        logger.error(f"Error evaluating model for {symbol}: {e}")
        return {}

def publish_model_update(symbol: str, model_file: str, model_data: Dict[str, Any]) -> bool:
    """
    Publish model update to Kafka
    """
    if not model_file or not os.path.exists(model_file):
        logger.error(f"Invalid model file: {model_file}")
        return False
        
    try:
        # Métricas que serán útiles para el broker
        metrics = model_data.get("metrics", {})
        best_model_name = model_data.get("best_model", "unknown_model")
        
        # Crear evaluación simplificada (sin el objeto modelo que es muy grande)
        evaluation = {
            "symbol": symbol,
            "model_type": best_model_name,
            "metrics": metrics.get(best_model_name, {}),
            "training_date": model_data.get("training_date", datetime.datetime.now().isoformat()),
            "model_path": model_file,
            "status": "trained"
        }
        
        # Crear evento para diferentes servicios
        event = {
            "event_type": "model_update",
            "service": "batch",
            "symbol": symbol,
            "model_file": model_file,
            "evaluation": evaluation,
            "timestamp": time.time()
        }
        
        # Enviar evento a Kafka asegurando que esté configurado correctamente
        logger.info(f"Sending model update to Kafka topic {KAFKA_TOPIC} with broker {KAFKA_BROKER}")
        
        # Verificar que el productor de Kafka está correctamente configurado
        if producer is None:
            logger.error("Kafka producer is not initialized")
            return False
            
        future = producer.send(KAFKA_TOPIC, event)
        producer.flush()
        record_metadata = future.get(timeout=10)
        
        logger.info(f"Model update published for {symbol} to {record_metadata.topic} partition {record_metadata.partition}")
        return True
    except Exception as e:
        logger.error(f"Error publishing model update for {symbol}: {e}")
        
        # En caso de error con Kafka, escribir un archivo de evento como respaldo
        try:
            event_file = os.path.join(MODEL_OUTPUT_PATH, f"{symbol}_event.json")
            with open(event_file, 'w') as f:
                json.dump({
                    "event_type": "model_update",
                    "service": "batch",
                    "symbol": symbol,
                    "model_file": model_file,
                    "timestamp": time.time()
                }, f)
            logger.info(f"Backup event file written to {event_file}")
        except Exception as e2:
            logger.error(f"Error writing backup event file: {e2}")
            
        return False

def generate_model_report(symbol: str, model_data: Dict[str, Any], evaluation: Dict[str, Any]) -> str:
    """
    Generate a comprehensive model report for stakeholders
    """
    try:
        report_path = os.path.join(MODEL_OUTPUT_PATH, f"{symbol}_report.html")
        
        # Create HTML report
        with open(report_path, 'w') as f:
            f.write(f"""
            <html>
            <head>
                <title>Model Report - {symbol}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .metric {{ font-weight: bold; }}
                    .good {{ color: green; }}
                    .bad {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>Model Report for {symbol}</h1>
                <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Model Summary</h2>
                <table>
                    <tr><th>Symbol</th><td>{symbol}</td></tr>
                    <tr><th>Model Type</th><td>{model_data.get('best_model', 'N/A')}</td></tr>
                    <tr><th>Training Date</th><td>{model_data.get('training_date', 'N/A')}</td></tr>
                    <tr><th>Features Used</th><td>{model_data.get('feature_count', 'N/A')}</td></tr>
                    <tr><th>Training Samples</th><td>{model_data.get('sample_count', 'N/A')}</td></tr>
                </table>
                
                <h2>Performance Metrics</h2>
                <table>
            """)
            
            if model_data.get('best_model') and model_data.get('best_model') in model_data.get('models', {}):
                metrics = model_data['models'][model_data['best_model']]
                f.write(f"""
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>MAE</td><td>{metrics.get('mae', 'N/A'):.4f}</td></tr>
                    <tr><td>RMSE</td><td>{metrics.get('rmse', 'N/A'):.4f}</td></tr>
                    <tr><td>R²</td><td>{metrics.get('r2', 'N/A'):.4f}</td></tr>
                """)
            
            f.write("""
                </table>
                
                <h2>Latest Prediction</h2>
                <table>
            """)
            
            prediction_delta = evaluation.get('prediction_delta_pct')
            delta_class = 'good' if prediction_delta and prediction_delta > 0 else 'bad'
            
            f.write(f"""
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Latest Close</td><td>${evaluation.get('latest_close', 'N/A'):.2f}</td></tr>
                <tr><td>Prediction</td><td>${evaluation.get('latest_prediction', 'N/A'):.2f}</td></tr>
                <tr><td>Predicted Change</td><td class="{delta_class}">{evaluation.get('prediction_delta_pct', 'N/A'):.2f}%</td></tr>
            """)
            
            f.write("""
                </table>
                
                <h2>Feature Importance</h2>
                <table>
                    <tr><th>Feature</th><th>Importance</th></tr>
            """)
            
            for feature in model_data.get('feature_importance', []):
                f.write(f"""
                    <tr><td>{feature['feature']}</td><td>{feature['importance']:.4f}</td></tr>
                """)
            
            f.write("""
                </table>
                
                <h2>Hyperparameters</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
            """)
            
            if model_data.get('best_model') and model_data.get('best_model') in model_data.get('models', {}):
                best_params = model_data['models'][model_data['best_model']].get('best_params', {})
                for param, value in best_params.items():
                    f.write(f"""
                        <tr><td>{param}</td><td>{value}</td></tr>
                    """)
            
            f.write("""
                </table>
                
                <h2>Conclusions</h2>
                <p>
            """)
            
            # Add some basic conclusions
            if prediction_delta:
                direction = "up" if prediction_delta > 0 else "down"
                f.write(f"""
                    The model predicts that {symbol} will go <span class="{delta_class}">{direction} by {abs(prediction_delta):.2f}%</span> in the next period.
                """)
            
            # Add model performance assessment
            if model_data.get('best_model') and model_data.get('best_model') in model_data.get('models', {}):
                r2 = model_data['models'][model_data['best_model']].get('r2', 0)
                if r2 > 0.7:
                    performance = "excellent"
                elif r2 > 0.5:
                    performance = "good"
                elif r2 > 0.3:
                    performance = "moderate"
                else:
                    performance = "poor"
                
                f.write(f"""
                    <br><br>The model shows <b>{performance}</b> predictive performance with an R² value of {r2:.4f}.
                """)
            
            f.write("""
                </p>
            </body>
            </html>
            """)
        
        logger.info(f"Model report generated for {symbol} at {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"Error generating model report for {symbol}: {e}")
        return ""

def main():
    """
    Main function to orchestrate the batch training process
    """
    logger.info("Starting batch training process")
    
    # List of symbols to process - originally all symbols will be processed
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "IAG.MC", "PHM.MC", "BKY.MC", "AENA.MC", "BA", "NLGO", "CAR", "DLTR", "CANTE.IS", "SASA.IS"]
    
    results = []
    successes = 0
    
    for symbol in symbols:
        logger.info(f"Processing {symbol}")
        
        # Step 1: Load historical data
        df = load_data(symbol)
        
        if df.empty:
            logger.error(f"No data available for {symbol}, skipping")
            results.append({"symbol": symbol, "status": "error", "message": "No data found"})
            continue
        
        # Step 2: Engineer features
        df = engineer_features(df)
        
        if df.empty:
            logger.error(f"Feature engineering failed for {symbol}, skipping")
            results.append({"symbol": symbol, "status": "error", "message": "Feature engineering failed"})
            continue
        
        # Step 3: Train models
        model_result = train_models(df, symbol)
        
        if model_result.get("status") == "success":
            # Extraer el objeto modelo del directorio
            model_dir = os.path.join(MODEL_OUTPUT_PATH, symbol)
            model_path = os.path.join(model_dir, f"{symbol}_model.pkl")
            
            if os.path.exists(model_path):
                # Cargar el modelo para publicarlo
                try:
                    model_object = joblib.load(model_path)
                    
                    # Preparar datos de modelo para publicación
                    model_data = {
                        "best_model": model_result.get("best_model"),
                        "metrics": model_result.get("metrics", {}),
                        "model_object": model_object,
                        "training_date": datetime.datetime.now().isoformat(),
                        "symbol": symbol
                    }
                    
                    # Publicar modelo - sin pasar el objeto modelo, solo el path
                    model_data_for_publish = model_data.copy()
                    del model_data_for_publish["model_object"]  # Eliminar objeto modelo para publicación
                    
                    # Publicar modelo
                    success = publish_model_update(symbol, model_path, model_data_for_publish)
                    
                    if success:
                        successes += 1
                        logger.info(f"Successfully saved and published model for {symbol}")
                    else:
                        logger.warning(f"Model trained but not published for {symbol}")
                    
                    results.append({
                        "symbol": symbol,
                        "status": "success",
                        "model": model_result.get("best_model"),
                        "metrics": model_result.get("metrics", {})
                    })
                except Exception as e:
                    logger.error(f"Error publishing model for {symbol}: {e}")
                    results.append({"symbol": symbol, "status": "error", "message": f"Model publication failed: {e}"})
            else:
                logger.error(f"Model file not found at {model_path}")
                results.append({"symbol": symbol, "status": "error", "message": "Model file not found"})
        else:
            results.append({
                "symbol": symbol, 
                "status": "error", 
                "message": model_result.get("message", "Unknown error")
            })
        
    # Save batch results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(MODEL_OUTPUT_PATH, f"batch_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "symbols_processed": len(symbols),
            "successful_models": successes,
            "results": results
        }, f, indent=4)
    
    logger.info(f"Batch results saved to {results_file}")
    logger.info(f"Batch process completed. Processed {len(symbols)} symbols with {successes} successes")
    
    # Close Kafka producer properly
    producer.close()

if __name__ == "__main__":
    main()
