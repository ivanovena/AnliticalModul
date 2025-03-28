from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import os

class AdaptiveOfflineModel:
    def __init__(self, symbol):
        self.symbol = symbol
        self.models = {
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'elastic_net': ElasticNet(random_state=42),
            'svr': SVR()
        }
        
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', None)  # Será reemplazado dinámicamente
        ])
        
        self.performance_metrics = {
            'mae': [],
            'rmse': [],
            'r2': []
        }
        
        self.feature_importances = {}
    
    def inject_online_patterns(self, online_knowledge):
        """
        Inyectar conocimiento del modelo online
        """
        if online_knowledge:
            # Ajustar hiperparámetros
            for model in self.models.values():
                if hasattr(model, 'learning_rate'):
                    model.learning_rate = online_knowledge.get('learning_rate', 0.01)
                
                # Ajustar pesos si es posible
                if hasattr(model, 'feature_importances_'):
                    model.feature_weights_ = online_knowledge.get('feature_importance', {})
        
        return self
    
    def train(self, X, y, model_type='random_forest'):
        """
        Entrenamiento con múltiples modelos
        """
        try:
            # Seleccionar modelo base
            base_model = self.models[model_type]
            
            # Configurar pipeline
            self.pipeline.set_params(model=base_model)
            
            # Configuración de búsqueda de hiperparámetros
            param_grid = {}
            
            # Aplicar parámetros según el tipo de modelo
            if model_type == 'random_forest':
                param_grid = {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [None, 10, 20]
                }
            elif model_type == 'gradient_boosting':
                param_grid = {
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.01, 0.1, 0.2]
                }
            elif model_type == 'elastic_net':
                param_grid = {
                    'model__alpha': [0.1, 1.0, 10.0],
                    'model__l1_ratio': [0.1, 0.5, 0.9]
                }
            elif model_type == 'svr':
                param_grid = {
                    'model__C': [0.1, 1.0, 10.0],
                    'model__epsilon': [0.01, 0.1, 0.2]
                }
                
            grid_search = GridSearchCV(
                self.pipeline,
                param_grid=param_grid,
                cv=TimeSeriesSplit(n_splits=5),
                scoring='neg_mean_squared_error'
            )
            
            # Entrenar
            grid_search.fit(X, y)
            
            # Mejores parámetros
            best_model = grid_search.best_estimator_
            
            # Extraer importancia de características
            if hasattr(best_model.named_steps['model'], 'feature_importances_'):
                self.feature_importances = {
                    f'feature_{i}': imp 
                    for i, imp in enumerate(best_model.named_steps['model'].feature_importances_)
                }
            
            # Registrar métricas
            y_pred = best_model.predict(X)
            self.performance_metrics = {
                'mae': mean_absolute_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2': r2_score(y, y_pred)
            }
            
            # Guardar modelo con ruta relativa o absoluta correcta
            from pathlib import Path
            models_dir = Path(os.getenv('MODEL_OUTPUT_PATH', '/models'))
            models_dir.mkdir(exist_ok=True)
            joblib.dump(best_model, models_dir / f'{self.symbol}_offline_model.pkl')
            
            return best_model
        
        except Exception as e:
            print(f"Error en entrenamiento: {e}")
            return None
    
    def predict(self, X):
        """
        Predicción con modelo entrenado
        """
        return self.pipeline.predict(X)
