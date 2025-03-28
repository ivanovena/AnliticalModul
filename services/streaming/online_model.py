from river import linear_model, preprocessing, feature_extraction, ensemble
import numpy as np

class AdaptiveOnlineModel:
    def __init__(self, symbol):
        self.symbol = symbol
        self.feature_weights = {}
        
        # Modelo base - usando regresores disponibles en river
        self.model = ensemble.EWARegressor([
            linear_model.LinearRegression(),
            linear_model.PARegressor(),
            linear_model.SGDRegressor()
        ])
        
        # Preprocesamiento adaptativo
        self.preprocessor = preprocessing.StandardScaler()
        
        # Métricas de desempeño
        self.performance_metrics = {
            'predictions': [],
            'actuals': [],
            'errors': []
        }
    
    def update_feature_weights(self, feature_importances):
        """
        Actualiza pesos de características desde conocimiento externo
        """
        self.feature_weights.update(feature_importances)
    
    def learn_one(self, features, target):
        """
        Aprendizaje incremental con adaptación de pesos
        """
        # Aplicar pesos de características si existen
        for feature, weight in self.feature_weights.items():
            if feature in features:
                features[feature] *= weight
        
        # Preprocesar y aprender
        processed_features = self.preprocessor.learn_one(features)
        self.model.learn_one(processed_features, target)
        
        # Registrar métricas
        self.performance_metrics['predictions'].append(
            self.model.predict_one(processed_features)
        )
        self.performance_metrics['actuals'].append(target)
        
        # Calcular error
        prediction = self.performance_metrics['predictions'][-1]
        error = abs(prediction - target)
        self.performance_metrics['errors'].append(error)
    
    def predict_one(self, features):
        """
        Predicción con características preprocesadas
        """
        processed_features = self.preprocessor.transform_one(features)
        return self.model.predict_one(processed_features)
    
    def estimate_feature_importance(self):
        """
        Estimar importancia de características
        """
        if not self.performance_metrics['predictions']:
            return {}
        
        # Método simple de importancia basado en correlación
        importances = {}
        for feature in features:
            feature_values = [f[feature] for f in features]
            correlation = np.corrcoef(
                feature_values, 
                self.performance_metrics['errors']
            )[0, 1]
            importances[feature] = abs(correlation)
        
        return importances
    
    def get_weights(self):
        """
        Obtener pesos actuales del modelo
        """
        return {
            'preprocessor_weights': self.preprocessor.mu,
            'preprocessor_std': self.preprocessor.sigma,
            'feature_weights': self.feature_weights
        }
