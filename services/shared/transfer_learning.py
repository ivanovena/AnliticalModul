import numpy as np
import json
import logging
from typing import Dict, List, Any
from datetime import datetime

class TransferLearningManager:
    def __init__(self, symbol: str):
        """
        Gestiona el transfer learning entre modelos online y offline
        """
        self.symbol = symbol
        self.online_knowledge_base = {}
        self.offline_knowledge_base = {}
        self.transfer_history = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"TransferLearning_{symbol}")
    
    def extract_online_knowledge(self, online_model):
        """
        Extrae conocimiento del modelo online
        """
        try:
            # Extraer patrones recientes
            recent_patterns = {
                'recent_weights': online_model.get_weights() if hasattr(online_model, "get_weights") else None,
                'learning_rate': getattr(online_model, "learning_rate", 0.01),
                'feature_importance': online_model.estimate_feature_importance() if hasattr(online_model, "estimate_feature_importance") else {}
            }
            
            self.online_knowledge_base = recent_patterns
            self.logger.info(f"Conocimiento online extraído para {self.symbol}")
            return recent_patterns
        except Exception as e:
            self.logger.error(f"Error extrayendo conocimiento online: {e}")
            return {}
    
    def extract_offline_knowledge(self, offline_model):
        """
        Extrae conocimiento del modelo offline
        """
        try:
            # Extraer características importantes y pesos
            offline_knowledge = {
                'feature_importances': getattr(offline_model, 'feature_importances_', {}),
                'model_parameters': offline_model.get_params() if hasattr(offline_model, "get_params") else {},
                'performance_metrics': getattr(offline_model, 'performance_metrics', {})
            }
            
            self.offline_knowledge_base = offline_knowledge
            self.logger.info(f"Conocimiento offline extraído para {self.symbol}")
            return offline_knowledge
        except Exception as e:
            self.logger.error(f"Error extrayendo conocimiento offline: {e}")
            return {}
    
    def transfer_online_to_offline(self, offline_model):
        """
        Transfiere conocimiento del modelo online al offline
        """
        try:
            # Ajustar hiperparámetros
            if self.online_knowledge_base:
                if hasattr(offline_model, "set_params"):
                    offline_model.set_params(
                        learning_rate=self.online_knowledge_base.get('learning_rate', 0.01),
                        feature_weights=self.online_knowledge_base.get('feature_importance', None)
                    )
                
                self.logger.info(f"Conocimiento online transferido al modelo offline para {self.symbol}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error en transferencia online a offline: {e}")
            return False
    
    def transfer_offline_to_online(self, online_model):
        """
        Transfiere conocimiento del modelo offline al online
        """
        try:
            # Inyectar características importantes
            if self.offline_knowledge_base:
                if hasattr(online_model, "update_feature_weights"):
                    online_model.update_feature_weights(
                        self.offline_knowledge_base.get('feature_importances', [])
                    )
                
                self.logger.info(f"Conocimiento offline transferido al modelo online para {self.symbol}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error en transferencia offline a online: {e}")
            return False
    
    def transfer_knowledge(self, source='offline', target='online', model_source=None, model_target=None):
        """
        Método unificado para transferir conocimientos entre modelos
        
        Args:
            source: Fuente de conocimiento ('offline' o 'online')
            target: Destino del conocimiento ('offline' o 'online')
            model_source: Modelo fuente (opcional)
            model_target: Modelo destino (opcional)
            
        Returns:
            bool: Éxito de la transferencia
        """
        try:
            # Extraer conocimiento adicional si se proporciona un modelo fuente
            if model_source:
                if source == 'online':
                    self.extract_online_knowledge(model_source)
                else:
                    self.extract_offline_knowledge(model_source)
            
            # Realizar transferencia de conocimiento
            if source == 'offline' and target == 'online':
                # Transferir feature importances de offline a online
                online_kb = self.online_knowledge_base.copy()
                
                # Obtener feature importances del modelo offline
                feature_importances = self.offline_knowledge_base.get('feature_importances', {})
                if not feature_importances and 'feature_importance' in self.offline_knowledge_base:
                    feature_importances = self.offline_knowledge_base.get('feature_importance', {})
                
                # Actualizar knowledge base online con feature importances offline
                if 'feature_importance' in online_kb:
                    # Combinar feature importances
                    for feature, importance in feature_importances.items():
                        if feature in online_kb['feature_importance']:
                            # Promedio ponderado (70% offline, 30% online para features importantes)
                            if importance > 0.2:  # Feature importante en offline
                                online_kb['feature_importance'][feature] = 0.7 * importance + 0.3 * online_kb['feature_importance'].get(feature, 0)
                        else:
                            # Añadir nueva feature
                            online_kb['feature_importance'][feature] = importance
                else:
                    online_kb['feature_importance'] = feature_importances
                
                # Actualizar knowledge base
                self.online_knowledge_base = online_kb
                
                # Aplicar a modelo si se proporcionó
                if model_target:
                    return self.transfer_offline_to_online(model_target)
                
                self.log_transfer('offline_to_online', {
                    'features_transferred': len(feature_importances),
                    'success': True
                })
                return True
                
            elif source == 'online' and target == 'offline':
                # Transferir patrones recientes de online a offline
                offline_kb = self.offline_knowledge_base.copy()
                
                # Obtener patrones recientes del modelo online
                recent_weights = self.online_knowledge_base.get('recent_weights', {})
                learning_rate = self.online_knowledge_base.get('learning_rate', 0.01)
                
                # Actualizar knowledge base offline con patrones online
                offline_kb['recent_patterns'] = recent_weights
                offline_kb['learning_rate'] = learning_rate
                
                # Actualizar knowledge base
                self.offline_knowledge_base = offline_kb
                
                # Aplicar a modelo si se proporcionó
                if model_target:
                    return self.transfer_online_to_offline(model_target)
                
                self.log_transfer('online_to_offline', {
                    'success': True,
                    'learning_rate': learning_rate
                })
                return True
            
            else:
                self.logger.error(f"Tipo de transferencia no válida: {source} a {target}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error en transferencia de conocimiento {source} a {target}: {e}")
            self.log_transfer(f'{source}_to_{target}', {
                'success': False,
                'error': str(e)
            })
            return False
    
    def log_transfer(self, transfer_type: str, details: Dict[str, Any]):
        """
        Registra cada evento de transferencia de conocimiento
        """
        transfer_event = {
            'symbol': self.symbol,
            'type': transfer_type,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        
        self.transfer_history.append(transfer_event)
        
        # Limitar historial
        if len(self.transfer_history) > 100:
            self.transfer_history = self.transfer_history[-100:]
        
        # Opcional: Persistir historial
        try:
            with open(f'/models/{self.symbol}_transfer_history.json', 'w') as f:
                json.dump(self.transfer_history, f, indent=2)
        except Exception as e:
            self.logger.warning(f"No se pudo guardar historial de transferencia: {e}")
