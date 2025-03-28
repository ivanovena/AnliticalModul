import os
import pickle

# Crear modelo dummy
dummy_model = {'weights': [1, 2, 3], 'bias': 0.5}

# Asegurarse de que la carpeta models exista
os.makedirs('models', exist_ok=True)

# Guardar el modelo
with open('models/AAPL_dummy_model.pkl', 'wb') as f:
    pickle.dump(dummy_model, f)

print('Modelo dummy creado en models/AAPL_dummy_model.pkl')
