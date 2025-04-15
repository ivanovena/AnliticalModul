from flask import Flask, jsonify, send_file, render_template, redirect, url_for, request
import logging
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import time
import datetime
import joblib
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Importar las funciones existentes
from config import MODEL_OUTPUT_PATH
from app import create_model_registry

# Configurar el Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("batch_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BatchAPIService")

# Inicializar aplicación Flask para API y dashboard
app = Flask(__name__, template_folder="templates")

# Métricas para Prometheus
MODEL_TRAINING_COUNTER = Counter('model_training_total', 'Total number of model trainings', ['symbol', 'model_type', 'status'])
MODEL_PERFORMANCE_GAUGE = Gauge('model_performance', 'Model performance metrics', ['symbol', 'metric'])
MODEL_TRAINING_TIME = Histogram('model_training_duration_seconds', 'Time spent training models', ['symbol'])

# Iniciar servidor de métricas de Prometheus en puerto 8000
try:
    prometheus_port = int(os.environ.get("PROMETHEUS_PORT", 8000))
    start_http_server(prometheus_port)
    logger.info(f"Servidor de métricas Prometheus iniciado en puerto {prometheus_port}")
except Exception as e:
    logger.error(f"Error al iniciar servidor de métricas Prometheus: {e}")
    # Continuar sin métricas de Prometheus si falla

# Crear directorio para plantillas si no existe
os.makedirs(os.path.join(os.path.dirname(__file__), "templates"), exist_ok=True)

# Crear plantilla básica para el dashboard
dashboard_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Model Training Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .good {
            color: green;
            font-weight: bold;
        }
        .bad {
            color: red;
            font-weight: bold;
        }
        .model-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .model-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .model-header h3 {
            margin: 0;
        }
        .metrics {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-bottom: 15px;
        }
        .metric {
            background: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            min-width: 120px;
        }
        .metric p {
            margin: 0;
            font-size: 0.9em;
        }
        .metric .value {
            font-size: 1.2em;
            font-weight: bold;
        }
        .model-actions {
            margin-top: 15px;
        }
        .btn {
            padding: 8px 12px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
            font-size: 14px;
            display: inline-block;
        }
        .btn:hover {
            background-color: #2980b9;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Training Dashboard</h1>
        <p>Last updated: {{ last_updated }}</p>
        
        <h2>Model Summary</h2>
        <table>
            <tr>
                <th>Symbol</th>
                <th>Latest Model</th>
                <th>Training Date</th>
                <th>R²</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>Actions</th>
            </tr>
            {% for symbol, model in models.items() %}
                {% if model.latest_version and model.versions|length > 0 %}
                <tr>
                    <td>{{ symbol }}</td>
                    <td>{{ model.versions[0].best_model }}</td>
                    <td>{{ model.versions[0].creation_date|replace('T', ' ')|truncate(16, true, '') }}</td>
                    <td class="{% if model.versions[0].metrics.r2 > 0.5 %}good{% elif model.versions[0].metrics.r2 < 0.2 %}bad{% endif %}">
                        {{ "%.4f"|format(model.versions[0].metrics.r2) if model.versions[0].metrics.r2 is defined else 'N/A' }}
                    </td>
                    <td>{{ "%.4f"|format(model.versions[0].metrics.mae) if model.versions[0].metrics.mae is defined else 'N/A' }}</td>
                    <td>{{ "%.4f"|format(model.versions[0].metrics.rmse) if model.versions[0].metrics.rmse is defined else 'N/A' }}</td>
                    <td>
                        <a href="/model/{{ symbol }}" class="btn">Details</a>
                        <a href="/report/{{ symbol }}" class="btn">Report</a>
                    </td>
                </tr>
                {% endif %}
            {% endfor %}
        </table>
        
        <h2>Recent Training Runs</h2>
        <table>
            <tr>
                <th>Timestamp</th>
                <th>Symbols Processed</th>
                <th>Success Rate</th>
                <th>Report</th>
            </tr>
            {% for run in recent_runs %}
            <tr>
                <td>{{ run.timestamp|replace('T', ' ')|truncate(16, true, '') }}</td>
                <td>{{ run.symbols_processed }}</td>
                <td>{{ "%.1f%%"|format(run.success_rate * 100) }}</td>
                <td><a href="/batch_results/{{ run.filename }}" class="btn">View Details</a></td>
            </tr>
            {% endfor %}
        </table>
    </div>
</body>
</html>
"""

# Guardar la plantilla del dashboard
os.makedirs(os.path.join(os.path.dirname(__file__), "templates"), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), "templates", "dashboard.html"), "w") as f:
    f.write(dashboard_template)

# Crear plantilla para detalles del modelo
model_details_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ symbol }} - Model Details</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd.
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .good {
            color: green;
            font-weight: bold.
        }
        .bad {
            color: red;
            font-weight: bold.
        }
        .model-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px.
        }
        .tabs {
            display: flex;
            border-bottom: 1px solid #ddd;
            margin-bottom: 20px.
        }
        .tab {
            padding: 10px 15px;
            cursor: pointer;
            margin-right: 5px.
        }
        .tab.active {
            border: 1px solid #ddd;
            border-bottom: 1px solid white;
            border-radius: 5px 5px 0 0;
            margin-bottom: -1px.
        }
        .tab-content {
            display: none.
        }
        .tab-content.active {
            display: block.
        }
        img {
            max-width: 100%;
            height: auto.
        }
        .btn {
            padding: 8px 12px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
            font-size: 14px;
            display: inline-block;
            margin-right: 10px.
        }
        .btn:hover {
            background-color: #2980b9.
        }
        .btn-back {
            background-color: #7f8c8d.
        }
        .btn-back:hover {
            background-color: #95a5a6.
        }
    </style>
</head>
<body>
    <div class="container">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h1>{{ symbol }} - Model Details</h1>
            <a href="/" class="btn btn-back">Back to Dashboard</a>
        </div>
        
        <div class="model-card">
            <div class="model-header">
                <h3>Latest Model: {{ latest_model.best_model }}</h3>
                <span>Trained on: {{ latest_model.creation_date|replace('T', ' ')|truncate(16, true, '') }}</span>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <p>R²</p>
                    <p class="value {% if latest_model.metrics.r2 > 0.5 %}good{% elif latest_model.metrics.r2 < 0.2 %}bad{% endif %}">
                        {{ "%.4f"|format(latest_model.metrics.r2) if latest_model.metrics.r2 is defined else 'N/A' }}
                    </p>
                </div>
                <div class="metric">
                    <p>MAE</p>
                    <p class="value">{{ "%.4f"|format(latest_model.metrics.mae) if latest_model.metrics.mae is defined else 'N/A' }}</p>
                </div>
                <div class="metric">
                    <p>RMSE</p>
                    <p class="value">{{ "%.4f"|format(latest_model.metrics.rmse) if latest_model.metrics.rmse is defined else 'N/A' }}</p>
                </div>
            </div>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="openTab(event, 'versions')">Model Versions</div>
            <div class="tab" onclick="openTab(event, 'features')">Feature Importance</div>
            <div class="tab" onclick="openTab(event, 'performance')">Performance History</div>
        </div>
        
        <div id="versions" class="tab-content active">
            <h2>Model Version History</h2>
            <table>
                <tr>
                    <th>Version</th>
                    <th>Model Type</th>
                    <th>Training Date</th>
                    <th>R²</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>Actions</th>
                </tr>
                {% for version in versions %}
                <tr>
                    <td>{{ loop.index }}</td>
                    <td>{{ version.best_model }}</td>
                    <td>{{ version.creation_date|replace('T', ' ')|truncate(16, true, '') }}</td>
                    <td class="{% if version.metrics.r2 > 0.5 %}good{% elif version.metrics.r2 < 0.2 %}bad{% endif %}">
                        {{ "%.4f"|format(version.metrics.r2) if version.metrics.r2 is defined else 'N/A' }}
                    </td>
                    <td>{{ "%.4f"|format(version.metrics.mae) if version.metrics.mae is defined else 'N/A' }}</td>
                    <td>{{ "%.4f"|format(version.metrics.rmse) if version.metrics.rmse is defined else 'N/A' }}</td>
                    <td>
                        <a href="/report/{{ symbol }}/{{ version.path }}" class="btn">Report</a>
                    </td>
                </tr>
                {% endfor %}
            </table>
        </div>
        
        <div id="features" class="tab-content">
            <h2>Feature Importance</h2>
            {% if feature_importance_path %}
                <img src="/static/{{ feature_importance_path }}" alt="Feature Importance">
            {% else %}
                <p>No feature importance visualization available for this model.</p>
            {% endif %}
        </div>
        
        <div id="performance" class="tab-content">
            <h2>Performance History</h2>
            <p>Chart with performance metrics over time for different model versions.</p>
            <!-- Si hubiera un gráfico de rendimiento a través del tiempo, iría aquí -->
            <p>This feature will be available in future versions.</p>
        </div>
    </div>
    
    <script>
        function openTab(evt, tabName) {
            // Ocultar todos los contenidos de pestañas
            var tabcontent = document.getElementsByClassName("tab-content");
            for (var i = 0; i < tabcontent.length; i++) {
                tabcontent[i].className = tabcontent[i].className.replace(" active", "");
            }
            
            // Desactivar todas las pestañas
            var tabs = document.getElementsByClassName("tab");
            for (var i = 0; i < tabs.length; i++) {
                tabs[i].className = tabs[i].className.replace(" active", "");
            }
            
            // Mostrar el contenido de la pestaña actual y activar la pestaña
            document.getElementById(tabName).className += " active";
            evt.currentTarget.className += " active";
        }
    </script>
</body>
</html>
"""

# Guardar la plantilla de detalles del modelo
with open(os.path.join(os.path.dirname(__file__), "templates", "model_details.html"), "w") as f:
    f.write(model_details_template)

@app.route('/')
def dashboard():
    """Muestra el dashboard principal con resumen de modelos"""
    try:
        # Obtener datos del registro de modelos
        registry_path = os.path.join(MODEL_OUTPUT_PATH, "model_registry.json")
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            # Si no existe el registro, crearlo
            create_model_registry(MODEL_OUTPUT_PATH)
            if os.path.exists(registry_path):
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
            else:
                registry = {"last_updated": datetime.datetime.now().isoformat(), "models": {}}
        
        # Obtener las ejecuciones recientes de batch
        recent_runs = []
        for filename in os.listdir(MODEL_OUTPUT_PATH):
            if filename.startswith("batch_results_") and filename.endswith(".json"):
                file_path = os.path.join(MODEL_OUTPUT_PATH, filename)
                try:
                    with open(file_path, 'r') as f:
                        run_data = json.load(f)
                        
                    # Calcular tasa de éxito
                    success_rate = run_data.get("successful_models", 0) / run_data.get("symbols_processed", 1)
                    
                    recent_runs.append({
                        "timestamp": run_data.get("timestamp", ""),
                        "symbols_processed": run_data.get("symbols_processed", 0),
                        "success_rate": success_rate,
                        "filename": filename
                    })
                except Exception as e:
                    logger.error(f"Error reading batch result file {filename}: {e}")
        
        # Ordenar por fecha, más recientes primero
        recent_runs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Limitar a los 10 más recientes
        recent_runs = recent_runs[:10]
        
        return render_template("dashboard.html", 
                              models=registry.get("models", {}),
                              last_updated=registry.get("last_updated", "").replace("T", " "),
                              recent_runs=recent_runs)
    
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/model/<symbol>')
def model_details(symbol):
    """Muestra detalles de un modelo específico"""
    try:
        # Obtener datos del registro de modelos
        registry_path = os.path.join(MODEL_OUTPUT_PATH, "model_registry.json")
        if not os.path.exists(registry_path):
            return jsonify({"error": "Model registry not found"}), 404
            
        with open(registry_path, 'r') as f:
            registry = json.load(f)
            
        if symbol not in registry.get("models", {}):
            return jsonify({"error": f"No models found for {symbol}"}), 404
            
        model_data = registry["models"][symbol]
        
        # Buscar gráfico de importancia de características
        feature_importance_path = None
        symbol_dir = os.path.join(MODEL_OUTPUT_PATH, symbol)
        
        for root, _, files in os.walk(symbol_dir):
            for file in files:
                if file.endswith("_feature_importance.png"):
                    # Obtener ruta relativa desde MODEL_OUTPUT_PATH
                    rel_path = os.path.relpath(os.path.join(root, file), start=MODEL_OUTPUT_PATH)
                    feature_importance_path = rel_path
                    break
            if feature_importance_path:
                break
                
        # Renderizar plantilla con datos
        latest_model = model_data["versions"][0] if model_data["versions"] else {}
        
        return render_template("model_details.html",
                              symbol=symbol,
                              latest_model=latest_model,
                              versions=model_data["versions"],
                              feature_importance_path=feature_importance_path)
    
    except Exception as e:
        logger.error(f"Error rendering model details for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/report/<symbol>')
@app.route('/report/<symbol>/<path:model_path>')
def model_report(symbol, model_path=None):
    """Muestra el informe HTML del modelo"""
    try:
        # Si no se especifica ruta, buscar el informe más reciente
        if not model_path:
            report_path = os.path.join(MODEL_OUTPUT_PATH, symbol, f"{symbol}_report.html")
            if not os.path.exists(report_path):
                # Buscar en subdirectorios
                for root, _, files in os.walk(os.path.join(MODEL_OUTPUT_PATH, symbol)):
                    for file in files:
                        if file.endswith("_report.html"):
                            report_path = os.path.join(root, file)
                            break
                    if os.path.exists(report_path):
                        break
        else:
            # Convertir model_path a report_path
            report_path = model_path.replace(".pkl", "_report.html")
            if not os.path.exists(report_path):
                return jsonify({"error": f"Report not found for {symbol} at {model_path}"}), 404
        
        if not os.path.exists(report_path):
            return jsonify({"error": f"No report found for {symbol}"}), 404
            
        # Servir el archivo HTML
        return send_file(report_path)
    
    except Exception as e:
        logger.error(f"Error serving report for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/batch_results/<filename>')
def batch_results(filename):
    """Muestra los resultados de una ejecución de batch"""
    try:
        file_path = os.path.join(MODEL_OUTPUT_PATH, filename)
        if not os.path.exists(file_path):
            return jsonify({"error": f"Batch results file not found: {filename}"}), 404
            
        with open(file_path, 'r') as f:
            results = json.load(f)
            
        # Renderizar como JSON por ahora; en el futuro podría tener una plantilla HTML
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error serving batch results {filename}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/models')
def list_models_api():
    """API para listar todos los modelos disponibles"""
    try:
        registry_path = os.path.join(MODEL_OUTPUT_PATH, "model_registry.json")
        if not os.path.exists(registry_path):
            return jsonify({"error": "Model registry not found"}), 404
            
        with open(registry_path, 'r') as f:
            registry = json.load(f)
            
        return jsonify(registry)
    
    except Exception as e:
        logger.error(f"Error in list_models_api: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/<symbol>')
def get_model_api(symbol):
    """API para obtener detalles de un modelo específico"""
    try:
        registry_path = os.path.join(MODEL_OUTPUT_PATH, "model_registry.json")
        if not os.path.exists(registry_path):
            return jsonify({"error": "Model registry not found"}), 404
            
        with open(registry_path, 'r') as f:
            registry = json.load(f)
            
        if symbol not in registry.get("models", {}):
            return jsonify({"error": f"No models found for {symbol}"}), 404
            
        return jsonify(registry["models"][symbol])
    
    except Exception as e:
        logger.error(f"Error in get_model_api for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/<symbol>/predict', methods=['POST'])
def predict_api(symbol):
    """API para hacer predicciones con un modelo"""
    try:
        # Verificar que el modelo existe
        registry_path = os.path.join(MODEL_OUTPUT_PATH, "model_registry.json")
        if not os.path.exists(registry_path):
            return jsonify({"error": "Model registry not found"}), 404
            
        with open(registry_path, 'r') as f:
            registry = json.load(f)
            
        if symbol not in registry.get("models", {}):
            return jsonify({"error": f"No models found for {symbol}"}), 404
            
        # Obtener el modelo más reciente
        if not registry["models"][symbol]["versions"]:
            return jsonify({"error": f"No model versions found for {symbol}"}), 404
            
        model_path = registry["models"][symbol]["versions"][0]["path"]
        
        # Cargar el modelo
        try:
            model = joblib.load(model_path)
        except Exception as e:
            return jsonify({"error": f"Error loading model: {e}"}), 500
            
        # Obtener datos de entrada del request
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400
            
        # Validar datos de entrada
        if not isinstance(data, dict):
            return jsonify({"error": "Input must be a JSON object"}), 400
            
        # Preparar datos para predicción
        try:
            # Esta parte depende de cómo está implementado tu modelo
            # y qué datos espera para hacer predicciones
            features = pd.DataFrame([data])
            prediction = model.predict(features)
            return jsonify({
                "symbol": symbol,
                "prediction": float(prediction[0]),
                "timestamp": datetime.datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"error": f"Error making prediction: {e}"}), 500
    
    except Exception as e:
        logger.error(f"Error in predict_api for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

# Configurar directorio estático para servir imágenes y otros archivos
@app.route('/static/<path:path>')
def serve_static(path):
    """Sirve archivos estáticos desde MODEL_OUTPUT_PATH"""
    return send_file(os.path.join(MODEL_OUTPUT_PATH, path))

@app.route('/health')
def health_check():
    """
    Endpoint para verificación de salud del servicio
    """
    return jsonify({
        "status": "healthy",
        "service": "batch_service",
        "timestamp": datetime.datetime.now().isoformat()
    })

# Punto de entrada para ejecución como servicio
def serve_api():
    """Inicia el servidor Flask para API y dashboard"""
    # Obtener puerto del ambiente o usar 8080 por defecto
    port = int(os.environ.get("API_PORT", 8080))
    logger.info(f"Starting batch API service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
    
if __name__ == "__main__":
    serve_api()