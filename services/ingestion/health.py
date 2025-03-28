from flask import Flask, jsonify
import time
import threading
import os

app = Flask(__name__)

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": time.time()
    })

def start_health_server():
    app.run(host='0.0.0.0', port=8000)

if __name__ == "__main__":
    # Iniciar en un hilo separado
    health_thread = threading.Thread(target=start_health_server)
    health_thread.daemon = True
    health_thread.start()
