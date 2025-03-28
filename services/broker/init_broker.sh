#!/bin/bash

# Script de inicialización del broker AI

echo "Iniciando configuración del broker IA..."

# Crear directorios necesarios
mkdir -p models
mkdir -p logs

# Verificar Python y pip
python3 --version || echo "ERROR: Python3 no está instalado"
pip3 --version || echo "ERROR: Pip no está instalado"

# Instalar dependencias
echo "Instalando dependencias..."
pip3 install -r requirements.txt

# Configurar modelo LLM
echo "Configurando modelo LLM..."
python3 setup_model.py

# Verificar configuración
if [ -f "models/llama-3.2-1B-chat.Q4_K_M.gguf" ]; then
    echo "Modelo LLM configurado correctamente"
else
    echo "AVISO: Modelo LLM no encontrado, se usará el modo fallback"
fi

# Iniciar servicio
echo "Iniciando servicio de broker..."
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
