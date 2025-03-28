#!/usr/bin/env python3
"""
Script para verificar la correcta configuración del entorno.
Comprueba que todas las variables de entorno necesarias estén configuradas.
"""

import os
import sys
import dotenv
from colorama import init, Fore, Style

# Inicializar colorama
init()

def load_env():
    """Carga las variables de entorno desde .env"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_path):
        dotenv.load_dotenv(env_path)
        print(f"{Fore.GREEN}✓ Archivo .env cargado correctamente{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}✗ No se encontró el archivo .env{Style.RESET_ALL}")
        print(f"  Ejecuta: cp .env.example .env y configura tus credenciales")
        return False
    return True

def check_required_vars():
    """Verifica variables de entorno requeridas"""
    required_vars = {
        # API Keys
        "FMP_API_KEY": "Clave API de Financial Modeling Prep",
        
        # Configuración de DB
        "DB_URI": "URI de conexión a la base de datos",
        "POSTGRES_USER": "Usuario de PostgreSQL",
        "POSTGRES_PASSWORD": "Contraseña de PostgreSQL",
        "POSTGRES_DB": "Nombre de la base de datos",
        
        # Configuración Kafka
        "KAFKA_BROKER": "Dirección del broker de Kafka",
        
        # Parámetros Trading
        "INITIAL_CASH": "Efectivo inicial para trading"
    }
    
    optional_vars = {
        "TELEGRAM_BOT_TOKEN": "Token del Bot de Telegram (opcional)",
        "MODEL_OUTPUT_PATH": "Ruta donde se guardarán los modelos",
        "USE_DATALAKE": "Usar datalake en lugar de API FMP directamente"
    }
    
    all_required_present = True
    for var, description in required_vars.items():
        if var not in os.environ or not os.environ[var]:
            print(f"{Fore.RED}✗ Falta la variable requerida: {var} - {description}{Style.RESET_ALL}")
            all_required_present = False
        else:
            value = os.environ[var]
            # Ocultar contraseñas y claves API
            if any(substr in var for substr in ["PASSWORD", "KEY", "SECRET", "TOKEN"]):
                value = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
            print(f"{Fore.GREEN}✓ {var}: {value}{Style.RESET_ALL}")
    
    print("\nVariables opcionales:")
    for var, description in optional_vars.items():
        if var not in os.environ or not os.environ[var]:
            print(f"{Fore.YELLOW}? {var} no configurada - {description}{Style.RESET_ALL}")
        else:
            value = os.environ[var]
            # Ocultar contraseñas y claves API
            if any(substr in var for substr in ["PASSWORD", "KEY", "SECRET", "TOKEN"]):
                value = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
            print(f"{Fore.GREEN}✓ {var}: {value}{Style.RESET_ALL}")
    
    return all_required_present

def check_docker_installed():
    """Verifica si Docker está instalado"""
    import subprocess
    try:
        subprocess.run(["docker", "--version"], check=True, stdout=subprocess.PIPE)
        print(f"{Fore.GREEN}✓ Docker está instalado{Style.RESET_ALL}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print(f"{Fore.RED}✗ Docker no está instalado o no está en el PATH{Style.RESET_ALL}")
        return False

def check_services_config():
    """Verifica que los archivos de configuración de servicios estén presentes"""
    services = ["ingestion", "broker", "streaming", "batch"]
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    all_configs_present = True
    for service in services:
        service_dir = os.path.join(root_dir, "services", service)
        config_path = os.path.join(service_dir, "config.py")
        
        if os.path.exists(config_path):
            print(f"{Fore.GREEN}✓ Configuración de {service} encontrada{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}✗ Falta la configuración de {service}: {config_path}{Style.RESET_ALL}")
            all_configs_present = False
    
    return all_configs_present

def main():
    """Función principal"""
    print(f"{Fore.CYAN}=== Verificación del Entorno ==={Style.RESET_ALL}\n")
    
    if not load_env():
        return 1
    
    print("\n" + f"{Fore.CYAN}=== Variables de Entorno ==={Style.RESET_ALL}")
    if not check_required_vars():
        print(f"\n{Fore.RED}✗ Faltan variables de entorno requeridas. Por favor, configura el archivo .env{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.GREEN}✓ Todas las variables de entorno requeridas están configuradas{Style.RESET_ALL}")
    
    print("\n" + f"{Fore.CYAN}=== Dependencias ==={Style.RESET_ALL}")
    check_docker_installed()
    
    print("\n" + f"{Fore.CYAN}=== Configuración de Servicios ==={Style.RESET_ALL}")
    check_services_config()
    
    print("\n" + f"{Fore.CYAN}=== Verificación Finalizada ==={Style.RESET_ALL}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
