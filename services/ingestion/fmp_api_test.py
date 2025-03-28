import os
import requests
from datetime import datetime, timedelta

def test_fmp_endpoints():
    # Cargar configuración desde variables de entorno
    API_KEY = os.environ.get('FMP_API_KEY')
    BASE_URL = os.environ.get('FMP_BASE_URL')
    
    # Símbolos de prueba
    SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'IAG.MC']
    
    # Intervalos de prueba
    INTERVALS = ['1min', '5min', '15min', '30min', '45min', '1hour']
    
    # Fechas en el pasado
    end_date = datetime(2024, 3, 1)
    start_date = end_date - timedelta(days=10)

    def test_endpoint(symbol, interval):
        url = f"{BASE_URL}/historical-chart/{interval}/{symbol}?from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&apikey={API_KEY}"
        
        print(f"\n📊 Probando: {symbol} - Intervalo {interval}")
        print(f"URL: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            
            print(f"Código de estado: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Registros recibidos: {len(data)}")
                
                if data:
                    print("Primer registro:")
                    print(data[0])
                    return True
                else:
                    print("❌ Sin datos")
                    return False
            else:
                print(f"❌ Error: {response.text}")
                return False
        
        except Exception as e:
            print(f"❌ Excepción: {e}")
            return False

    # Resultados generales
    results = {
        'total_tests': 0,
        'successful_tests': 0,
        'failed_tests': 0
    }

    # Ejecutar pruebas
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            results['total_tests'] += 1
            success = test_endpoint(symbol, interval)
            
            if success:
                results['successful_tests'] += 1
            else:
                results['failed_tests'] += 1

    # Resumen final
    print("\n📈 Resumen de pruebas:")
    print(f"Total de pruebas: {results['total_tests']}")
    print(f"Pruebas exitosas: {results['successful_tests']}")
    print(f"Pruebas fallidas: {results['failed_tests']}")

    # Devolver estado final
    return results['failed_tests'] == 0

# Ejecutar pruebas y salir con código de estado apropiado
if __name__ == '__main__':
    success = test_fmp_endpoints()
    exit(0 if success else 1)