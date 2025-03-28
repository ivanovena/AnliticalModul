import requests
from datetime import datetime, timedelta

def test_fmp_api():
    # Configuraciones
    API_KEY = 'h5JPnHPAdjxBAXAGwTOL3Acs3W5zaByx'
    BASE_URL = 'https://financialmodelingprep.com/api/v3'
    SYMBOL = 'AAPL'

    # Usar fechas en el pasado
    end_date = datetime(2024, 3, 1)  # Fecha concreta en el pasado
    start_date = end_date - timedelta(days=10)

    # Formar URL
    url = f"{BASE_URL}/historical-chart/5min/{SYMBOL}?from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&apikey={API_KEY}"

    print(f"Probando URL: {url}")

    # Realizar solicitud
    response = requests.get(url)

    # Verificar respuesta
    print(f"Código de estado: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Número de registros recibidos: {len(data)}")
        
        if data:
            print("\nPrimer registro:")
            print(data[0])
    else:
        print("Error en la solicitud")
        print(response.text)

# Ejecutar test
test_fmp_api()