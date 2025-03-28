from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import sys

# Cargar variables de entorno
load_dotenv()

def verify_database_connection():
    """Verificar conexión y datos en la base de datos"""
    try:
        # Obtener la URI de la base de datos del entorno
        DB_URI = os.getenv("DB_URI", "postgresql://usuario:contraseña@postgres:5432/tu_basedatos")
        
        # Crear motor de base de datos
        engine = create_engine(DB_URI)
        
        # Intentar conectar
        with engine.connect() as conn:
            print("✅ Conexión a la base de datos establecida correctamente")
            
            # Verificar tablas
            tables_query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = conn.execute(tables_query)
            table_list = [table[0] for table in tables]
            
            print("\n📋 Tablas existentes:")
            for table in table_list:
                print(f"- {table}")
            
            # Verificar datos en market_data
            market_data_query = text("""
                SELECT 
                    COUNT(*) as total_records, 
                    MIN(timestamp) as earliest_timestamp, 
                    MAX(timestamp) as latest_timestamp,
                    COUNT(DISTINCT symbol) as unique_symbols
                FROM market_data
            """)
            result = conn.execute(market_data_query).fetchone()
            
            print(f"\n📊 Estadísticas de market_data:")
            print(f"- Total de registros: {result[0]}")
            print(f"- Símbolos únicos: {result[3]}")
            
            if result[0] > 0:
                print(f"- Timestamps: desde {result[1]} hasta {result[2]}")
            
            # Verificar variedad de datos
            data_type_query = text("""
                SELECT 
                    data_type, 
                    COUNT(*) as count 
                FROM market_data 
                GROUP BY data_type
            """)
            data_types = conn.execute(data_type_query)
            
            print("\n📈 Distribución de tipos de datos:")
            for row in data_types:
                print(f"- {row[0]}: {row[1]} registros")
            
            return True
    
    except Exception as e:
        print(f"❌ Error al verificar la base de datos: {e}")
        return False

def verify_market_data():
    """Verificar datos de mercado específicos"""
    try:
        # Obtener la URI de la base de datos del entorno
        DB_URI = os.getenv("DB_URI", "postgresql://usuario:contraseña@postgres:5432/tu_basedatos")
        
        # Crear motor de base de datos
        engine = create_engine(DB_URI)
        
        # Intentar conectar
        with engine.connect() as conn:
            print("\n🔍 Muestreo de datos de mercado:")
            
            # Obtener algunos datos de ejemplo
            sample_query = text("""
                SELECT symbol, price, volume, timestamp, data_type
                FROM market_data
                ORDER BY RANDOM()
                LIMIT 10
            """)
            sample_data = conn.execute(sample_query)
            
            for row in sample_data:
                print(f"- {row[0]}: Precio={row[1]}, Volumen={row[2]}, Timestamp={row[3]}, Tipo={row[4]}")
            
            return True
    
    except Exception as e:
        print(f"❌ Error al verificar datos de mercado: {e}")
        return False

def main():
    print("🕹️ Sistema de Verificación de Base de Datos")
    
    # Verificar conexión y estructura
    connection_status = verify_database_connection()
    
    # Verificar datos de mercado
    market_data_status = verify_market_data()
    
    # Resultado final
    if connection_status and market_data_status:
        print("\n✅ VERIFICACIÓN COMPLETADA CON ÉXITO")
        sys.exit(0)
    else:
        print("\n❌ SE ENCONTRARON PROBLEMAS DURANTE LA VERIFICACIÓN")
        sys.exit(1)

if __name__ == "__main__":
    main()
