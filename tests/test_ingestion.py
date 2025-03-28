import unittest
import requests
import os
import sys
import json
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
import pandas as pd
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import service modules
from services.ingestion.app import fetch_data, store_data, validate_symbol
from services.ingestion.config import FMP_API_KEY, FMP_BASE_URL, DB_URI

class TestIngestionService(unittest.TestCase):
    """Tests for the ingestion service"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a test database connection
        self.test_engine = create_engine('sqlite:///:memory:')
        
        # Create test table
        self.test_engine.execute('''
            CREATE TABLE market_data (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                datetime TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER
            )
        ''')
        
        # Sample market data for testing
        self.sample_data = [
            {
                "date": "2023-01-01 10:00:00",
                "open": 150.0,
                "high": 155.0,
                "low": 149.0,
                "close": 153.0,
                "volume": 1000000
            },
            {
                "date": "2023-01-01 10:01:00",
                "open": 153.0,
                "high": 154.0,
                "low": 152.0,
                "close": 152.5,
                "volume": 500000
            }
        ]
    
    def tearDown(self):
        """Clean up after tests"""
        self.test_engine.dispose()
    
    def test_validate_symbol_valid(self):
        """Test symbol validation with valid symbols"""
        self.assertTrue(validate_symbol("AAPL"))
        self.assertTrue(validate_symbol("MSFT"))
        self.assertTrue(validate_symbol("GOOGL"))
        self.assertTrue(validate_symbol("AMZN"))
    
    def test_validate_symbol_invalid(self):
        """Test symbol validation with invalid symbols"""
        self.assertFalse(validate_symbol(""))
        self.assertFalse(validate_symbol(None))
        self.assertFalse(validate_symbol("aapl"))  # lowercase
        self.assertFalse(validate_symbol("AAPL123"))  # alphanumeric
        self.assertFalse(validate_symbol("TOOBIG123456"))  # too long
        self.assertFalse(validate_symbol(123))  # non-string
    
    @patch('services.ingestion.app.requests.get')
    def test_fetch_data_success(self, mock_get):
        """Test successful data fetching"""
        # Configure mock
        mock_response = MagicMock()
        mock_response.json.return_value = self.sample_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Call function
        result = fetch_data("AAPL", interval="1min")
        
        # Assertions
        self.assertEqual(result, self.sample_data)
        mock_get.assert_called_once_with(
            f"{FMP_BASE_URL}/historical-chart/1min/AAPL?apikey={FMP_API_KEY}",
            timeout=15
        )
    
    @patch('services.ingestion.app.requests.get')
    def test_fetch_data_http_error(self, mock_get):
        """Test HTTP error handling"""
        # Configure mock to raise HTTP error
        mock_get.side_effect = requests.exceptions.HTTPError("404 Client Error")
        
        # Call function
        result = fetch_data("AAPL", interval="1min")
        
        # Assertions
        self.assertEqual(result, [])
    
    @patch('services.ingestion.app.requests.get')
    def test_fetch_data_timeout(self, mock_get):
        """Test timeout error handling"""
        # Configure mock to raise timeout
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
        
        # Call function
        result = fetch_data("AAPL", interval="1min")
        
        # Assertions
        self.assertEqual(result, [])
    
    @patch('services.ingestion.app.requests.get')
    def test_fetch_data_connection_error(self, mock_get):
        """Test connection error handling"""
        # Configure mock to raise connection error
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        # Call function
        result = fetch_data("AAPL", interval="1min")
        
        # Assertions
        self.assertEqual(result, [])
    
    @patch('services.ingestion.app.engine', new_callable=lambda: create_engine('sqlite:///:memory:'))
    def test_store_data(self, mock_engine):
        """Test data storage"""
        # Setup test database
        mock_engine.execute('''
            CREATE TABLE market_data (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                datetime TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                UNIQUE(symbol, datetime)
            )
        ''')
        
        # Call function
        rows = store_data("AAPL", self.sample_data)
        
        # Verify data was stored
        result = mock_engine.execute("SELECT * FROM market_data").fetchall()
        
        # Assertions
        self.assertEqual(rows, 2)  # 2 rows processed
        self.assertEqual(len(result), 2)  # 2 rows in database
        self.assertEqual(result[0][1], "AAPL")  # Symbol matches
        self.assertEqual(result[0][2], "2023-01-01 10:00:00")  # Datetime matches
    
    @patch('services.ingestion.app.engine', new_callable=lambda: create_engine('sqlite:///:memory:'))
    def test_store_data_empty(self, mock_engine):
        """Test storing empty data"""
        # Setup test database
        mock_engine.execute('''
            CREATE TABLE market_data (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                datetime TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                UNIQUE(symbol, datetime)
            )
        ''')
        
        # Call function with empty data
        rows = store_data("AAPL", [])
        
        # Assertions
        self.assertEqual(rows, 0)  # 0 rows processed
    
    @patch('services.ingestion.app.engine', new_callable=lambda: create_engine('sqlite:///:memory:'))
    def test_store_data_duplicate(self, mock_engine):
        """Test handling duplicate data"""
        # Setup test database
        mock_engine.execute('''
            CREATE TABLE market_data (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                datetime TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                UNIQUE(symbol, datetime)
            )
        ''')
        
        # Store data first time
        store_data("AAPL", self.sample_data)
        
        # Store same data again (should update existing records)
        rows = store_data("AAPL", self.sample_data)
        
        # Verify data was stored without duplicates
        result = mock_engine.execute("SELECT * FROM market_data").fetchall()
        
        # Assertions
        self.assertEqual(rows, 2)  # 2 rows processed
        self.assertEqual(len(result), 2)  # Still only 2 rows in database
    
    @patch('services.ingestion.app.engine', new_callable=lambda: create_engine('sqlite:///:memory:'))
    def test_store_data_incomplete(self, mock_engine):
        """Test handling incomplete data"""
        # Setup test database
        mock_engine.execute('''
            CREATE TABLE market_data (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                datetime TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                UNIQUE(symbol, datetime)
            )
        ''')
        
        # Incomplete data missing required fields
        incomplete_data = [
            {
                "date": "2023-01-01 10:00:00",
                "open": 150.0,
                # Missing high, low, close
                "volume": 1000000
            }
        ]
        
        # Call function
        rows = store_data("AAPL", incomplete_data)
        
        # Assertions
        self.assertEqual(rows, 0)  # 0 rows processed due to incomplete data
    
    @patch('services.ingestion.app.fetch_data')
    @patch('services.ingestion.app.store_data')
    def test_integration_fetch_and_store(self, mock_store, mock_fetch):
        """Test integration between fetch and store functions"""
        # Configure mocks
        mock_fetch.return_value = self.sample_data
        mock_store.return_value = 2
        
        # Import main function
        from services.ingestion.app import main
        
        # Mock time.sleep to speed up test
        with patch('time.sleep'):
            # Run the main function
            main()
        
        # Verify fetch_data and store_data were called for each symbol
        self.assertEqual(mock_fetch.call_count, 10)  # 5 symbols * 2 intervals
        self.assertEqual(mock_store.call_count, 10)  # 5 symbols * 2 intervals

if __name__ == '__main__':
    unittest.main()
