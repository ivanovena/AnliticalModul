import unittest
import os
import sys
import json
import time
import datetime
from unittest.mock import patch, MagicMock, Mock
import requests
import numpy as np
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import service modules
from services.broker.app import app, execute_order, get_stock_quote, advanced_risk_management
from services.broker.config import INITIAL_CASH

class TestBrokerService(unittest.TestCase):
    """Tests for the broker service"""
    
    def setUp(self):
        """Set up test environment"""
        self.client = TestClient(app)
        
        # Reset portfolio to initial state
        from services.broker.app import portfolio
        portfolio["cash"] = INITIAL_CASH
        portfolio["positions"] = {}
        portfolio["last_update"] = time.time()
        
        # Reset orders
        from services.broker.app import orders
        orders.clear()
        
        # Sample stock data
        self.sample_quote = {
            "symbol": "AAPL",
            "price": 150.0,
            "change": 1.5,
            "changesPercentage": 1.0,
            "volume": 1000000
        }
        
        self.sample_historical = {
            "symbol": "AAPL",
            "historical": [
                {"date": "2023-01-05", "close": 152.0},
                {"date": "2023-01-04", "close": 151.0},
                {"date": "2023-01-03", "close": 150.0},
                {"date": "2023-01-02", "close": 149.0},
                {"date": "2023-01-01", "close": 148.0}
            ]
        }
    
    @patch('services.broker.app.get_stock_quote')
    def test_portfolio_endpoint(self, mock_get_quote):
        """Test portfolio endpoint"""
        # Configure mock
        mock_get_quote.return_value = self.sample_quote
        
        # Execute a test order
        execute_order("AAPL", "BUY", 10, 150.0)
        
        # Call endpoint
        response = self.client.get("/portfolio")
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["cash"], INITIAL_CASH - 10 * 150.0)
        self.assertEqual(data["positions"]["AAPL"]["quantity"], 10)
    
    def test_orders_endpoint(self):
        """Test orders endpoint"""
        # Execute a test order
        execute_order("AAPL", "BUY", 10, 150.0)
        
        # Call endpoint
        response = self.client.get("/orders")
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["symbol"], "AAPL")
        self.assertEqual(data[0]["action"], "BUY")
        self.assertEqual(data[0]["quantity"], 10)
        self.assertEqual(data[0]["price"], 150.0)
    
    def test_place_order_endpoint_buy(self):
        """Test place order endpoint - buy"""
        # Prepare order request
        order_request = {
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 10,
            "price": 150.0
        }
        
        # Call endpoint
        response = self.client.post("/order", json=order_request)
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["order"]["symbol"], "AAPL")
        
        # Check portfolio update
        from services.broker.app import portfolio
        self.assertEqual(portfolio["cash"], INITIAL_CASH - 10 * 150.0)
        self.assertEqual(portfolio["positions"]["AAPL"], 10)
    
    def test_place_order_endpoint_sell(self):
        """Test place order endpoint - sell"""
        # Set up portfolio with existing position
        from services.broker.app import portfolio
        portfolio["positions"]["AAPL"] = 10
        
        # Prepare order request
        order_request = {
            "symbol": "AAPL",
            "action": "SELL",
            "quantity": 5,
            "price": 150.0
        }
        
        # Call endpoint
        response = self.client.post("/order", json=order_request)
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["order"]["symbol"], "AAPL")
        
        # Check portfolio update
        self.assertEqual(portfolio["positions"]["AAPL"], 5)
        self.assertEqual(portfolio["cash"], INITIAL_CASH + 5 * 150.0)
    
    def test_place_order_endpoint_sell_insufficient(self):
        """Test place order endpoint - sell with insufficient shares"""
        # Set up portfolio with insufficient position
        from services.broker.app import portfolio
        portfolio["positions"]["AAPL"] = 3
        
        # Prepare order request
        order_request = {
            "symbol": "AAPL",
            "action": "SELL",
            "quantity": 5,
            "price": 150.0
        }
        
        # Call endpoint
        response = self.client.post("/order", json=order_request)
        
        # Assertions
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["detail"], "Acciones insuficientes")
        
        # Check portfolio remained unchanged
        self.assertEqual(portfolio["positions"]["AAPL"], 3)
        self.assertEqual(portfolio["cash"], INITIAL_CASH)
    
    def test_place_order_endpoint_buy_insufficient_cash(self):
        """Test place order endpoint - buy with insufficient cash"""
        # Set up portfolio with insufficient cash
        from services.broker.app import portfolio
        portfolio["cash"] = 100.0
        
        # Prepare order request
        order_request = {
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 10,
            "price": 150.0
        }
        
        # Call endpoint
        response = self.client.post("/order", json=order_request)
        
        # Assertions
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["detail"], "Efectivo insuficiente")
        
        # Check portfolio remained unchanged
        self.assertEqual(portfolio["cash"], 100.0)
        self.assertEqual(len(portfolio["positions"]), 0)
    
    @patch('services.broker.app.get_stock_quote')
    @patch('services.broker.app.get_historical_prices')
    @patch('services.broker.app.get_stock_sentiment')
    def test_metrics_endpoint(self, mock_sentiment, mock_historical, mock_quote):
        """Test metrics endpoint"""
        # Configure mocks
        mock_quote.return_value = self.sample_quote
        mock_historical.return_value = [x["close"] for x in self.sample_historical["historical"]]
        mock_sentiment.return_value = 0.5
        
        # Set up portfolio with positions
        from services.broker.app import portfolio
        portfolio["positions"]["AAPL"] = 10
        portfolio["positions"]["MSFT"] = 5
        
        # Call endpoint
        response = self.client.get("/metrics")
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue("total_value" in data)
        self.assertTrue("cash" in data)
        self.assertTrue("positions_value" in data)
        self.assertTrue("risk_metrics" in data)
        
        # Verify risk metrics calculated for each position
        self.assertTrue("AAPL" in data["risk_metrics"])
        self.assertTrue("MSFT" in data["risk_metrics"])
        self.assertTrue("VaR" in data["risk_metrics"]["AAPL"])
        self.assertTrue("sentiment" in data["risk_metrics"]["AAPL"])
    
    def test_chat_endpoint(self):
        """Test chat endpoint"""
        # Prepare chat request
        chat_request = {
            "message": "¿Cuál es mi capital actual?"
        }
        
        # Call endpoint
        response = self.client.post("/chat", json=chat_request)
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue("response" in data)
        self.assertTrue("conversation_id" in data)
        self.assertIn("efectivo", data["response"].lower())
    
    @patch('services.broker.app.get_stock_quote')
    def test_execute_order_buy(self, mock_get_quote):
        """Test execute_order function - buy"""
        # Configure mock
        mock_get_quote.return_value = self.sample_quote
        
        # Execute order
        order = execute_order("AAPL", "BUY", 10, 150.0)
        
        # Assertions
        from services.broker.app import portfolio, orders
        self.assertEqual(portfolio["cash"], INITIAL_CASH - 10 * 150.0)
        self.assertEqual(portfolio["positions"]["AAPL"], 10)
        self.assertEqual(len(orders), 1)
        self.assertEqual(order["symbol"], "AAPL")
        self.assertEqual(order["action"], "BUY")
        self.assertEqual(order["quantity"], 10)
        self.assertEqual(order["price"], 150.0)
    
    @patch('services.broker.app.get_stock_quote')
    def test_execute_order_sell(self, mock_get_quote):
        """Test execute_order function - sell"""
        # Configure mock
        mock_get_quote.return_value = self.sample_quote
        
        # Set up portfolio with existing position
        from services.broker.app import portfolio
        portfolio["positions"]["AAPL"] = 10
        
        # Execute order
        order = execute_order("AAPL", "SELL", 5, 150.0)
        
        # Assertions
        from services.broker.app import orders
        self.assertEqual(portfolio["cash"], INITIAL_CASH + 5 * 150.0)
        self.assertEqual(portfolio["positions"]["AAPL"], 5)
        self.assertEqual(len(orders), 1)
        self.assertEqual(order["symbol"], "AAPL")
        self.assertEqual(order["action"], "SELL")
        self.assertEqual(order["quantity"], 5)
        self.assertEqual(order["price"], 150.0)
    
    @patch('services.broker.app.get_stock_quote')
    def test_execute_order_sell_all(self, mock_get_quote):
        """Test execute_order function - sell all shares"""
        # Configure mock
        mock_get_quote.return_value = self.sample_quote
        
        # Set up portfolio with existing position
        from services.broker.app import portfolio
        portfolio["positions"]["AAPL"] = 10
        
        # Execute order
        order = execute_order("AAPL", "SELL", 10, 150.0)
        
        # Assertions
        from services.broker.app import orders
        self.assertEqual(portfolio["cash"], INITIAL_CASH + 10 * 150.0)
        self.assertEqual(portfolio["positions"].get("AAPL", 0), 0)
        self.assertEqual(len(orders), 1)
    
    @patch('requests.get')
    def test_get_stock_quote(self, mock_get):
        """Test get_stock_quote function"""
        # Configure mock
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [self.sample_quote]
        mock_get.return_value = mock_response
        
        # Call function
        result = get_stock_quote("AAPL")
        
        # Assertions
        self.assertEqual(result, self.sample_quote)
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_get_stock_quote_error(self, mock_get):
        """Test get_stock_quote function - error handling"""
        # Configure mock to raise error
        mock_get.side_effect = requests.exceptions.RequestException("API error")
        
        # Assertions
        with self.assertRaises(Exception):
            get_stock_quote("AAPL")
    
    @patch('services.broker.app.get_historical_prices')
    @patch('services.broker.app.get_stock_sentiment')
    def test_advanced_risk_management(self, mock_sentiment, mock_historical):
        """Test advanced_risk_management function"""
        # Configure mocks
        mock_historical.return_value = [152.0, 151.0, 150.0, 149.0, 148.0]
        mock_sentiment.return_value = 0.5
        
        # Set up portfolio with positions
        from services.broker.app import portfolio
        portfolio["positions"]["AAPL"] = 10
        
        # Call function
        risk_results = advanced_risk_management(num_simulations=1000)
        
        # Assertions
        self.assertTrue("AAPL" in risk_results)
        self.assertIsNotNone(risk_results["AAPL"]["VaR"])
        self.assertIsNotNone(risk_results["AAPL"]["CVaR"])
        self.assertIsNotNone(risk_results["AAPL"]["sentiment"])
        self.assertIsNotNone(risk_results["AAPL"]["max_drawdown"])
        self.assertIsNotNone(risk_results["AAPL"]["sharpe_ratio"])
    
    def test_plan_endpoint(self):
        """Test plan endpoint"""
        # Prepare plan request
        plan_request = {
            "predictions": {
                "AAPL": 5.0,
                "GOOGL": -2.0,
                "MSFT": 1.0
            }
        }
        
        # Call endpoint
        response = self.client.post("/plan", json=plan_request)
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue("investment_plan" in data)
        self.assertTrue("risk_assessment" in data)
        
        # Check plan decisions
        self.assertEqual(data["investment_plan"]["AAPL"]["action"], "BUY")
        self.assertEqual(data["investment_plan"]["GOOGL"]["action"], "SELL")
        self.assertEqual(data["investment_plan"]["MSFT"]["action"], "HOLD")

if __name__ == '__main__':
    unittest.main()
