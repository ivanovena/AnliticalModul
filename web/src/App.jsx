import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

// Proveedores de contexto
import { PortfolioProvider } from './contexts/PortfolioContext';
import { MarketDataProvider } from './contexts/MarketDataContext';
import { PredictionProvider } from './contexts/PredictionContext';
import { ModelStatusProvider } from './contexts/ModelStatusContext';
import { NotificationProvider } from './contexts/NotificationContext';

// Componentes y layouts
import MainLayout from './components/layouts/MainLayout';

// Páginas
import Dashboard from './pages/Dashboard';
import TradingSimulator from './pages/TradingSimulator';
import ModelPerformance from './pages/ModelPerformance';
import PredictionAnalysis from './pages/PredictionAnalysis';
import BrokerChat from './pages/BrokerChat';
import MarketAnalysis from './pages/MarketAnalysis';

function App() {
  return (
    <NotificationProvider>
      <MarketDataProvider>
        <PortfolioProvider>
          <PredictionProvider>
            <ModelStatusProvider>
              <Router>
                <Routes>
                  <Route path="/" element={<MainLayout />}>
                    <Route index element={<Dashboard />} />
                    <Route path="trading" element={<TradingSimulator />} />
                    <Route path="models" element={<ModelPerformance />} />
                    <Route path="predictions" element={<PredictionAnalysis />} />
                    <Route path="broker-chat" element={<BrokerChat />} />
                    <Route path="market" element={<MarketAnalysis />} />
                  </Route>
                </Routes>
              </Router>
            </ModelStatusProvider>
          </PredictionProvider>
        </PortfolioProvider>
      </MarketDataProvider>
    </NotificationProvider>
  );
}

export default App;
