// src/components/Portfolio/PortfolioSummary.jsx
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Card, Row, Col, Spin } from 'antd';

const PortfolioSummary = () => {
  const [portfolioData, setPortfolioData] = useState({
    totalBalance: 0,
    availableCash: 0,
    activePositions: 0,
    isLoading: true,
    error: null
  });

  useEffect(() => {
    const fetchPortfolioData = async () => {
      try {
        // Obtener datos del portfolio desde el Broker Service
        const portfolioResponse = await axios.get('/api/portfolio');
        
        // Obtener órdenes activas
        const ordersResponse = await axios.get('/api/orders?status=active');
        
        // Calcular el valor total del portfolio
        // Valor total = Efectivo disponible + Valor de mercado de las posiciones
        const positions = portfolioResponse.data.positions || [];
        const positionsValue = positions.reduce((total, position) => {
          return total + (position.quantity * position.currentPrice);
        }, 0);
        
        const totalBalance = portfolioResponse.data.cash + positionsValue;
        
        setPortfolioData({
          totalBalance: totalBalance,
          availableCash: portfolioResponse.data.cash,
          activePositions: positions.length,
          isLoading: false,
          error: null
        });
      } catch (error) {
        console.error('Error fetching portfolio data:', error);
        setPortfolioData({
          ...portfolioData,
          isLoading: false,
          error: 'Error al cargar datos del portfolio'
        });
      }
    };

    fetchPortfolioData();
    
    // Actualizar datos cada 30 segundos
    const intervalId = setInterval(fetchPortfolioData, 30000);
    
    return () => clearInterval(intervalId);
  }, []);

  if (portfolioData.isLoading) {
    return <Spin tip="Cargando datos del portfolio..." />;
  }

  if (portfolioData.error) {
    return <div className="error-message">{portfolioData.error}</div>;
  }

  return (
    <div className="portfolio-summary">
      <h2>Resumen del Portfolio</h2>
      <Row gutter={16}>
        <Col span={8}>
          <Card className="summary-card bg-light-blue">
            <h3>Balance Total</h3>
            <h2>${portfolioData.totalBalance.toFixed(2)}</h2>
          </Card>
        </Col>
        <Col span={8}>
          <Card className="summary-card bg-light-green">
            <h3>Efectivo Disponible</h3>
            <h2>${portfolioData.availableCash.toFixed(2)}</h2>
          </Card>
        </Col>
        <Col span={8}>
          <Card className="summary-card bg-light-purple">
            <h3>Posiciones Activas</h3>
            <h2>{portfolioData.activePositions}</h2>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default PortfolioSummary;