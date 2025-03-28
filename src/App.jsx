// src/App.jsx
import React, { useState, useEffect } from 'react';
import { Layout, Menu, Input, Button, Select } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import 'antd/dist/antd.css';
import './App.css';

// Importar componentes
import PortfolioSummary from './components/Portfolio/PortfolioSummary';
import StockPriceChart from './components/Charts/StockPriceChart';
import TradingSimulator from './components/Trading/TradingSimulator';

const { Header, Content, Footer } = Layout;
const { Option } = Select;

const App = () => {
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [searchQuery, setSearchQuery] = useState('');

  const handleMenuClick = (e) => {
    setCurrentPage(e.key);
  };

  const handleSymbolChange = (value) => {
    setSelectedSymbol(value);
  };

  const handleSearch = (value) => {
    setSearchQuery(value);
    // Aquí se podría realizar una búsqueda y cambiar el símbolo seleccionado
  };

  // Renderizar el contenido basado en la página actual
  const renderContent = () => {
    switch (currentPage) {
      case 'dashboard':
        return (
          <>
            <PortfolioSummary />
            <div className="content-section" style={{ marginTop: 24 }}>
              <h2>Predicciones del Mercado</h2>
              <div className="button-container" style={{ textAlign: 'right', marginBottom: 16 }}>
                <Button type="primary">Simulador de Trading</Button>
              </div>
              <div className="predictions-table">
                {/* Aquí iría una tabla de predicciones */}
              </div>
            </div>
          </>
        );
      case 'graficos':
        return (
          <div className="content-section">
            <StockPriceChart symbol={selectedSymbol} />
          </div>
        );
      case 'tiemporeal':
        return (
          <div className="content-section">
            <h2>Datos en Tiempo Real</h2>
            {/* Componente de datos en tiempo real */}
          </div>
        );
      case 'metricas':
        return (
          <div className="content-section">
            <h2>Métricas de Rendimiento</h2>
            {/* Componente de métricas */}
          </div>
        );
      case 'modelos':
        return (
          <div className="content-section">
            <h2>Gestión de Modelos</h2>
            {/* Componente de modelos */}
          </div>
        );
      case 'broker':
        return (
          <div className="content-section">
            <h2>Administración de Órdenes</h2>
            {/* Componente de broker */}
          </div>
        );
      case 'trading':
        return (
          <div className="content-section">
            <TradingSimulator />
          </div>
        );
      default:
        return <div>Página no encontrada</div>;
    }
  };

  return (
    <Layout className="layout">
      <Header className="header">
        <div className="logo">Modelo Bursátil</div>
        <Menu
          theme="light"
          mode="horizontal"
          defaultSelectedKeys={['dashboard']}
          onClick={handleMenuClick}
        >
          <Menu.Item key="dashboard">Dashboard</Menu.Item>
          <Menu.Item key="graficos">Gráficos</Menu.Item>
          <Menu.Item key="tiemporeal">Tiempo Real</Menu.Item>
          <Menu.Item key="metricas">Métricas</Menu.Item>
          <Menu.Item key="modelos">Modelos</Menu.Item>
          <Menu.Item key="broker">Broker</Menu.Item>
          <Menu.Item key="trading">Trading</Menu.Item>
        </Menu>
        <div className="search-container">
          <Input.Group compact>
            <Select 
              defaultValue={selectedSymbol} 
              style={{ width: 120 }} 
              onChange={handleSymbolChange}
            >
              <Option value="AAPL">AAPL</Option>
              <Option value="MSFT">MSFT</Option>
              <Option value="GOOGL">GOOGL</Option>
              <Option value="AMZN">AMZN</Option>
              <Option value="TSLA">TSLA</Option>
            </Select>
            <Input.Search 
              placeholder="Buscar símbolo..." 
              onSearch={handleSearch} 
              style={{ width: 200 }} 
            />
          </Input.Group>
        </div>
      </Header>
      <Content className="content">
        <div className="site-layout-content">
          {renderContent()}
        </div>
      </Content>
      <Footer style={{ textAlign: 'center' }}>
        Modelo Bursátil ©{new Date().getFullYear()} - Plataforma de Predicción y Trading
      </Footer>
    </Layout>
  );
};

export default App;