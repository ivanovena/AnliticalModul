// src/components/Trading/TradingSimulator.jsx
import React, { useState, useEffect } from 'react';
import { 
  Card, Form, Input, Button, Select, InputNumber, 
  Divider, Row, Col, Statistic, Alert, Modal, Table, Spin 
} from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, InfoCircleOutlined } from '@ant-design/icons';
import axios from 'axios';
import moment from 'moment';

const { Option } = Select;

const TradingSimulator = () => {
  const [form] = Form.useForm();
  const [portfolioData, setPortfolioData] = useState({
    availableCash: 0,
    positions: []
  });
  const [stockData, setStockData] = useState({});
  const [predictionData, setPredictionData] = useState({});
  const [simulationResult, setSimulationResult] = useState(null);
  const [isLoadingPortfolio, setIsLoadingPortfolio] = useState(true);
  const [isLoadingStock, setIsLoadingStock] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const [showSimulationResult, setShowSimulationResult] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  
  // Cargar datos del portfolio al iniciar
  useEffect(() => {
    fetchPortfolioData();
  }, []);
  
  const fetchPortfolioData = async () => {
    try {
      setIsLoadingPortfolio(true);
      const response = await axios.get('/api/portfolio');
      setPortfolioData({
        availableCash: response.data.cash,
        positions: response.data.positions || []
      });
      setIsLoadingPortfolio(false);
    } catch (error) {
      console.error('Error fetching portfolio data:', error);
      setErrorMessage('Error al cargar datos del portfolio');
      setIsLoadingPortfolio(false);
    }
  };
  
  const fetchStockData = async (symbol) => {
    if (!symbol) return;
    
    try {
      setIsLoadingStock(true);
      setErrorMessage('');
      
      // Obtener datos actuales del mercado
      const stockResponse = await axios.get(`/market-data/${symbol}`);
      
      // Obtener predicciones para el símbolo
      const predictionResponse = await axios.get(`/predictions/${symbol}`);
      
      setStockData(stockResponse.data);
      setPredictionData(predictionResponse.data);
      
      // Extraer precio del texto de análisis
      const analysisText = stockResponse.data.analysis || '';
      const priceMatch = analysisText.match(/\$(\d+\.\d+)/);
      const price = priceMatch ? parseFloat(priceMatch[1]) : 100;
      
      // Actualizar campo de precio en el formulario
      form.setFieldsValue({
        price: price
      });
      
      setIsLoadingStock(false);
    } catch (error) {
      console.error('Error fetching stock data:', error);
      setErrorMessage('Error al cargar datos del símbolo');
      setIsLoadingStock(false);
    }
  };
  
  const handleSymbolChange = (value) => {
    fetchStockData(value);
  };
  
  const handleQuantityChange = (value) => {
    // Recalcular costo total basado en la cantidad y precio
    const price = form.getFieldValue('price') || 0;
    const orderType = form.getFieldValue('orderType') || 'buy';
    
    simulateOrder(orderType, value, price);
  };
  
  const handlePriceChange = (value) => {
    // Recalcular costo total basado en la cantidad y precio
    const quantity = form.getFieldValue('quantity') || 0;
    const orderType = form.getFieldValue('orderType') || 'buy';
    
    simulateOrder(orderType, quantity, value);
  };
  
  const handleOrderTypeChange = (value) => {
    // Recalcular basado en el nuevo tipo de orden
    const quantity = form.getFieldValue('quantity') || 0;
    const price = form.getFieldValue('price') || 0;
    
    simulateOrder(value, quantity, price);
  };
  
  const simulateOrder = (orderType, quantity, price) => {
    if (!quantity || !price) {
      setSimulationResult(null);
      return;
    }
    
    // Calcular costo total
    const totalCost = quantity * price;
    
    // Verificar si hay suficientes fondos (para compra) o acciones (para venta)
    let canExecute = true;
    let errorMsg = '';
    
    if (orderType === 'buy') {
      if (totalCost > portfolioData.availableCash) {
        canExecute = false;
        errorMsg = 'Fondos insuficientes para esta operación';
      }
    } else {
      // Verificar si tenemos suficientes acciones para vender
      const symbol = form.getFieldValue('symbol');
      const position = portfolioData.positions.find(p => p.symbol === symbol);
      if (!position || position.quantity < quantity) {
        canExecute = false;
        errorMsg = 'No tienes suficientes acciones para vender';
      }
    }
    
    // Calcular balance proyectado después de la operación
    let projectedCash = portfolioData.availableCash;
    if (orderType === 'buy') {
      projectedCash -= totalCost;
    } else {
      projectedCash += totalCost;
    }
    
    // Calcular ganancia/pérdida potencial basada en predicciones
    let potentialGainLoss = 0;
    let potentialGainLossPercentage = 0;
    
    if (predictionData && predictionData.predictions) {
      // Extraer porcentaje de predicción del texto
      const predictionText = predictionData.predictions;
      const predictionMatch = predictionText.match(/\+(\d+\.\d+)%/) || predictionText.match(/\-(\d+\.\d+)%/);
      
      if (predictionMatch) {
        potentialGainLossPercentage = predictionMatch[0].startsWith('-') 
          ? -parseFloat(predictionMatch[1]) 
          : parseFloat(predictionMatch[1]);
          
        const priceDifference = price * (potentialGainLossPercentage / 100);
        potentialGainLoss = priceDifference * quantity;
      }
    }
    
    // Actualizar resultado de la simulación
    setSimulationResult({
      orderType,
      symbol: form.getFieldValue('symbol'),
      quantity,
      price,
      totalCost,
      projectedCash,
      canExecute,
      errorMsg,
      potentialGainLoss,
      potentialGainLossPercentage
    });
  };
  
  const handleSubmit = async (values) => {
    if (!simulationResult || !simulationResult.canExecute) {
      return;
    }
    
    setIsSimulating(true);
    
    try {
      // Enviar orden al servicio de broker
      const orderResponse = await axios.post('/api/order', {
        symbol: values.symbol,
        quantity: values.quantity,
        price: values.price,
        orderType: values.orderType,
        orderSource: 'simulator'
      });
      
      // Actualizar datos del portfolio después de la orden
      await fetchPortfolioData();
      
      // Mostrar resultados
      setShowSimulationResult(true);
      setIsSimulating(false);
      
      // Limpiar formulario
      form.resetFields();
      setSimulationResult(null);
    } catch (error) {
      console.error('Error executing order:', error);
      setErrorMessage('Error al ejecutar la orden');
      setIsSimulating(false);
    }
  };
  
  const closeSimulationResult = () => {
    setShowSimulationResult(false);
  };
  
  if (isLoadingPortfolio) {
    return <Spin tip="Cargando datos del portfolio..." />;
  }
  
  return (
    <div className="trading-simulator">
      <Card title="Simulador de Trading">
        {errorMessage && (
          <Alert message={errorMessage} type="error" showIcon style={{ marginBottom: 16 }} />
        )}
        
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          initialValues={{
            orderType: 'buy',
            quantity: 1
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="Símbolo"
                name="symbol"
                rules={[{ required: true, message: 'Por favor selecciona un símbolo' }]}
              >
                <Select
                  showSearch
                  placeholder="Selecciona un símbolo"
                  optionFilterProp="children"
                  onChange={handleSymbolChange}
                  loading={isLoadingStock}
                >
                  <Option value="AAPL">AAPL - Apple Inc.</Option>
                  <Option value="MSFT">MSFT - Microsoft Corp.</Option>
                  <Option value="GOOGL">GOOGL - Alphabet Inc.</Option>
                  <Option value="AMZN">AMZN - Amazon.com Inc.</Option>
                  <Option value="TSLA">TSLA - Tesla Inc.</Option>
                  <Option value="META">META - Meta Platforms Inc.</Option>
                  <Option value="NVDA">NVDA - NVIDIA Corp.</Option>
                  <Option value="JPM">JPM - JPMorgan Chase & Co.</Option>
                  <Option value="V">V - Visa Inc.</Option>
                  <Option value="JNJ">JNJ - Johnson & Johnson</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="Tipo de Orden"
                name="orderType"
                rules={[{ required: true, message: 'Por favor selecciona un tipo de orden' }]}
              >
                <Select onChange={handleOrderTypeChange}>
                  <Option value="buy">Comprar</Option>
                  <Option value="sell">Vender</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="Cantidad"
                name="quantity"
                rules={[{ required: true, message: 'Por favor ingresa una cantidad' }]}
              >
                <InputNumber
                  min={1}
                  style={{ width: '100%' }}
                  onChange={handleQuantityChange}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="Precio"
                name="price"
                rules={[{ required: true, message: 'Por favor ingresa un precio' }]}
              >
                <InputNumber
                  min={0.01}
                  step={0.01}
                  formatter={value => `$ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                  parser={value => value.replace(/\$\s?|(,*)/g, '')}
                  style={{ width: '100%' }}
                  onChange={handlePriceChange}
                />
              </Form.Item>
            </Col>
          </Row>
          
          <Divider />
          
          {simulationResult && (
            <div className="simulation-results">
              <h3>Resumen de la Operación</h3>
              
              <Row gutter={16}>
                <Col span={8}>
                  <Statistic
                    title="Costo Total"
                    value={simulationResult.totalCost}
                    precision={2}
                    prefix="$"
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="Efectivo Resultante"
                    value={simulationResult.projectedCash}
                    precision={2}
                    prefix="$"
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="Ganancia/Pérdida Potencial"
                    value={simulationResult.potentialGainLoss}
                    precision={2}
                    prefix="$"
                    valueStyle={{
                      color: simulationResult.potentialGainLoss >= 0 ? '#3f8600' : '#cf1322',
                    }}
                    suffix={
                      simulationResult.potentialGainLossPercentage ? 
                      `(${simulationResult.potentialGainLossPercentage.toFixed(2)}%)` : 
                      ''
                    }
                    prefix={simulationResult.potentialGainLoss >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                  />
                </Col>
              </Row>
              
              {!simulationResult.canExecute && (
                <Alert
                  message="Advertencia"
                  description={simulationResult.errorMsg}
                  type="warning"
                  showIcon
                  style={{ marginTop: 16 }}
                />
              )}
            </div>
          )}
          
          <div className="form-actions" style={{ marginTop: 16, textAlign: 'right' }}>
            <Button
              type="primary"
              htmlType="submit"
              loading={isSimulating}
              disabled={!simulationResult || !simulationResult.canExecute}
            >
              Ejecutar Operación
            </Button>
          </div>
        </Form>
      </Card>
      
      <Modal
        title="Resultado de la Operación"
        visible={showSimulationResult}
        onCancel={closeSimulationResult}
        footer={[
          <Button key="close" onClick={closeSimulationResult}>
            Cerrar
          </Button>
        ]}
      >
        <p>La operación se ha ejecutado correctamente.</p>
        <p>Tu portfolio ha sido actualizado.</p>
      </Modal>
    </div>
  );
};

export default TradingSimulator;