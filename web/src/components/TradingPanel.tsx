import React, { useState, useEffect } from 'react';
import { 
  Box, Typography, TextField, Button, 
  FormControl, InputLabel, Select, MenuItem,
  Grid, Divider, Alert, CircularProgress,
  InputAdornment, Slider, Tabs, Tab, Card, CardContent,
  Chip, Tooltip, Paper, IconButton, Switch, FormControlLabel
} from '@mui/material';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';
import SellIcon from '@mui/icons-material/Sell';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import RefreshIcon from '@mui/icons-material/Refresh';
import VisibilityIcon from '@mui/icons-material/Visibility';
import { MarketData, Order } from '../types/api';
import { useTradingContext } from '../contexts/TradingContext';

interface TradingPanelProps {
  symbol: string;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel = (props: TabPanelProps) => {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`trading-tabpanel-${index}`}
      aria-labelledby={`trading-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ pt: 2 }}>
          {children}
        </Box>
      )}
    </div>
  );
};

const TradingPanel: React.FC<TradingPanelProps> = ({ symbol }) => {
  // Estados locales para la interfaz
  const [tabValue, setTabValue] = useState(0);
  const [quantity, setQuantity] = useState<number>(1);
  const [price, setPrice] = useState<number>(0);
  const [action, setAction] = useState<'BUY' | 'SELL'>('BUY');
  const [orderType, setOrderType] = useState<'market' | 'limit' | 'stop'>('market');
  const [stopLoss, setStopLoss] = useState<number | null>(null);
  const [takeProfit, setTakeProfit] = useState<number | null>(null);
  const [useStopLoss, setUseStopLoss] = useState<boolean>(false);
  const [useTakeProfit, setUseTakeProfit] = useState<boolean>(false);
  const [riskPercentage, setRiskPercentage] = useState<number>(2); // % del capital en riesgo
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [useRiskCalculation, setUseRiskCalculation] = useState<boolean>(true);
  const [riskAmount, setRiskAmount] = useState<number>(0);
  const [showOrderForm, setShowOrderForm] = useState<boolean>(true);
  const [showDetails, setShowDetails] = useState<boolean>(false);
  
  // Usar contexto global
  const { 
    marketData, 
    portfolio, 
    placeOrder,
    refreshPortfolio,
    predictions
  } = useTradingContext();
  
  // Obtener datos actuales del mercado
  const currentMarketData = marketData[symbol];
  const prediction = predictions[symbol];
  
  // Actualizar precio cuando cambia el símbolo o llegan nuevos datos
  useEffect(() => {
    if (currentMarketData && typeof currentMarketData.price === 'number') {
      setPrice(currentMarketData.price);
      
      // Establecer valores de stop loss y take profit basados en la predicción
      if (prediction && prediction.predictions) {
        const predValues = Object.values(prediction.predictions);
        const predValue = predValues.length > 0 ? predValues[0] : 0;
        
        if (predValue > 0) {
          // Si la predicción es positiva, establecer take profit
          setTakeProfit(currentMarketData.price * (1 + predValue / 100));
          setStopLoss(currentMarketData.price * 0.95); // 5% de stop loss por defecto
        } else {
          // Si la predicción es negativa, ajustar el stop loss
          setStopLoss(currentMarketData.price * 0.98); // 2% de stop loss más conservador
          setTakeProfit(currentMarketData.price * 1.05); // 5% de take profit por defecto
        }
      }
    }
  }, [currentMarketData, symbol, prediction]);

  // Verificar si tenemos posición actual en este símbolo
  const hasPosition = portfolio?.positions && symbol in portfolio.positions;
  const currentPosition = hasPosition ? portfolio?.positions[symbol] : null;

  // Calcular valor total
  const totalValue = price * quantity;
  
  // Calcular riesgo
  const calculateRiskAmount = () => {
    if (!portfolio) return 0;
    const amount = (portfolio.total_value * riskPercentage) / 100;
    return amount;
  };

  useEffect(() => {
    const amount = calculateRiskAmount();
    setRiskAmount(amount);
  }, [riskPercentage, portfolio]);
  
  // Calcular cantidad basada en el riesgo
  const calculateQuantityFromRisk = () => {
    const calculatedRiskAmount = calculateRiskAmount();
    if (stopLoss && price) {
      const riskPerShare = Math.abs(price - stopLoss);
      if (riskPerShare > 0) {
        return Math.floor(calculatedRiskAmount / riskPerShare);
      }
    }
    return quantity;
  };
  
  // Actualizar cantidad basada en el riesgo cuando cambian los parámetros relevantes
  useEffect(() => {
    if (useRiskCalculation && useStopLoss && stopLoss !== null) {
      const calculatedQty = calculateQuantityFromRisk();
      setQuantity(calculatedQty > 0 ? calculatedQty : 1);
    }
  }, [useRiskCalculation, useStopLoss, stopLoss, riskPercentage, price]);
  
  // Verificar si hay suficientes fondos para comprar
  const hasSufficientFunds = portfolio ? portfolio.cash >= totalValue : false;
  
  // Verificar si hay suficientes acciones para vender
  const hasSufficientShares = currentPosition ? currentPosition.quantity >= quantity : false;

  // Manejar cambio de pestaña
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Manejar cambio en el deslizador de riesgo
  const handleRiskChange = (event: Event, newValue: number | number[]) => {
    setRiskPercentage(newValue as number);
  };

  // Manejar envío del formulario
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!price || !quantity) {
      setError('Por favor ingresa precio y cantidad válidos');
      return;
    }
    
    // Validaciones adicionales
    if (action === 'BUY' && !hasSufficientFunds) {
      setError('No hay suficientes fondos para esta operación');
      return;
    }
    
    if (action === 'SELL' && !hasSufficientShares) {
      setError('No tienes suficientes acciones para vender');
      return;
    }
    
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      // Crear objeto de orden
      const order: Omit<Order, 'timestamp' | 'total_value'> = {
        symbol,
        action,
        quantity,
        price
      };
      
      // Enviar orden
      await placeOrder(order);
      
      // Actualizar portfolio
      await refreshPortfolio();
      
      // Mostrar mensaje de éxito
      setSuccess(`Orden de ${action === 'BUY' ? 'compra' : 'venta'} ejecutada con éxito`);
      
      // Resetear cantidad
      setQuantity(1);
    } catch (err) {
      setError(`Error al colocar la orden: ${err instanceof Error ? err.message : String(err)}`);
      console.error('Error placing order:', err);
    } finally {
      setLoading(false);
    }
  };

  // Calcular potencial de ganancia/pérdida
  const calculatePotentialGain = () => {
    if (action === 'BUY') {
      if (useTakeProfit && takeProfit) {
        const gainPerShare = takeProfit - price;
        return gainPerShare * quantity;
      }
    } else if (action === 'SELL') {
      if (useTakeProfit && takeProfit) {
        const gainPerShare = price - takeProfit;
        return gainPerShare * quantity;
      }
    }
    return 0;
  };

  // Calcular potencial de pérdida
  const calculatePotentialLoss = () => {
    if (action === 'BUY') {
      if (useStopLoss && stopLoss) {
        const lossPerShare = price - stopLoss;
        return lossPerShare * quantity;
      }
    } else if (action === 'SELL') {
      if (useStopLoss && stopLoss) {
        const lossPerShare = stopLoss - price;
        return lossPerShare * quantity;
      }
    }
    return 0;
  };

  // Calcular ratio riesgo/recompensa
  const calculateRiskRewardRatio = () => {
    const potentialGain = calculatePotentialGain();
    const potentialLoss = calculatePotentialLoss();
    
    if (potentialLoss <= 0) return 0;
    return potentialGain / potentialLoss;
  };

  // Función para formatear moneda
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('es-ES', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center' }}>
          Panel de Trading
          <Chip 
            label={currentMarketData && typeof currentMarketData.price === 'number' ? currentMarketData.price.toFixed(2) : '0.00'} 
            color="primary" 
            sx={{ ml: 1 }} 
          />
          {prediction && prediction.predictions && (
            <Tooltip title="Predicción de cambio en el precio">
              <Chip 
                icon={Object.values(prediction.predictions)[0] > 0 ? <TrendingUpIcon /> : <TrendingDownIcon />}
                label={`${Object.values(prediction.predictions)[0].toFixed(2)}%`}
                color={Object.values(prediction.predictions)[0] > 0 ? "success" : "error"}
                size="small"
                sx={{ ml: 1 }}
              />
            </Tooltip>
          )}
        </Typography>

        <Box display="flex" alignItems="center">
          <FormControlLabel
            control={
              <Switch 
                checked={showOrderForm} 
                onChange={(e) => setShowOrderForm(e.target.checked)}
                size="small"
              />
            }
            label="Formulario"
            sx={{ mr: 1 }}
          />
          <IconButton 
            size="small" 
            color="primary"
            onClick={() => refreshPortfolio()}
            title="Refrescar datos"
          >
            <RefreshIcon fontSize="small" />
          </IconButton>
        </Box>
      </Box>

      {/* Resumen de la posición actual */}
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent sx={{ pb: 1, '&:last-child': { pb: 1 } }}>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Posición actual en {symbol}
            </Typography>
            <IconButton 
              size="small" 
              onClick={() => setShowDetails(!showDetails)}
              title={showDetails ? "Ocultar detalles" : "Mostrar detalles"}
            >
              <VisibilityIcon fontSize="small" />
            </IconButton>
          </Box>
          {hasPosition ? (
            <>
              <Grid container spacing={1}>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="text.secondary">Cantidad</Typography>
                  <Typography variant="body1" fontWeight="medium">
                    {currentPosition?.quantity} acciones
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="text.secondary">Precio promedio</Typography>
                  <Typography variant="body1" fontWeight="medium">
                    ${currentPosition?.avg_cost.toFixed(2)}
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="text.secondary">Valor actual</Typography>
                  <Typography variant="body1" fontWeight="medium">
                    ${currentPosition?.market_value.toFixed(2)}
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="body2" color="text.secondary">P/L</Typography>
                  <Typography 
                    variant="body1" 
                    fontWeight="medium"
                    color={currentPosition?.current_profit && currentPosition.current_profit >= 0 ? 'success.main' : 'error.main'}
                  >
                    {currentPosition?.current_profit && currentPosition.current_profit >= 0 ? '+' : ''}
                    ${currentPosition?.current_profit?.toFixed(2) || '0.00'} ({currentPosition?.current_profit_pct?.toFixed(2) || '0.00'}%)
                  </Typography>
                </Grid>
              </Grid>
              
              {showDetails && (
                <Box mt={2}>
                  <Divider sx={{ my: 1 }} />
                  <Grid container spacing={1}>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary">Stop Loss recomendado</Typography>
                      <Typography variant="body2">
                        ${currentPosition && (currentPosition.avg_cost * 0.95).toFixed(2)} (-5%)
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary">Take Profit recomendado</Typography>
                      <Typography variant="body2">
                        ${currentPosition && (currentPosition.avg_cost * 1.1).toFixed(2)} (+10%)
                      </Typography>
                    </Grid>
                  </Grid>
                </Box>
              )}
            </>
          ) : (
            <Box textAlign="center" py={1}>
              <Typography variant="body2">
                No tienes posición en {symbol}
              </Typography>
              {showDetails && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                  Utiliza el formulario para realizar tu primera operación con este símbolo.
                </Typography>
              )}
            </Box>
          )}
        </CardContent>
      </Card>

      {showOrderForm && (
        <>
          {/* Pestañas para diferentes tipos de órdenes */}
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={tabValue} onChange={handleTabChange}>
              <Tab label="Básico" />
              <Tab label="Avanzado" />
            </Tabs>
          </Box>

          {/* Formulario de órdenes básico */}
          <TabPanel value={tabValue} index={0}>
            <form onSubmit={handleSubmit}>
              <Grid container spacing={2}>
                {/* Tipo de operación */}
                <Grid item xs={12}>
                  <Box display="flex" gap={1}>
                    <Button
                      variant={action === 'BUY' ? 'contained' : 'outlined'}
                      color="primary"
                      fullWidth
                      onClick={() => setAction('BUY')}
                      startIcon={<ShoppingCartIcon />}
                    >
                      Comprar
                    </Button>
                    <Button
                      variant={action === 'SELL' ? 'contained' : 'outlined'}
                      color="error"
                      fullWidth
                      onClick={() => setAction('SELL')}
                      startIcon={<SellIcon />}
                      disabled={!hasPosition}
                    >
                      Vender
                    </Button>
                  </Box>
                </Grid>

                {/* Precio */}
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Precio"
                    type="number"
                    value={price}
                    onChange={(e) => setPrice(parseFloat(e.target.value) || 0)}
                    inputProps={{ step: "0.01", min: 0.01 }}
                    size="small"
                    required
                    InputProps={{
                      startAdornment: <InputAdornment position="start">$</InputAdornment>,
                    }}
                  />
                </Grid>

                {/* Cantidad */}
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Cantidad"
                    type="number"
                    value={quantity}
                    onChange={(e) => setQuantity(parseInt(e.target.value) || 0)}
                    inputProps={{ min: 1 }}
                    size="small"
                    required
                  />
                </Grid>

                {/* Valor total */}
                <Grid item xs={12}>
                  <Paper variant="outlined" sx={{ p: 1.5, bgcolor: 'background.paper' }}>
                    <Grid container justifyContent="space-between" alignItems="center">
                      <Typography variant="body2">Valor total:</Typography>
                      <Typography variant="h6" fontWeight="bold">
                        ${totalValue.toFixed(2)}
                      </Typography>
                    </Grid>
                    {action === 'BUY' && (
                      <Grid container justifyContent="space-between" alignItems="center" mt={1}>
                        <Typography variant="caption" color="text.secondary">Saldo disponible:</Typography>
                        <Typography variant="body2" color={hasSufficientFunds ? 'success.main' : 'error.main'}>
                          ${portfolio?.cash.toFixed(2) || '0.00'}
                        </Typography>
                      </Grid>
                    )}
                  </Paper>
                </Grid>

                {/* Mensajes de error o éxito */}
                {error && (
                  <Grid item xs={12}>
                    <Alert severity="error">{error}</Alert>
                  </Grid>
                )}
                
                {success && (
                  <Grid item xs={12}>
                    <Alert severity="success">{success}</Alert>
                  </Grid>
                )}

                {/* Botón de envío */}
                <Grid item xs={12}>
                  <Button
                    type="submit"
                    variant="contained"
                    color={action === 'BUY' ? 'primary' : 'error'}
                    fullWidth
                    size="large"
                    disabled={loading || (action === 'BUY' && !hasSufficientFunds) || (action === 'SELL' && !hasSufficientShares)}
                    startIcon={action === 'BUY' ? <ShoppingCartIcon /> : <SellIcon />}
                  >
                    {loading ? (
                      <CircularProgress size={24} color="inherit" />
                    ) : (
                      action === 'BUY' ? 'Comprar' : 'Vender'
                    )}
                  </Button>
                </Grid>
              </Grid>
            </form>
          </TabPanel>

          {/* Formulario de órdenes avanzado */}
          <TabPanel value={tabValue} index={1}>
            <form onSubmit={handleSubmit}>
              <Grid container spacing={2}>
                {/* Tipo de operación y orden */}
                <Grid item xs={7}>
                  <Box display="flex" gap={1}>
                    <Button
                      variant={action === 'BUY' ? 'contained' : 'outlined'}
                      color="primary"
                      fullWidth
                      onClick={() => setAction('BUY')}
                      startIcon={<ShoppingCartIcon />}
                    >
                      Comprar
                    </Button>
                    <Button
                      variant={action === 'SELL' ? 'contained' : 'outlined'}
                      color="error"
                      fullWidth
                      onClick={() => setAction('SELL')}
                      startIcon={<SellIcon />}
                      disabled={!hasPosition}
                    >
                      Vender
                    </Button>
                  </Box>
                </Grid>

                {/* Tipo de orden */}
                <Grid item xs={5}>
                  <FormControl fullWidth size="small">
                    <InputLabel id="order-type-label">Tipo de orden</InputLabel>
                    <Select
                      labelId="order-type-label"
                      value={orderType}
                      label="Tipo de orden"
                      onChange={(e) => setOrderType(e.target.value as 'market' | 'limit' | 'stop')}
                    >
                      <MenuItem value="market">Mercado</MenuItem>
                      <MenuItem value="limit">Límite</MenuItem>
                      <MenuItem value="stop">Stop</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                {/* Precio y cantidad */}
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label={orderType === 'stop' ? 'Precio stop' : 'Precio'}
                    type="number"
                    value={price}
                    onChange={(e) => setPrice(parseFloat(e.target.value) || 0)}
                    inputProps={{ step: "0.01", min: 0.01 }}
                    size="small"
                    required
                    InputProps={{
                      startAdornment: <InputAdornment position="start">$</InputAdornment>,
                    }}
                    disabled={orderType === 'market'}
                  />
                </Grid>

                {/* Cantidad */}
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Cantidad"
                    type="number"
                    value={quantity}
                    onChange={(e) => setQuantity(parseInt(e.target.value) || 0)}
                    inputProps={{ min: 1 }}
                    size="small"
                    required
                  />
                </Grid>

                {/* Manejo de riesgo */}
                <Grid item xs={12}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Gestión de Riesgo
                      <Tooltip title="Configura el riesgo y los niveles automáticos de stop loss y take profit">
                        <InfoOutlinedIcon fontSize="small" sx={{ ml: 1, verticalAlign: 'middle', opacity: 0.7 }} />
                      </Tooltip>
                    </Typography>
                    
                    <Grid container spacing={2}>
                      <Grid item xs={12}>
                        <FormControlLabel
                          control={
                            <Switch 
                              checked={useRiskCalculation} 
                              onChange={(e) => setUseRiskCalculation(e.target.checked)}
                              size="small"
                            />
                          }
                          label="Calcular cantidad basada en riesgo"
                        />
                      </Grid>
                      
                      {useRiskCalculation && (
                        <>
                          <Grid item xs={12}>
                            <Typography variant="body2" gutterBottom>
                              Riesgo del capital: {riskPercentage}%
                            </Typography>
                            <Slider
                              value={riskPercentage}
                              onChange={handleRiskChange}
                              aria-labelledby="risk-percentage-slider"
                              valueLabelDisplay="auto"
                              step={0.5}
                              marks
                              min={1}
                              max={10}
                            />
                            <Typography variant="caption" color="text.secondary">
                              Monto en riesgo: ${riskAmount.toFixed(2)}
                            </Typography>
                          </Grid>
                          
                          <Grid item xs={6}>
                            <FormControlLabel
                              control={
                                <Switch 
                                  checked={useStopLoss} 
                                  onChange={(e) => setUseStopLoss(e.target.checked)}
                                  size="small"
                                />
                              }
                              label="Stop Loss"
                            />
                            <TextField
                              fullWidth
                              label="Nivel Stop Loss"
                              type="number"
                              value={stopLoss || ''}
                              onChange={(e) => setStopLoss(parseFloat(e.target.value) || null)}
                              disabled={!useStopLoss}
                              size="small"
                              InputProps={{
                                startAdornment: <InputAdornment position="start">$</InputAdornment>,
                              }}
                              sx={{ mt: 1 }}
                            />
                          </Grid>
                          
                          <Grid item xs={6}>
                            <FormControlLabel
                              control={
                                <Switch 
                                  checked={useTakeProfit} 
                                  onChange={(e) => setUseTakeProfit(e.target.checked)}
                                  size="small"
                                />
                              }
                              label="Take Profit"
                            />
                            <TextField
                              fullWidth
                              label="Nivel Take Profit"
                              type="number"
                              value={takeProfit || ''}
                              onChange={(e) => setTakeProfit(parseFloat(e.target.value) || null)}
                              disabled={!useTakeProfit}
                              size="small"
                              InputProps={{
                                startAdornment: <InputAdornment position="start">$</InputAdornment>,
                              }}
                              sx={{ mt: 1 }}
                            />
                          </Grid>
                        </>
                      )}
                    </Grid>
                  </Paper>
                </Grid>

                {/* Resumen de la operación */}
                <Grid item xs={12}>
                  <Paper variant="outlined" sx={{ p: 1.5, bgcolor: 'background.paper' }}>
                    <Typography variant="subtitle2" gutterBottom>Resumen de la Operación</Typography>
                    
                    <Grid container spacing={1}>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">Valor total:</Typography>
                        <Typography variant="body1" fontWeight="medium">${totalValue.toFixed(2)}</Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">Comisión estimada:</Typography>
                        <Typography variant="body1" fontWeight="medium">${(totalValue * 0.001).toFixed(2)}</Typography>
                      </Grid>
                      
                      {useStopLoss && stopLoss && (
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">Pérdida máxima:</Typography>
                          <Typography variant="body1" color="error.main">
                            -${Math.abs(calculatePotentialLoss()).toFixed(2)}
                          </Typography>
                        </Grid>
                      )}
                      
                      {useTakeProfit && takeProfit && (
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">Ganancia potencial:</Typography>
                          <Typography variant="body1" color="success.main">
                            +${calculatePotentialGain().toFixed(2)}
                          </Typography>
                        </Grid>
                      )}
                      
                      {useStopLoss && useTakeProfit && stopLoss && takeProfit && (
                        <Grid item xs={12}>
                          <Divider sx={{ my: 1 }} />
                          <Typography variant="body2" color="text.secondary">Ratio Riesgo/Recompensa:</Typography>
                          <Typography variant="body1" fontWeight="medium">
                            1:{calculateRiskRewardRatio().toFixed(2)}
                          </Typography>
                        </Grid>
                      )}
                    </Grid>
                    
                    {action === 'BUY' && (
                      <Box mt={1}>
                        <Typography variant="caption" color="text.secondary">
                          Saldo disponible: <span style={{ color: hasSufficientFunds ? 'green' : 'red' }}>
                            ${portfolio?.cash.toFixed(2) || '0.00'}
                          </span>
                        </Typography>
                      </Box>
                    )}
                  </Paper>
                </Grid>

                {/* Mensajes de error o éxito */}
                {error && (
                  <Grid item xs={12}>
                    <Alert severity="error">{error}</Alert>
                  </Grid>
                )}
                
                {success && (
                  <Grid item xs={12}>
                    <Alert severity="success">{success}</Alert>
                  </Grid>
                )}

                {/* Botón de envío */}
                <Grid item xs={12}>
                  <Button
                    type="submit"
                    variant="contained"
                    color={action === 'BUY' ? 'primary' : 'error'}
                    fullWidth
                    size="large"
                    disabled={loading || (action === 'BUY' && !hasSufficientFunds) || (action === 'SELL' && !hasSufficientShares)}
                  >
                    {loading ? (
                      <CircularProgress size={24} color="inherit" />
                    ) : (
                      `${action === 'BUY' ? 'Comprar' : 'Vender'} ${quantity} ${symbol} a $${price}`
                    )}
                  </Button>
                </Grid>
              </Grid>
            </form>
          </TabPanel>
        </>
      )}
    </Box>
  );
};

export default TradingPanel;