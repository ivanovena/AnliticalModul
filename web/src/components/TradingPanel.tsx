import React, { useState, useEffect } from 'react';
import { 
  Box, Typography, TextField, Button, 
  FormControl, InputLabel, Select, MenuItem,
  Grid, Divider, Alert, CircularProgress,
  InputAdornment 
} from '@mui/material';
import { MarketData, Order } from '../types/api';
import { useTradingContext } from '../contexts/TradingContext';

interface TradingPanelProps {
  symbol: string;
}

const TradingPanel: React.FC<TradingPanelProps> = ({ symbol }) => {
  // Estados locales para el formulario
  const [quantity, setQuantity] = useState<number>(1);
  const [price, setPrice] = useState<number>(0);
  const [action, setAction] = useState<'BUY' | 'SELL'>('BUY');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // Usar contexto global
  const { 
    marketData, 
    portfolio, 
    placeOrder,
    refreshPortfolio
  } = useTradingContext();
  
  // Obtener datos actuales del mercado
  const currentMarketData = marketData[symbol];
  
  // Actualizar precio cuando cambia el símbolo o llegan nuevos datos
  useEffect(() => {
    if (currentMarketData) {
      setPrice(currentMarketData.price);
    }
  }, [currentMarketData, symbol]);

  // Verificar si tenemos posición actual en este símbolo
  const hasPosition = portfolio?.positions && symbol in portfolio.positions;
  const currentPosition = hasPosition ? portfolio?.positions[symbol] : null;

  // Calcular valor total
  const totalValue = price * quantity;
  
  // Verificar si hay suficientes fondos para comprar
  const hasSufficientFunds = portfolio ? portfolio.cash >= totalValue : false;
  
  // Verificar si hay suficientes acciones para vender
  const hasSufficientShares = currentPosition ? currentPosition.quantity >= quantity : false;

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

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Panel de Trading
      </Typography>

      {/* Resumen de la posición actual */}
      <Box mb={2}>
        <Typography variant="subtitle2" color="text.secondary">
          Posición actual en {symbol}
        </Typography>
        {hasPosition ? (
          <Typography variant="body1">
            {currentPosition?.quantity} acciones a ${currentPosition?.avg_cost.toFixed(2)}
            {currentPosition?.current_profit !== undefined && (
              <span style={{ 
                color: currentPosition.current_profit >= 0 ? 'green' : 'red',
                marginLeft: '8px'
              }}>
                ({currentPosition.current_profit >= 0 ? '+' : ''}{currentPosition.current_profit.toFixed(2)} USD)
              </span>
            )}
          </Typography>
        ) : (
          <Typography variant="body1">
            No tienes posición en {symbol}
          </Typography>
        )}
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* Formulario de órdenes */}
      <form onSubmit={handleSubmit}>
        <Grid container spacing={2}>
          {/* Tipo de operación */}
          <Grid item xs={12}>
            <FormControl fullWidth size="small">
              <InputLabel id="action-select-label">Operación</InputLabel>
              <Select
                labelId="action-select-label"
                id="action-select"
                value={action}
                label="Operación"
                onChange={(e) => setAction(e.target.value as 'BUY' | 'SELL')}
              >
                <MenuItem value="BUY">Comprar</MenuItem>
                <MenuItem value="SELL">Vender</MenuItem>
              </Select>
            </FormControl>
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

          {/* Valor total */}
          <Grid item xs={12}>
            <Box 
              p={1} 
              bgcolor="background.paper"
              borderRadius={1}
              border="1px solid"
              borderColor="divider"
            >
              <Grid container justifyContent="space-between">
                <Typography variant="body2">Valor total:</Typography>
                <Typography variant="body1" fontWeight="bold">
                  ${totalValue.toFixed(2)}
                </Typography>
              </Grid>
            </Box>
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
              color={action === 'BUY' ? 'primary' : 'secondary'}
              fullWidth
              disabled={loading || (action === 'BUY' && !hasSufficientFunds) || (action === 'SELL' && !hasSufficientShares)}
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

      {/* Información sobre el portafolio */}
      <Box mt={2}>
        <Typography variant="subtitle2" color="text.secondary">
          Efectivo disponible
        </Typography>
        <Typography variant="body1">
          ${portfolio?.cash.toFixed(2) || '0.00'}
        </Typography>
      </Box>
    </Box>
  );
};

export default TradingPanel;