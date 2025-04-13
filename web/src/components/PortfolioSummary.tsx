import React, { useState } from 'react';
import {
  Box, Typography, Grid, Paper, Divider,
  Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Chip, LinearProgress,
  Card, CardContent, Button, TextField, Dialog,
  DialogActions, DialogContent, DialogContentText,
  DialogTitle, InputAdornment, Snackbar, Alert
} from '@mui/material';
import { useTradingContext } from '../contexts/TradingContext';
import { Position } from '../types/api';

const PortfolioSummary: React.FC = () => {
  const { portfolio, metrics, placeOrder } = useTradingContext();
  const [openDialog, setOpenDialog] = useState(false);
  const [initialCapital, setInitialCapital] = useState(100000);
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState<'success' | 'error'>('success');
  const [isProcessing, setIsProcessing] = useState(false);

  const handleDialogOpen = () => {
    setOpenDialog(true);
    if (portfolio) {
      setInitialCapital(portfolio.total_value);
    }
  };

  const handleDialogClose = () => {
    setOpenDialog(false);
  };

  const handleSetInitialCapital = async () => {
    setIsProcessing(true);
    try {
      // Crear una orden de tipo "INIT_CAPITAL" para actualizar el capital inicial
      await placeOrder({
        symbol: "CASH", // Símbolo especial para operaciones de caja
        action: "INIT_CAPITAL" as any, // Tipo especial de acción para inicializar capital
        quantity: 1,
        price: initialCapital // El precio aquí representa el monto del capital inicial
      });
      
      // Mostrar mensaje de éxito
      setSnackbarMessage(`Capital inicial actualizado a $${initialCapital.toFixed(2)}`);
      setSnackbarSeverity('success');
      setOpenSnackbar(true);
      
      // Cerrar el diálogo
      setOpenDialog(false);
    } catch (error) {
      console.error("Error al actualizar capital inicial:", error);
      setSnackbarMessage("Error al actualizar el capital inicial. Intente nuevamente.");
      setSnackbarSeverity('error');
      setOpenSnackbar(true);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSnackbarClose = () => {
    setOpenSnackbar(false);
  };

  if (!portfolio) {
    return (
      <Box p={2}>
        <Typography variant="h6" gutterBottom>
          Resumen del Portafolio
        </Typography>
        <Typography color="text.secondary">
          Cargando información del portafolio...
        </Typography>
      </Box>
    );
  }

  // Calcular estadísticas del portafolio
  const totalPositionsValue = Object.values(portfolio.positions).reduce(
    (sum, position) => sum + position.market_value, 0
  );
  
  const totalProfit = Object.values(portfolio.positions).reduce(
    (sum, position) => sum + (position.current_profit || 0), 0
  );
  
  const totalProfitPercent = totalProfit / (portfolio.total_value - totalProfit) * 100;

  // Ordenar posiciones por valor de mercado (de mayor a menor)
  const sortedPositions = Object.values(portfolio.positions)
    .sort((a, b) => b.market_value - a.market_value);

  // Calcular diversificación
  const diversificationScore = metrics?.risk_metrics.portfolio.diversification_score || 0;

  // Función para obtener el precio actual desde métricas
  const getCurrentPrice = (symbol: string): number => {
    if (metrics?.stock_performance?.[symbol]?.current_price) {
      return metrics.stock_performance[symbol].current_price;
    }
    // Si no hay métricas, calculamos una estimación
    const position = portfolio.positions[symbol];
    return position.market_value / position.quantity;
  };

  // Función para obtener el porcentaje de ganancia/pérdida
  const getProfitPercentage = (symbol: string): number => {
    // Primero verificamos si está en las métricas
    if (metrics?.stock_performance?.[symbol]?.profit_percent) {
      return metrics.stock_performance[symbol].profit_percent;
    }
    
    // Si no, calculamos una estimación basada en la posición
    const position = portfolio.positions[symbol];
    if (position.current_profit === undefined) return 0;
    const costBasis = position.avg_cost * position.quantity;
    return costBasis > 0 ? (position.current_profit / costBasis) * 100 : 0;
  };

  // Estimar el efectivo inicial (podría ser cash + profit)
  const estimatedInitialCash = portfolio.cash - totalProfit;

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">
          Resumen del Portafolio
        </Typography>
        <Button 
          variant="outlined" 
          size="small" 
          onClick={handleDialogOpen}
        >
          Establecer Capital Inicial
        </Button>
      </Box>

      {/* Tarjetas con métricas principales */}
      <Grid container spacing={2} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Valor Total
              </Typography>
              <Typography variant="h5">
                ${portfolio.total_value.toFixed(2)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Efectivo Disponible
              </Typography>
              <Typography variant="h5">
                ${portfolio.cash.toFixed(2)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {((portfolio.cash / portfolio.total_value) * 100).toFixed(1)}% del total
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Ganancia/Pérdida
              </Typography>
              <Typography 
                variant="h5" 
                color={totalProfit >= 0 ? 'success.main' : 'error.main'}
              >
                {totalProfit >= 0 ? '+' : ''}{totalProfit.toFixed(2)} USD
              </Typography>
              <Typography 
                variant="body2" 
                color={totalProfitPercent >= 0 ? 'success.main' : 'error.main'}
              >
                {totalProfitPercent >= 0 ? '+' : ''}{totalProfitPercent.toFixed(2)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Diversificación
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={diversificationScore * 100} 
                color={
                  diversificationScore > 0.7 ? "success" :
                  diversificationScore > 0.4 ? "info" : "warning"
                }
                sx={{ mt: 1, mb: 1, height: 8, borderRadius: 4 }}
              />
              <Typography variant="body2" color="text.secondary">
                {(diversificationScore * 100).toFixed(0)}% - {
                  diversificationScore > 0.7 ? "Buena" :
                  diversificationScore > 0.4 ? "Media" : "Baja"
                }
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabla de posiciones */}
      <Typography variant="subtitle1" gutterBottom>
        Posiciones Actuales
      </Typography>
      
      {sortedPositions.length > 0 ? (
        <TableContainer component={Paper} sx={{ mb: 3 }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Símbolo</TableCell>
                <TableCell align="right">Cantidad</TableCell>
                <TableCell align="right">Precio Actual</TableCell>
                <TableCell align="right">Precio Promedio</TableCell>
                <TableCell align="right">Valor</TableCell>
                <TableCell align="right">Ganancia/Pérdida</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sortedPositions.map((position: Position) => {
                const currentPrice = getCurrentPrice(position.symbol);
                const profitPercent = getProfitPercentage(position.symbol);
                
                return (
                  <TableRow key={position.symbol}>
                    <TableCell component="th" scope="row">
                      {position.symbol}
                    </TableCell>
                    <TableCell align="right">{position.quantity}</TableCell>
                    <TableCell align="right">${currentPrice.toFixed(2)}</TableCell>
                    <TableCell align="right">${position.avg_cost.toFixed(2)}</TableCell>
                    <TableCell align="right">${position.market_value.toFixed(2)}</TableCell>
                    <TableCell align="right">
                      {position.current_profit !== undefined && (
                        <>
                          <Typography 
                            variant="body2" 
                            color={position.current_profit >= 0 ? 'success.main' : 'error.main'}
                          >
                            {position.current_profit >= 0 ? '+' : ''}{position.current_profit.toFixed(2)} USD
                          </Typography>
                          <Chip
                            label={`${profitPercent ? 
                              (profitPercent >= 0 ? '+' : '') + profitPercent.toFixed(2) : '0.00'}%`}
                            size="small"
                            color={profitPercent >= 0 ? 'success' : 'error'}
                          />
                        </>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
      ) : (
        <Typography color="text.secondary" sx={{ mt: 2, mb: 3 }}>
          No hay posiciones activas en el portafolio.
        </Typography>
      )}

      {/* Distribución del portafolio */}
      <Typography variant="subtitle1" gutterBottom>
        Distribución del Portafolio
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                Distribución por Activo
              </Typography>
              <Box mt={2} height={200} display="flex" flexDirection="column" justifyContent="center">
                {sortedPositions.length > 0 ? (
                  sortedPositions.map((position) => (
                    <Box key={position.symbol} mb={1}>
                      <Grid container justifyContent="space-between" alignItems="center">
                        <Typography variant="body2">{position.symbol}</Typography>
                        <Typography variant="body2">
                          {((position.market_value / portfolio.total_value) * 100).toFixed(1)}%
                        </Typography>
                      </Grid>
                      <LinearProgress 
                        variant="determinate" 
                        value={(position.market_value / portfolio.total_value) * 100} 
                        sx={{ height: 8, borderRadius: 4 }}
                      />
                    </Box>
                  ))
                ) : (
                  <Typography color="text.secondary" align="center">
                    No hay posiciones para mostrar
                  </Typography>
                )}
                
                {/* Efectivo */}
                <Box mb={1}>
                  <Grid container justifyContent="space-between" alignItems="center">
                    <Typography variant="body2">Efectivo</Typography>
                    <Typography variant="body2">
                      {((portfolio.cash / portfolio.total_value) * 100).toFixed(1)}%
                    </Typography>
                  </Grid>
                  <LinearProgress 
                    variant="determinate" 
                    value={(portfolio.cash / portfolio.total_value) * 100} 
                    color="success"
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                Rendimiento del Portafolio
              </Typography>
              <Box mt={3} display="flex" justifyContent="center" alignItems="center" height={150}>
                {/* Aquí se podría agregar un gráfico de rendimiento */}
                <Box textAlign="center">
                  <Typography variant="h4" color={totalProfit >= 0 ? 'success.main' : 'error.main'}>
                    {totalProfitPercent >= 0 ? '+' : ''}{totalProfitPercent.toFixed(2)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Rendimiento desde el inicio
                  </Typography>
                  <Box mt={2}>
                    <Chip 
                      label={`Efectivo inicial: $${estimatedInitialCash.toFixed(2)}`} 
                      size="small" 
                      sx={{ mr: 1 }} 
                    />
                    <Chip 
                      label={`Valor actual: $${portfolio.total_value.toFixed(2)}`} 
                      size="small" 
                      color="primary" 
                    />
                  </Box>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Diálogo para establecer capital inicial */}
      <Dialog open={openDialog} onClose={handleDialogClose}>
        <DialogTitle>Establecer Capital Inicial</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Introduce el monto de capital inicial para tu cartera de inversiones. Esto reiniciará tu cartera actual.
          </DialogContentText>
          <TextField
            autoFocus
            margin="dense"
            label="Capital Inicial"
            type="number"
            fullWidth
            variant="outlined"
            value={initialCapital}
            onChange={(e) => setInitialCapital(Number(e.target.value))}
            InputProps={{
              startAdornment: <InputAdornment position="start">$</InputAdornment>,
            }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDialogClose} disabled={isProcessing}>Cancelar</Button>
          <Button 
            onClick={handleSetInitialCapital} 
            variant="contained" 
            disabled={isProcessing}
          >
            {isProcessing ? "Procesando..." : "Establecer"}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar para mostrar mensajes */}
      <Snackbar 
        open={openSnackbar} 
        autoHideDuration={6000} 
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleSnackbarClose} 
          severity={snackbarSeverity}
          variant="filled"
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default PortfolioSummary;