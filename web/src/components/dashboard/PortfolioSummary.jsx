import React, { useState } from 'react';
import { Card, CardContent, Typography, Button, Grid, TextField, Dialog, DialogActions, DialogContent, DialogTitle, CircularProgress } from '@mui/material';
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import MoneyIcon from '@mui/icons-material/Money';
import { usePortfolio } from '../../contexts/PortfolioContext';

const PortfolioSummary = () => {
  const { portfolio, initialCash, updateInitialCash, resetPortfolio, loading, error } = usePortfolio();
  const [openDialog, setOpenDialog] = useState(false);
  const [newAmount, setNewAmount] = useState(initialCash);
  const [openResetDialog, setOpenResetDialog] = useState(false);

  // Calcular ganancias y pérdidas
  const cashAvailable = portfolio?.cash || 0;
  const investedAmount = portfolio?.totalValue ? portfolio.totalValue - cashAvailable : 0;
  const profitLoss = portfolio?.totalValue ? portfolio.totalValue - portfolio.initialCash : 0;
  const profitLossPercentage = portfolio?.initialCash ? (profitLoss / portfolio.initialCash) * 100 : 0;

  // Manejar cambio de cantidad inicial
  const handleAmountChange = (event) => {
    setNewAmount(parseFloat(event.target.value));
  };

  // Abrir diálogo
  const handleOpenDialog = () => {
    setOpenDialog(true);
  };

  // Cerrar diálogo
  const handleCloseDialog = () => {
    setOpenDialog(false);
  };

  // Confirmar cambio
  const handleConfirmChange = () => {
    updateInitialCash(newAmount);
    setOpenDialog(false);
  };

  // Abrir diálogo de reinicio
  const handleOpenResetDialog = () => {
    setOpenResetDialog(true);
  };

  // Cerrar diálogo de reinicio
  const handleCloseResetDialog = () => {
    setOpenResetDialog(false);
  };

  // Confirmar reinicio
  const handleConfirmReset = () => {
    resetPortfolio();
    setOpenResetDialog(false);
  };

  return (
    <Card 
      elevation={3} 
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        position: 'relative'
      }}
    >
      {loading && (
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          backgroundColor: 'rgba(255, 255, 255, 0.7)',
          zIndex: 1
        }}>
          <CircularProgress />
        </div>
      )}

      <CardContent sx={{ flexGrow: 1 }}>
        <Typography variant="h6" gutterBottom>
          Resumen de Cartera
        </Typography>
        
        {error ? (
          <Typography color="error" variant="body2" sx={{ mb: 2 }}>
            Error: {error}. Mostrando últimos datos disponibles.
          </Typography>
        ) : null}
        
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Typography variant="body2" color="textSecondary">
              <MoneyIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
              Capital Inicial
            </Typography>
            <Typography variant="h6">
              ${portfolio?.initialCash?.toLocaleString('es-ES', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
            </Typography>
          </Grid>
          
          <Grid item xs={6}>
            <Typography variant="body2" color="textSecondary">
              <AccountBalanceWalletIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
              Efectivo Disponible
            </Typography>
            <Typography variant="h6">
              ${cashAvailable.toLocaleString('es-ES', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </Typography>
          </Grid>
          
          <Grid item xs={6}>
            <Typography variant="body2" color="textSecondary">
              <MoneyIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
              Capital Invertido
            </Typography>
            <Typography variant="h6">
              ${investedAmount.toLocaleString('es-ES', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </Typography>
          </Grid>
          
          <Grid item xs={6}>
            <Typography variant="body2" color="textSecondary">
              <MoneyIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
              Valor Total
            </Typography>
            <Typography variant="h6">
              ${portfolio?.totalValue?.toLocaleString('es-ES', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
            </Typography>
          </Grid>
          
          <Grid item xs={12}>
            <Typography variant="body2" color="textSecondary">
              {profitLoss >= 0 ? (
                <TrendingUpIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5, color: 'success.main' }} />
              ) : (
                <TrendingDownIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5, color: 'error.main' }} />
              )}
              Ganancia/Pérdida
            </Typography>
            <Typography 
              variant="h6" 
              color={profitLoss >= 0 ? 'success.main' : 'error.main'}
            >
              ${profitLoss.toLocaleString('es-ES', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              {' '}
              ({profitLossPercentage.toFixed(2)}%)
            </Typography>
          </Grid>
        </Grid>
        
        <Grid container spacing={1} sx={{ mt: 2 }}>
          <Grid item xs={6}>
            <Button 
              variant="outlined" 
              size="small" 
              fullWidth
              onClick={handleOpenDialog}
            >
              Cambiar Capital
            </Button>
          </Grid>
          <Grid item xs={6}>
            <Button 
              variant="outlined" 
              size="small" 
              color="error"
              fullWidth
              onClick={handleOpenResetDialog}
            >
              Reiniciar Cartera
            </Button>
          </Grid>
        </Grid>
      </CardContent>
      
      {/* Diálogo para cambiar el capital inicial */}
      <Dialog open={openDialog} onClose={handleCloseDialog}>
        <DialogTitle>Cambiar Capital Inicial</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Nuevo Capital Inicial ($)"
            type="number"
            fullWidth
            value={newAmount}
            onChange={handleAmountChange}
            inputProps={{ min: "1000", step: "1000" }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog} color="primary">
            Cancelar
          </Button>
          <Button onClick={handleConfirmChange} color="primary">
            Confirmar
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Diálogo para reiniciar cartera */}
      <Dialog open={openResetDialog} onClose={handleCloseResetDialog}>
        <DialogTitle>¿Reiniciar Cartera?</DialogTitle>
        <DialogContent>
          <Typography>
            Esta acción eliminará todas tus posiciones y transacciones, y reiniciará tu cartera con el capital inicial de ${initialCash?.toLocaleString('es-ES')}.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseResetDialog} color="primary">
            Cancelar
          </Button>
          <Button onClick={handleConfirmReset} color="error">
            Reiniciar
          </Button>
        </DialogActions>
      </Dialog>
    </Card>
  );
};

export default PortfolioSummary; 