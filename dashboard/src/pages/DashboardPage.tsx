import { 
  Typography, 
  Container, 
  Paper, 
  Box, 
  Button,
  Card,
  CardContent,
  CardActions,
  Divider,
  Chip,
  Stack,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Tooltip
} from '@mui/material';
import { 
  ArrowUpward as ArrowUpIcon,
  ArrowDownward as ArrowDownIcon,
  ArrowRight as ArrowRightIcon,
  Refresh as RefreshIcon,
  Lightbulb as LightbulbIcon,
  ShowChart as ShowChartIcon,
  MonetizationOn as MonetizationOnIcon,
  Timeline as TimelineIcon,
  Settings as SettingsIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const DashboardPage = () => {
  const navigate = useNavigate();
  
  // Datos simulados para demostración
  const marketSummary = [
    { symbol: 'BTC-USD', name: 'Bitcoin', price: 45230.50, change: 2.45, volume: 28500000000 },
    { symbol: 'ETH-USD', name: 'Ethereum', price: 3150.25, change: -1.20, volume: 15700000000 },
    { symbol: 'AAPL', name: 'Apple Inc.', price: 190.75, change: 0.85, volume: 95600000 },
    { symbol: 'MSFT', name: 'Microsoft', price: 380.60, change: 1.10, volume: 28900000 },
  ];
  
  const recentPredictions = [
    { 
      id: 1, 
      symbol: 'BTC-USD', 
      predictedPrice: 47500, 
      actualPrice: 46800, 
      accuracy: 98.5, 
      timestamp: new Date(2023, 11, 5) 
    },
    { 
      id: 2, 
      symbol: 'ETH-USD', 
      predictedPrice: 3200, 
      actualPrice: 3050, 
      accuracy: 95.3, 
      timestamp: new Date(2023, 11, 4) 
    },
    { 
      id: 3, 
      symbol: 'AAPL', 
      predictedPrice: 195, 
      actualPrice: 191, 
      accuracy: 97.9, 
      timestamp: new Date(2023, 11, 3) 
    },
  ];
  
  const topRecommendations = [
    {
      id: 1,
      symbol: 'BTC-USD',
      action: 'BUY',
      confidence: 85,
      reasoning: 'Tendencia alcista con soporte fuerte',
    },
    {
      id: 2,
      symbol: 'AAPL',
      action: 'SELL',
      confidence: 79,
      reasoning: 'Sobrevaluada en indicadores técnicos',
    },
  ];
  
  const systemStatusSummary = {
    servicesUp: 6,
    servicesDown: 0,
    servicesDegraded: 1,
    cpuUsage: 32,
    memoryUsage: 68,
    predictionAccuracy: 97.2
  };
  
  const recentTransactions = [
    { id: 1, symbol: 'BTC-USD', type: 'BUY', quantity: 0.5, price: 43000, timestamp: new Date(2023, 10, 5, 14, 35) },
    { id: 2, symbol: 'ETH-USD', type: 'BUY', quantity: 3.2, price: 3200, timestamp: new Date(2023, 10, 7, 9, 12) },
    { id: 3, symbol: 'AAPL', type: 'SELL', quantity: 10, price: 195.5, timestamp: new Date(2023, 10, 8, 11, 45) },
  ];
  
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };
  
  const formatLargeNumber = (value: number) => {
    if (value >= 1e9) {
      return (value / 1e9).toFixed(1) + 'B';
    } else if (value >= 1e6) {
      return (value / 1e6).toFixed(1) + 'M';
    } else if (value >= 1e3) {
      return (value / 1e3).toFixed(1) + 'K';
    }
    return value.toString();
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Dashboard
        </Typography>
        <Button 
          variant="outlined" 
          startIcon={<RefreshIcon />}
          onClick={() => window.location.reload()}
        >
          Actualizar
        </Button>
      </Box>
      
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
        {/* Resumen del mercado */}
        <Box sx={{ flexBasis: { xs: '100%', md: 'calc(66.666% - 12px)' } }}>
          <Paper elevation={3} sx={{ p: 2, mb: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Resumen del Mercado
              </Typography>
              <Button 
                variant="text" 
                endIcon={<ShowChartIcon />}
                onClick={() => navigate('/market')}
              >
                Ver más
              </Button>
            </Box>
            
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
              {marketSummary.map((asset) => (
                <Box key={asset.symbol} sx={{ flexBasis: { xs: '100%', sm: 'calc(50% - 8px)', md: 'calc(25% - 12px)' } }}>
                  <Card variant="outlined">
                    <CardContent sx={{ p: 2, pb: 1 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <Box>
                          <Typography variant="subtitle2" color="text.secondary">
                            {asset.symbol}
                          </Typography>
                          <Typography variant="body2" noWrap>
                            {asset.name}
                          </Typography>
                        </Box>
                        <Chip 
                          size="small"
                          icon={asset.change >= 0 ? <ArrowUpIcon fontSize="small" /> : <ArrowDownIcon fontSize="small" />}
                          label={`${asset.change >= 0 ? '+' : ''}${asset.change}%`}
                          color={asset.change >= 0 ? 'success' : 'error'}
                        />
                      </Box>
                      
                      <Typography variant="h6" sx={{ my: 1 }}>
                        {formatCurrency(asset.price)}
                      </Typography>
                      
                      <Typography variant="caption" color="text.secondary">
                        Vol: {formatLargeNumber(asset.volume)}
                      </Typography>
                    </CardContent>
                  </Card>
                </Box>
              ))}
            </Box>
          </Paper>
        </Box>
        
        {/* Resumen del sistema */}
        <Box sx={{ flexBasis: { xs: '100%', md: 'calc(33.333% - 12px)' } }}>
          <Paper elevation={3} sx={{ p: 2, height: '100%' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Estado del Sistema
              </Typography>
              <Button 
                variant="text" 
                endIcon={<SettingsIcon />}
                onClick={() => navigate('/system')}
              >
                Ver más
              </Button>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Stack direction="row" spacing={1}>
                  <Chip 
                    size="small"
                    icon={<CheckCircleIcon />}
                    label={`${systemStatusSummary.servicesUp} UP`}
                    color="success"
                    variant="outlined"
                  />
                  <Chip 
                    size="small"
                    icon={<WarningIcon />}
                    label={`${systemStatusSummary.servicesDegraded} DEGRADED`}
                    color="warning"
                    variant="outlined"
                  />
                  <Chip 
                    size="small"
                    icon={<ErrorIcon />}
                    label={`${systemStatusSummary.servicesDown} DOWN`}
                    color="error"
                    variant="outlined"
                  />
                </Stack>
              </Box>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2">CPU</Typography>
              <LinearProgress 
                variant="determinate" 
                value={systemStatusSummary.cpuUsage} 
                color={systemStatusSummary.cpuUsage > 80 ? "error" : systemStatusSummary.cpuUsage > 60 ? "warning" : "success"}
                sx={{ height: 6, borderRadius: 1, my: 0.5 }}
              />
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption" color="text.secondary">
                  Uso
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {systemStatusSummary.cpuUsage}%
                </Typography>
              </Box>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2">Memoria</Typography>
              <LinearProgress 
                variant="determinate" 
                value={systemStatusSummary.memoryUsage} 
                color={systemStatusSummary.memoryUsage > 80 ? "error" : systemStatusSummary.memoryUsage > 60 ? "warning" : "success"}
                sx={{ height: 6, borderRadius: 1, my: 0.5 }}
              />
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption" color="text.secondary">
                  Uso
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {systemStatusSummary.memoryUsage}%
                </Typography>
              </Box>
            </Box>
            
            <Box>
              <Typography variant="body2">Precisión de Predicciones</Typography>
              <LinearProgress 
                variant="determinate" 
                value={systemStatusSummary.predictionAccuracy} 
                color="success"
                sx={{ height: 6, borderRadius: 1, my: 0.5 }}
              />
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption" color="text.secondary">
                  Precisión
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {systemStatusSummary.predictionAccuracy}%
                </Typography>
              </Box>
            </Box>
          </Paper>
        </Box>
        
        {/* Últimas predicciones */}
        <Box sx={{ flexBasis: { xs: '100%', md: 'calc(50% - 12px)' } }}>
          <Paper elevation={3} sx={{ p: 2, height: '100%' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Últimas Predicciones
              </Typography>
              <Button 
                variant="text" 
                endIcon={<TimelineIcon />}
                onClick={() => navigate('/predictions')}
              >
                Ver más
              </Button>
            </Box>
            
            <List>
              {recentPredictions.map((prediction) => (
                <ListItem key={prediction.id} sx={{ px: 1 }}>
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: prediction.accuracy > 95 ? 'success.light' : 'warning.light' }}>
                      {prediction.symbol.substring(0, 1)}
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText
                    primary={
                      <Typography variant="body2">
                        {prediction.symbol}
                      </Typography>
                    }
                    secondary={
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          {prediction.timestamp.toLocaleDateString()}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                          <Tooltip title="Precio predicho">
                            <Typography variant="body2" color="primary" sx={{ mr: 1 }}>
                              {formatCurrency(prediction.predictedPrice)}
                            </Typography>
                          </Tooltip>
                          <ArrowRightIcon sx={{ fontSize: 16, mx: 0.5 }} />
                          <Tooltip title="Precio real">
                            <Typography variant="body2">
                              {formatCurrency(prediction.actualPrice)}
                            </Typography>
                          </Tooltip>
                        </Box>
                      </Box>
                    }
                  />
                  <Box sx={{ textAlign: 'right' }}>
                    <Chip 
                      size="small"
                      label={`${prediction.accuracy}%`}
                      color={prediction.accuracy > 95 ? "success" : "warning"}
                    />
                  </Box>
                </ListItem>
              ))}
            </List>
            
            <Divider sx={{ my: 1 }} />
            
            <Box sx={{ textAlign: 'center' }}>
              <Button 
                variant="contained" 
                color="primary" 
                size="small"
                onClick={() => navigate('/predictions')}
              >
                Analizar Predicciones
              </Button>
            </Box>
          </Paper>
        </Box>
        
        {/* Recomendaciones destacadas */}
        <Box sx={{ flexBasis: { xs: '100%', md: 'calc(50% - 12px)' } }}>
          <Paper elevation={3} sx={{ p: 2, height: '100%' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                  />
                  <Box sx={{ textAlign: 'right' }}>
                    <Chip 
                      size="small"
                      label={`${prediction.accuracy}%`}
                      color={prediction.accuracy > 95 ? "success" : "warning"}
                    />
                  </Box>
                </ListItem>
              ))}
            </List>
            
            <Divider sx={{ my: 1 }} />
            
            <Box sx={{ textAlign: 'center' }}>
              <Button 
                variant="contained" 
                color="primary" 
                size="small"
                onClick={() => navigate('/predictions')}
              >
                Analizar Predicciones
              </Button>
            </Box>
          </Paper>
        </Grid>
        
        {/* Recomendaciones destacadas */}
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 2, height: '100%' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Recomendaciones Destacadas
              </Typography>
              <Button 
                variant="text" 
                endIcon={<LightbulbIcon />}
                onClick={() => navigate('/recommendations')}
              >
                Ver más
              </Button>
            </Box>
            
            <Grid container spacing={2}>
              {topRecommendations.map((recommendation) => (
                <Grid item xs={12} key={recommendation.id}>
                  <Card variant="outlined">
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <Chip 
                          icon={recommendation.action === 'BUY' ? <ArrowUpIcon /> : <ArrowDownIcon />}
                          label={recommendation.action}
                          color={recommendation.action === 'BUY' ? 'success' : 'error'}
                          size="small"
                          sx={{ mr: 1 }}
                        />
                        <Typography variant="subtitle1">
                          {recommendation.symbol}
                        </Typography>
                        <Box sx={{ ml: 'auto' }}>
                          <Tooltip title="Nivel de confianza">
                            <Box sx={{ 
                              display: 'inline-flex', 
                              alignItems: 'center', 
                              border: '2px solid', 
                              borderColor: recommendation.confidence >= 80 ? 'success.main' : 'warning.main',
                              borderRadius: '50%',
                              p: 0.5
                            }}>
                              <Typography variant="body2" component="div">
                                {recommendation.confidence}%
                              </Typography>
                            </Box>
                          </Tooltip>
                        </Box>
                      </Box>
                      
                      <Typography variant="body2" color="text.secondary">
                        {recommendation.reasoning}
                      </Typography>
                    </CardContent>
                    <CardActions>
                      <Button 
                        size="small" 
                        startIcon={<MonetizationOnIcon />}
                        onClick={() => navigate('/trading')}
                      >
                        Aplicar al Simulador
                      </Button>
                    </CardActions>
                  </Card>
                </Grid>
              ))}
            </Grid>
            
            <Divider sx={{ my: 2 }} />
            
            <Box sx={{ textAlign: 'center' }}>
              <Button 
                variant="contained" 
                color="primary" 
                size="small"
                startIcon={<LightbulbIcon />}
                onClick={() => navigate('/recommendations')}
              >
                Ver todas las recomendaciones
              </Button>
            </Box>
          </Paper>
        </Grid>
        
        {/* Últimas transacciones */}
        <Grid item xs={12}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Actividad Reciente - Simulador de Trading
              </Typography>
              <Button 
                variant="text" 
                endIcon={<MonetizationOnIcon />}
                onClick={() => navigate('/trading')}
              >
                Abrir Simulador
              </Button>
            </Box>
            
            <Grid container spacing={2}>
              {recentTransactions.map((transaction) => (
                <Grid item xs={12} sm={6} md={4} key={transaction.id}>
                  <Card variant="outlined">
                    <CardContent sx={{ p: 2, pb: '16px !important' }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Chip 
                          icon={transaction.type === 'BUY' ? <ArrowUpIcon /> : <ArrowDownIcon />}
                          label={transaction.type}
                          color={transaction.type === 'BUY' ? 'success' : 'error'}
                          size="small"
                        />
                        <Typography variant="caption" color="text.secondary">
                          {transaction.timestamp.toLocaleString()}
                        </Typography>
                      </Box>
                      
                      <Typography variant="subtitle1">
                        {transaction.symbol}
                      </Typography>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                        <Typography variant="body2" color="text.secondary">
                          Cantidad: {transaction.quantity}
                        </Typography>
                        <Typography variant="body2">
                          Precio: {formatCurrency(transaction.price)}
                        </Typography>
                      </Box>
                      
                      <Typography variant="body1" sx={{ mt: 1, textAlign: 'right', fontWeight: 500 }}>
                        Total: {formatCurrency(transaction.price * transaction.quantity)}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default DashboardPage;