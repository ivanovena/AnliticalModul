import React, { useState } from 'react';
import { 
  Container, Grid, Paper, Box, Typography, 
  AppBar, Toolbar, IconButton, MenuItem, FormControl, 
  InputLabel, Select, CircularProgress, Tabs, Tab,
  Card, CardContent, Divider, Button, TextField, InputAdornment,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import { useTradingContext } from '../contexts/TradingContext';

// Importación de componentes
import StockChart from '../components/StockChart';
import PortfolioSummary from '../components/PortfolioSummary';
import TradingPanel from '../components/TradingPanel';
import PredictionCard from '../components/PredictionCard';
import AIChat from '../components/AIChat';
import ModelStatus from '../components/ModelStatus';
import TradingRecommendations from '../components/TradingRecommendations';

// Define the type for a single market overview item
type MarketOverviewItem = {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  prediction: number;
  direction: string;
  volume: number;
  high: number;
  low: number;
};

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
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
      style={{ height: '100%' }}
    >
      {value === index && (
        <Box sx={{ p: 2, height: '100%' }}>
          {children}
        </Box>
      )}
    </div>
  );
};

const Dashboard: React.FC = () => {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [tabValue, setTabValue] = useState(0);
  const [initialCapital, setInitialCapital] = useState<number>(100000);
  const [showCapitalInput, setShowCapitalInput] = useState<boolean>(false);
  
  const { 
    selectedSymbol, 
    setSelectedSymbol, 
    availableSymbols, 
    marketData, 
    predictions,
    portfolio,
    isLoading,
    error,
    placeOrder
  } = useTradingContext();

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleCapitalSubmit = async () => {
    try {
      // Crear una orden de tipo "INIT_CAPITAL" para actualizar el capital inicial
      await placeOrder({
        symbol: "CASH", // Símbolo especial para operaciones de caja
        action: "INIT_CAPITAL" as any, // Tipo especial de acción para inicializar capital
        quantity: 1,
        price: initialCapital // El precio aquí representa el monto del capital inicial
      });
      
      // Mostrar mensaje de éxito
      alert(`Capital inicial actualizado a $${initialCapital.toFixed(2)}`);
      
      // Ocultar el formulario
      setShowCapitalInput(false);
    } catch (error) {
      console.error("Error al actualizar capital inicial:", error);
      alert("Error al actualizar el capital inicial. Intente nuevamente.");
    }
  };

  // Función para formatear un número con 2 decimales y signo
  const formatNumber = (num: number, includeSign: boolean = true) => {
    const formatted = Math.abs(num).toFixed(2);
    if (includeSign) {
      return num >= 0 ? `+${formatted}` : `-${formatted}`;
    }
    return formatted;
  };

  // Type guard to check if an item is a valid MarketOverviewItem
  const isMarketOverviewItem = (item: any): item is MarketOverviewItem => {
    return item !== null;
  };

  // Preparar una vista general de todos los símbolos disponibles con sus datos
  const marketOverview = availableSymbols.map(symbol => {
    const data = marketData[symbol];
    const prediction = predictions[symbol];
    
    // Si no hay datos, mostramos un placeholder con valores por defecto
    if (!data) return null;
    
    // Calcular la dirección de la predicción
    let predictionValue = 0;
    let direction = 'neutral';
    
    if (prediction && prediction.predictions) {
      const predValues = Object.values(prediction.predictions);
      predictionValue = predValues.length > 0 ? predValues[0] : 0;
      direction = predictionValue > 0 ? 'up' : predictionValue < 0 ? 'down' : 'neutral';
    }
    
    return {
      symbol,
      price: typeof data.price === 'number' ? data.price : 100,
      change: typeof data.change === 'number' ? data.change : 0,
      changePercent: typeof data.changePercent === 'number' ? data.changePercent : 0,
      prediction: predictionValue,
      direction,
      volume: typeof data.volume === 'number' ? data.volume : 0,
      high: typeof data.high === 'number' ? data.high : 0,
      low: typeof data.low === 'number' ? data.low : 0
    };
  });
  
  // Filter out nulls using the type guard
  const filteredMarketOverview: MarketOverviewItem[] = marketOverview.filter(isMarketOverviewItem);
  
  // Now sort the guaranteed non-null array
  const sortedByPotential = [...filteredMarketOverview].sort((a, b) => b.prediction - a.prediction);

  if (isLoading) {
    return (
      <Box 
        display="flex" 
        justifyContent="center" 
        alignItems="center" 
        minHeight="100vh"
      >
        <CircularProgress size={60} />
        <Typography variant="h6" ml={2}>
          Cargando dashboard...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box 
        display="flex" 
        flexDirection="column"
        justifyContent="center" 
        alignItems="center" 
        minHeight="100vh"
        p={3}
      >
        <Typography variant="h5" color="error" gutterBottom>
          Error al cargar los datos
        </Typography>
        <Typography>
          {error}
        </Typography>
        <Typography variant="body2" mt={2}>
          Por favor, verifica la conexión con los servicios del backend.
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Barra de navegación */}
      <AppBar position="static">
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={() => setDrawerOpen(true)}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Trading Dashboard
          </Typography>
          <Box display="flex" alignItems="center">
            <Typography variant="body2" color="inherit" sx={{ mr: 2 }}>
              {portfolio && typeof portfolio.total_value === 'number' ? 
                `Portfolio: $${portfolio.total_value.toFixed(2)}` : 
                'Cargando portfolio...'}
            </Typography>
            <AccountBalanceWalletIcon />
          </Box>
        </Toolbar>
      </AppBar>

      {/* Selector de Símbolo */}
      <Box sx={{ p: 2, bgcolor: 'background.paper' }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={4}>
            <FormControl fullWidth variant="outlined" size="small">
              <InputLabel id="symbol-select-label">Símbolo</InputLabel>
              <Select
                labelId="symbol-select-label"
                id="symbol-select"
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value as string)}
                label="Símbolo"
              >
                {availableSymbols.map(symbol => (
                  <MenuItem key={symbol} value={symbol}>
                    {symbol} {marketData[symbol] && 
                      <Box component="span" ml={1} color={marketData[symbol].change >= 0 ? 'success.main' : 'error.main'}>
                        (${typeof marketData[symbol].price === 'number' ? marketData[symbol].price.toFixed(2) : '0.00'})
                      </Box>
                    }
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={8}>
            <Box display="flex" justifyContent="flex-end">
              {portfolio && (
                <>
                  <Box mr={3}>
                    <Typography variant="subtitle2" color="textSecondary">Efectivo</Typography>
                    <Typography variant="h6">
                      ${typeof portfolio?.cash === 'number' ? portfolio.cash.toFixed(2) : '0.00'}
                    </Typography>
                  </Box>
                  <Box mr={3}>
                    <Typography variant="subtitle2" color="textSecondary">Invertido</Typography>
                    <Typography variant="h6">
                      ${typeof portfolio?.invested_value === 'number' ? portfolio.invested_value.toFixed(2) : '0.00'}
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="subtitle2" color="textSecondary">P&L Total</Typography>
                    <Typography variant="h6" color={portfolio?.total_profit && portfolio.total_profit >= 0 ? 'success.main' : 'error.main'}>
                      {portfolio?.total_profit && portfolio.total_profit >= 0 ? '+' : ''}
                      {typeof portfolio?.total_profit === 'number' ? portfolio.total_profit.toFixed(2) : '0.00'} 
                      ({typeof portfolio?.total_profit_pct === 'number' ? portfolio.total_profit_pct.toFixed(2) : '0.00'}%)
                    </Typography>
                  </Box>
                </>
              )}
            </Box>
          </Grid>
        </Grid>
      </Box>

      {/* Contenido principal */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
        {/* Tabs principales */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange}
            aria-label="dashboard tabs"
            variant="scrollable"
            scrollButtons="auto"
            sx={{ 
              '.MuiTabs-flexContainer': { 
                display: 'flex', 
                justifyContent: 'space-between',
                width: '100%'
              }
            }}
          >
            <Tab label="Vista General" />
            <Tab label="Trading" />
            <Tab label="Análisis de Modelos" />
            <Tab label="Cartera" />
            <Tab label="Asistente IA" />
          </Tabs>
        </Box>

        {/* Panel de Vista General */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            {/* Vista general del mercado */}
            <Grid item xs={12} lg={4}>
              <Card elevation={2} sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Vista General del Mercado
                  </Typography>
                  <TableContainer component={Paper} elevation={0} variant="outlined" sx={{ maxHeight: 400, overflow: 'auto' }}>
                    <Table size="small" stickyHeader>
                      <TableHead>
                        <TableRow>
                          <TableCell>Símbolo</TableCell>
                          <TableCell align="right">Precio</TableCell>
                          <TableCell align="right">Cambio</TableCell>
                          <TableCell align="right">Predicción</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {filteredMarketOverview.map((item) => (
                          <TableRow 
                            key={item.symbol}
                            hover
                            onClick={() => setSelectedSymbol(item.symbol)}
                            selected={selectedSymbol === item.symbol}
                            sx={{ cursor: 'pointer' }}
                          >
                            <TableCell component="th" scope="row">
                              <Box display="flex" alignItems="center">
                                <Typography variant="body2" fontWeight={500}>
                                  {item.symbol}
                                </Typography>
                              </Box>  
                            </TableCell>
                            <TableCell align="right">
                              <Typography variant="body2">
                                ${typeof item.price === 'number' ? item.price.toFixed(2) : '0.00'}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Typography variant="body2" color={item.change >= 0 ? 'success.main' : 'error.main'}>
                                {item.change >= 0 ? '+' : ''}{typeof item.change === 'number' ? item.change.toFixed(2) : '0.00'} 
                                ({typeof item.changePercent === 'number' ? item.changePercent.toFixed(2) : '0.00'}%)
                                {item.change > 0 ? <TrendingUpIcon fontSize="small" /> : 
                                 item.change < 0 ? <TrendingDownIcon fontSize="small" /> : null}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Chip
                                label={`${item.prediction >= 0 ? '+' : ''}${typeof item.prediction === 'number' ? item.prediction.toFixed(2) : '0.00'}%`}
                                color={item.prediction > 1 ? 'success' : 
                                      item.prediction < -1 ? 'error' : 'default'}
                                size="small"
                                variant="outlined"
                                icon={item.direction === 'up' ? <TrendingUpIcon /> : 
                                     item.direction === 'down' ? <TrendingDownIcon /> : undefined}
                              />
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>

            {/* Gráfico */}
            <Grid item xs={12} lg={8}>
              <Card elevation={2} sx={{ height: '100%' }}>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Typography variant="h6" gutterBottom>
                      {selectedSymbol} - Gráfico
                    </Typography>
                    {marketData && marketData[selectedSymbol] && (
                      <Box>
                        <Typography variant="h5" component="span" fontWeight="bold">
                          ${typeof marketData[selectedSymbol].price === 'number' ? 
                              marketData[selectedSymbol].price.toFixed(2) : '0.00'}
                        </Typography>
                        <Typography 
                          variant="body1" 
                          component="span" 
                          sx={{ ml: 1 }}
                          color={marketData[selectedSymbol].change >= 0 ? 'success.main' : 'error.main'}
                        >
                          {marketData[selectedSymbol].change >= 0 ? '+' : ''}
                          {typeof marketData[selectedSymbol].change === 'number' ? 
                              marketData[selectedSymbol].change.toFixed(2) : '0.00'} 
                          ({typeof marketData[selectedSymbol].changePercent === 'number' ? 
                              marketData[selectedSymbol].changePercent.toFixed(2) : '0.00'}%)
                        </Typography>
                      </Box>
                    )}
                  </Box>
                  <Box sx={{ height: 400, width: '100%' }}>
                    <StockChart symbol={selectedSymbol} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Resumen de predicciones */}
            <Grid item xs={12} md={6}>
              <PredictionCard symbol={selectedSymbol} />
            </Grid>

            {/* Recomendaciones */}
            <Grid item xs={12} md={6}>
              <TradingRecommendations symbol={selectedSymbol} />
            </Grid>
          </Grid>
        </TabPanel>

        {/* Panel de Trading */}
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={7}>
              <Card elevation={2} sx={{ height: '100%' }}>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Typography variant="h6" gutterBottom>
                      {selectedSymbol} - Gráfico
                    </Typography>
                    {marketData && marketData[selectedSymbol] && (
                      <Box>
                        <Typography variant="h5" component="span" fontWeight="bold">
                          ${typeof marketData[selectedSymbol].price === 'number' ? 
                              marketData[selectedSymbol].price.toFixed(2) : '0.00'}
                        </Typography>
                        <Typography 
                          variant="body1" 
                          component="span" 
                          sx={{ ml: 1 }}
                          color={marketData[selectedSymbol].change >= 0 ? 'success.main' : 'error.main'}
                        >
                          {marketData[selectedSymbol].change >= 0 ? '+' : ''}
                          {typeof marketData[selectedSymbol].change === 'number' ? 
                              marketData[selectedSymbol].change.toFixed(2) : '0.00'} 
                          ({typeof marketData[selectedSymbol].changePercent === 'number' ? 
                              marketData[selectedSymbol].changePercent.toFixed(2) : '0.00'}%)
                        </Typography>
                      </Box>
                    )}
                  </Box>
                  <Box sx={{ height: 400, width: '100%' }}>
                    <StockChart symbol={selectedSymbol} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={5}>
              <TradingPanel symbol={selectedSymbol} />
            </Grid>
          </Grid>
        </TabPanel>

        {/* Panel de Modelos */}
        <TabPanel value={tabValue} index={2}>
          <ModelStatus />
        </TabPanel>

        {/* Panel de Cartera */}
        <TabPanel value={tabValue} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <PortfolioSummary />
            </Grid>
          </Grid>
        </TabPanel>

        {/* Panel de Asistente IA */}
        <TabPanel value={tabValue} index={4}>
          <AIChat />
        </TabPanel>
      </Box>
    </Box>
  );
};

export default Dashboard;