import React, { useState } from 'react';
import { 
  Container, Grid, Paper, Box, Typography, 
  AppBar, Toolbar, IconButton, MenuItem, FormControl, 
  InputLabel, Select, CircularProgress, Tabs, Tab
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import { useTradingContext } from '../contexts/TradingContext';

// Importación de componentes
import StockChart from '../components/StockChart';
import PortfolioSummary from '../components/PortfolioSummary';
import TradingPanel from '../components/TradingPanel';
import PredictionCard from '../components/PredictionCard';
import AIChat from '../components/AIChat';
import ModelStatus from '../components/ModelStatus';
import TradingRecommendations from '../components/TradingRecommendations';

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
  
  const { 
    selectedSymbol, 
    setSelectedSymbol, 
    availableSymbols, 
    marketData, 
    predictions,
    portfolio,
    isLoading,
    error
  } = useTradingContext();

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

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
            Dashboard de Trading con IA
          </Typography>
          <FormControl variant="outlined" size="small" sx={{ minWidth: 120, mr: 1 }}>
            <InputLabel id="symbol-select-label">Símbolo</InputLabel>
            <Select
              labelId="symbol-select-label"
              id="symbol-select"
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              label="Símbolo"
            >
              {availableSymbols.map((symbol) => (
                <MenuItem key={symbol} value={symbol}>
                  {symbol}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Toolbar>
      </AppBar>

      {/* Contenido principal */}
      <Container maxWidth="xl" sx={{ flexGrow: 1, py: 4, overflow: 'auto' }}>
        <Grid container spacing={3}>
          {/* Gráfico principal */}
          <Grid item xs={12} lg={8}>
            <Paper sx={{ p: 2, height: '400px' }}>
              <StockChart 
                symbol={selectedSymbol}
                marketData={marketData[selectedSymbol]}
              />
            </Paper>
          </Grid>

          {/* Panel de trading */}
          <Grid item xs={12} lg={4}>
            <Paper sx={{ p: 2, height: '400px' }}>
              <TradingPanel symbol={selectedSymbol} />
            </Paper>
          </Grid>

          {/* Panel con pestañas para alternar entre diferentes vistas */}
          <Grid item xs={12}>
            <Paper sx={{ width: '100%' }}>
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs 
                  value={tabValue} 
                  onChange={handleTabChange} 
                  aria-label="trading dashboard tabs"
                  variant="scrollable"
                  scrollButtons="auto"
                >
                  <Tab label="Recomendaciones" />
                  <Tab label="Portafolio" />
                  <Tab label="Predicciones" />
                  <Tab label="Estado de Modelos" />
                  <Tab label="Asistente IA" />
                </Tabs>
              </Box>

              <Box sx={{ height: '500px', overflow: 'auto' }}>
                <TabPanel value={tabValue} index={0}>
                  <TradingRecommendations />
                </TabPanel>
                <TabPanel value={tabValue} index={1}>
                  <PortfolioSummary />
                </TabPanel>
                <TabPanel value={tabValue} index={2}>
                  <PredictionCard 
                    symbol={selectedSymbol}
                    prediction={predictions[selectedSymbol]}
                  />
                </TabPanel>
                <TabPanel value={tabValue} index={3}>
                  <ModelStatus />
                </TabPanel>
                <TabPanel value={tabValue} index={4}>
                  <AIChat />
                </TabPanel>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default Dashboard;