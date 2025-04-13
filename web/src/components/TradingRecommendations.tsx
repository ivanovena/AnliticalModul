import React, { useState, useEffect } from 'react';
import {
  Box, Typography, Card, CardContent, Grid, Chip,
  Alert, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Paper, Button, CircularProgress,
  Tab, Tabs, Divider, Snackbar
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import TrendingFlatIcon from '@mui/icons-material/TrendingFlat';
import RefreshIcon from '@mui/icons-material/Refresh';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import { useRecommendationsWebSocket } from '../hooks/useWebSocket';
import { useTradingContext } from '../contexts/TradingContext';
import { analysisService } from '../services/api';

interface TradingRecommendation {
  symbol: string;
  action: 'comprar' | 'vender' | 'mantener';
  price: number;
  confidence: number;
  timeframe: string;
  stopLoss: number;
  takeProfit: number;
  rationale: string;
  timestamp: string;
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
      id={`rec-tabpanel-${index}`}
      aria-labelledby={`rec-tab-${index}`}
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

interface TradingRecommendationProps {
  symbol?: string; // Símbolo opcional para filtrar recomendaciones
}

const TradingRecommendations: React.FC<TradingRecommendationProps> = ({ symbol }) => {
  const [recommendations, setRecommendations] = useState<TradingRecommendation[]>([]);
  const [allRecommendations, setAllRecommendations] = useState<Record<string, TradingRecommendation>>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [orderLoading, setOrderLoading] = useState<boolean>(false);
  
  const { 
    selectedSymbol, 
    portfolio, 
    placeOrder, 
    availableSymbols,
    marketData,
    setSelectedSymbol
  } = useTradingContext();

  // Usar el símbolo proporcionado como prop o el seleccionado del contexto
  const targetSymbol = symbol || selectedSymbol;

  // Tab change handler
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Suscribirse a las recomendaciones en tiempo real
  const { isConnected } = useRecommendationsWebSocket(targetSymbol, (data) => {
    if (data && data.recommendation) {
      // Actualizar recomendaciones cuando se recibe una nueva
      const newRecommendation: TradingRecommendation = {
        symbol: data.symbol,
        action: data.recommendation.action,
        price: data.recommendation.price,
        confidence: data.recommendation.confidence,
        timeframe: data.recommendation.timeframe,
        stopLoss: data.recommendation.stopLoss,
        takeProfit: data.recommendation.takeProfit,
        rationale: data.recommendation.rationale || data.summary || '',
        timestamp: new Date().toISOString()
      };
      
      setRecommendations(prev => {
        // Mantener las últimas 10 recomendaciones (la más reciente primero)
        const updated = [newRecommendation, ...prev];
        return updated.slice(0, 10);
      });
      
      // Actualizar también en el mapa de todas las recomendaciones
      setAllRecommendations(prev => ({
        ...prev,
        [data.symbol]: newRecommendation
      }));
    }
  });

  // Ejecutar orden basada en recomendación
  const executeRecommendedOrder = async (symbol: string, action: 'comprar' | 'vender', price: number, quantity: number = 1) => {
    setOrderLoading(true);
    try {
      // Convertir acción de español a inglés para la API
      const apiAction = action === 'comprar' ? 'BUY' : 'SELL';
      
      await placeOrder({
        symbol,
        action: apiAction as any,
        quantity,
        price
      });
      
      setSuccessMessage(`Orden de ${action} para ${symbol} ejecutada correctamente.`);
    } catch (err) {
      console.error('Error executing order:', err);
      setError(`Error al ejecutar la orden: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setOrderLoading(false);
    }
  };

  // Handler para refrescar todas las recomendaciones
  const refreshAllRecommendations = async () => {
    setLoading(true);
    try {
      const allSymbolRecs: Record<string, TradingRecommendation> = {};
      
      // Solicitar recomendaciones para cada símbolo disponible
      for (const symbol of availableSymbols) {
        try {
          const data = await analysisService.getStrategy(symbol);
          
          if (data && data.recommendation) {
            allSymbolRecs[symbol] = {
              symbol: data.symbol,
              action: data.recommendation.action,
              price: data.recommendation.price,
              confidence: data.recommendation.confidence,
              timeframe: data.recommendation.timeframe,
              stopLoss: data.recommendation.stopLoss,
              takeProfit: data.recommendation.takeProfit,
              rationale: data.summary || '',
              timestamp: new Date().toISOString()
            };
          }
        } catch (symbolErr) {
          console.error(`Error fetching recommendation for ${symbol}:`, symbolErr);
          // Continuamos con el siguiente símbolo aunque este falle
        }
      }
      
      setAllRecommendations(allSymbolRecs);
      setError(null);
    } catch (err) {
      console.error('Error refreshing all recommendations:', err);
      setError('Error al actualizar todas las recomendaciones.');
    } finally {
      setLoading(false);
    }
  };

  // Cargar recomendaciones iniciales
  useEffect(() => {
    const fetchRecommendations = async () => {
      setLoading(true);
      try {
        // Obtener recomendaciones iniciales
        const data = await analysisService.getStrategy(targetSymbol);
        
        if (data && data.recommendation) {
          const initialRecommendation: TradingRecommendation = {
            symbol: data.symbol,
            action: data.recommendation.action,
            price: data.recommendation.price,
            confidence: data.recommendation.confidence,
            timeframe: data.recommendation.timeframe,
            stopLoss: data.recommendation.stopLoss,
            takeProfit: data.recommendation.takeProfit,
            rationale: data.summary || '',
            timestamp: new Date().toISOString()
          };
          
          setRecommendations([initialRecommendation]);
          
          // Actualizar también en el mapa de todas las recomendaciones
          setAllRecommendations(prev => ({
            ...prev,
            [data.symbol]: initialRecommendation
          }));
        }
        
        setError(null);
      } catch (err) {
        console.error('Error fetching recommendations:', err);
        setError('No se pudieron cargar las recomendaciones. Por favor, intenta de nuevo más tarde.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchRecommendations();
    
    // Cargar recomendaciones para todos los símbolos solo la primera vez
    if (Object.keys(allRecommendations).length === 0) {
      refreshAllRecommendations();
    }
  }, [targetSymbol]);

  // Verificar si hay posición existente para el símbolo seleccionado
  const hasPosition = portfolio?.positions && targetSymbol in portfolio.positions;
  const position = hasPosition ? portfolio.positions[targetSymbol] : null;
  
  // Determinar la cantidad de acciones a operar
  const getQuantityForRecommendation = (rec: TradingRecommendation, action: 'comprar' | 'vender'): number => {
    if (action === 'vender' && hasPosition) {
      return position!.quantity;
    } else if (action === 'comprar' && portfolio) {
      // Calcular una cantidad razonable basada en el capital disponible
      const availableFunds = portfolio.cash * 0.2; // Usar el 20% del capital disponible
      const suggestedQuantity = Math.floor(availableFunds / rec.price);
      return Math.max(1, suggestedQuantity);
    }
    return 1;
  };

  // Obtener recomendaciones para todos los símbolos ordenadas por confianza
  const getRecommendationsByConfidence = () => {
    return Object.values(allRecommendations)
      .filter(rec => rec.action === 'comprar' || rec.action === 'vender')
      .sort((a, b) => b.confidence - a.confidence);
  };

  // Cerrar mensajes de éxito
  const handleCloseSuccessMessage = () => {
    setSuccessMessage(null);
  };

  if (loading && recommendations.length === 0) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="300px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">
          Recomendaciones de Trading
          {isConnected && (
            <Chip 
              label="En vivo" 
              color="success" 
              size="small" 
              sx={{ ml: 1 }} 
            />
          )}
        </Typography>
        <Button 
          startIcon={<RefreshIcon />}
          onClick={refreshAllRecommendations}
          disabled={loading}
          size="small"
        >
          Actualizar
        </Button>
      </Box>
      
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabValue} onChange={handleTabChange}>
          <Tab label="Símbolo actual" icon={<ShowChartIcon />} iconPosition="start" />
          <Tab label="Todas las recomendaciones" />
        </Tabs>
      </Box>
      
      <TabPanel value={tabValue} index={0}>
        {recommendations.length === 0 ? (
          <Alert severity="info">
            No hay recomendaciones disponibles para {targetSymbol}
          </Alert>
        ) : (
          <Grid container spacing={2}>
            {/* Recomendación más reciente */}
            <Grid item xs={12}>
              <Card variant="outlined">
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                    <Typography variant="subtitle1" fontWeight="bold">
                      {recommendations[0].symbol}
                    </Typography>
                    <Chip 
                      label={recommendations[0].action.toUpperCase()}
                      color={
                        recommendations[0].action === 'comprar' ? 'success' :
                        recommendations[0].action === 'vender' ? 'error' : 'default'
                      }
                      icon={
                        recommendations[0].action === 'comprar' ? <TrendingUpIcon /> :
                        recommendations[0].action === 'vender' ? <TrendingDownIcon /> : <TrendingFlatIcon />
                      }
                    />
                  </Box>
                  
                  <Grid container spacing={1}>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="body2" color="text.secondary">Precio objetivo</Typography>
                      <Typography variant="body1">${recommendations[0].price.toFixed(2)}</Typography>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="body2" color="text.secondary">Confianza</Typography>
                      <Typography variant="body1">{recommendations[0].confidence}%</Typography>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="body2" color="text.secondary">Stop Loss</Typography>
                      <Typography variant="body1">${recommendations[0].stopLoss.toFixed(2)}</Typography>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Typography variant="body2" color="text.secondary">Take Profit</Typography>
                      <Typography variant="body1">${recommendations[0].takeProfit.toFixed(2)}</Typography>
                    </Grid>
                  </Grid>
                  
                  {position && (
                    <Box mt={2} p={1} bgcolor="background.paper" borderRadius={1}>
                      <Typography variant="subtitle2">Posición actual</Typography>
                      <Grid container spacing={1}>
                        <Grid item xs={6} sm={3}>
                          <Typography variant="body2" color="text.secondary">Cantidad</Typography>
                          <Typography variant="body1">{position.quantity}</Typography>
                        </Grid>
                        <Grid item xs={6} sm={3}>
                          <Typography variant="body2" color="text.secondary">Precio promedio</Typography>
                          <Typography variant="body1">${position.avg_cost.toFixed(2)}</Typography>
                        </Grid>
                        <Grid item xs={6} sm={3}>
                          <Typography variant="body2" color="text.secondary">Valor actual</Typography>
                          <Typography variant="body1">${position.market_value.toFixed(2)}</Typography>
                        </Grid>
                        <Grid item xs={6} sm={3}>
                          <Typography variant="body2" color="text.secondary">P/L</Typography>
                          <Typography 
                            variant="body1" 
                            color={position.current_profit && position.current_profit >= 0 ? 'success.main' : 'error.main'}
                          >
                            {position.current_profit && position.current_profit >= 0 ? '+' : ''}
                            ${position.current_profit?.toFixed(2) || '0.00'}
                          </Typography>
                        </Grid>
                      </Grid>
                    </Box>
                  )}
                  
                  <Box mt={2}>
                    <Typography variant="body2" fontStyle="italic">
                      "{recommendations[0].rationale}"
                    </Typography>
                    <Typography variant="caption" color="text.secondary" display="block" mt={1}>
                      Horizonte: {recommendations[0].timeframe} • Actualizado: {new Date(recommendations[0].timestamp).toLocaleTimeString()}
                    </Typography>
                  </Box>
                  
                  <Box mt={2} display="flex" gap={1}>
                    <Button 
                      variant="contained" 
                      color="success" 
                      size="small"
                      disabled={recommendations[0].action !== 'comprar' || orderLoading}
                      onClick={() => executeRecommendedOrder(
                        targetSymbol, 
                        'comprar', 
                        recommendations[0].price,
                        getQuantityForRecommendation(recommendations[0], 'comprar')
                      )}
                    >
                      {orderLoading ? <CircularProgress size={24} /> : 'Comprar'}
                    </Button>
                    <Button 
                      variant="contained" 
                      color="error" 
                      size="small"
                      disabled={recommendations[0].action !== 'vender' || !hasPosition || orderLoading}
                      onClick={() => executeRecommendedOrder(
                        targetSymbol, 
                        'vender', 
                        recommendations[0].price,
                        getQuantityForRecommendation(recommendations[0], 'vender')
                      )}
                    >
                      {orderLoading ? <CircularProgress size={24} /> : 'Vender'}
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            
            {/* Historial de recomendaciones */}
            {recommendations.length > 1 && (
              <Grid item xs={12}>
                <Typography variant="subtitle1" gutterBottom>
                  Historial de recomendaciones
                </Typography>
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Símbolo</TableCell>
                        <TableCell>Acción</TableCell>
                        <TableCell align="right">Precio</TableCell>
                        <TableCell align="right">Confianza</TableCell>
                        <TableCell>Horizonte</TableCell>
                        <TableCell>Fecha</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {recommendations.slice(1).map((rec, index) => (
                        <TableRow key={index}>
                          <TableCell>{rec.symbol}</TableCell>
                          <TableCell>
                            <Chip 
                              label={rec.action.toUpperCase()}
                              size="small"
                              color={
                                rec.action === 'comprar' ? 'success' :
                                rec.action === 'vender' ? 'error' : 'default'
                              }
                            />
                          </TableCell>
                          <TableCell align="right">${rec.price.toFixed(2)}</TableCell>
                          <TableCell align="right">{rec.confidence}%</TableCell>
                          <TableCell>{rec.timeframe}</TableCell>
                          <TableCell>{new Date(rec.timestamp).toLocaleString()}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Grid>
            )}
          </Grid>
        )}
      </TabPanel>
      
      <TabPanel value={tabValue} index={1}>
        <Typography variant="subtitle1" gutterBottom>
          Mejores oportunidades de trading
        </Typography>
        
        {Object.keys(allRecommendations).length === 0 ? (
          <Alert severity="info">
            No hay recomendaciones disponibles para ningún símbolo. Intenta actualizando.
          </Alert>
        ) : (
          <TableContainer component={Paper} variant="outlined">
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Símbolo</TableCell>
                  <TableCell>Acción</TableCell>
                  <TableCell align="right">Precio actual</TableCell>
                  <TableCell align="right">Precio objetivo</TableCell>
                  <TableCell align="right">Potencial</TableCell>
                  <TableCell align="right">Confianza</TableCell>
                  <TableCell>Acciones</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {getRecommendationsByConfidence().map((rec) => {
                  const currentPrice = marketData[rec.symbol]?.price || 0;
                  const potential = (rec.price - currentPrice) / currentPrice * 100;
                  
                  return (
                    <TableRow 
                      key={rec.symbol}
                      hover
                      onClick={() => setSelectedSymbol(rec.symbol)}
                      sx={{ cursor: 'pointer' }}
                    >
                      <TableCell component="th" scope="row">
                        <Typography variant="body2" fontWeight="bold">
                          {rec.symbol}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip 
                          label={rec.action.toUpperCase()}
                          size="small"
                          color={
                            rec.action === 'comprar' ? 'success' :
                            rec.action === 'vender' ? 'error' : 'default'
                          }
                          icon={
                            rec.action === 'comprar' ? <TrendingUpIcon fontSize="small" /> :
                            rec.action === 'vender' ? <TrendingDownIcon fontSize="small" /> : 
                            <TrendingFlatIcon fontSize="small" />
                          }
                        />
                      </TableCell>
                      <TableCell align="right">
                        ${typeof currentPrice === 'number' ? currentPrice.toFixed(2) : '0.00'}
                      </TableCell>
                      <TableCell align="right">${rec.price.toFixed(2)}</TableCell>
                      <TableCell 
                        align="right"
                        sx={{ color: potential > 0 ? 'success.main' : 'error.main' }}
                      >
                        {potential > 0 ? '+' : ''}{potential.toFixed(2)}%
                      </TableCell>
                      <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                        {rec.confidence}%
                      </TableCell>
                      <TableCell>
                        <Button
                          variant="outlined"
                          size="small"
                          color={rec.action === 'comprar' ? 'success' : 'error'}
                          onClick={(e) => {
                            e.stopPropagation();
                            // Solo ejecutar si la acción es comprar o vender
                            if (rec.action === 'comprar' || rec.action === 'vender') {
                              executeRecommendedOrder(
                                rec.symbol,
                                rec.action as 'comprar' | 'vender',
                                rec.price,
                                getQuantityForRecommendation(rec, rec.action as 'comprar' | 'vender')
                              );
                            }
                          }}
                          disabled={orderLoading || rec.action === 'mantener' || (rec.action === 'vender' && !(portfolio?.positions && rec.symbol in portfolio.positions))}
                        >
                          {rec.action === 'comprar' ? 'Comprar' : rec.action === 'vender' ? 'Vender' : 'Mantener'}
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </TabPanel>
      
      {/* Snackbar para mensajes de éxito */}
      <Snackbar
        open={!!successMessage}
        autoHideDuration={6000}
        onClose={handleCloseSuccessMessage}
        message={successMessage}
      />
      
      {/* Mostrar mensaje de error */}
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
    </Box>
  );
};

export default TradingRecommendations;