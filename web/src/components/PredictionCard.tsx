import React, { useState, useEffect } from 'react';
import { 
  Box, Typography, Card, CardContent, 
  Chip, LinearProgress, Grid, Button,
  CircularProgress, Divider, Table, TableBody,
  TableCell, TableContainer, TableHead, TableRow,
  Paper, Tab, Tabs, Tooltip, IconButton
} from '@mui/material';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import RemoveIcon from '@mui/icons-material/Remove';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import RefreshIcon from '@mui/icons-material/Refresh';
import TimelineIcon from '@mui/icons-material/Timeline';
import { Prediction } from '../types/api';
import { analysisService } from '../services/api';
import { usePredictionsWebSocket } from '../hooks/useWebSocket';

// Interfaces auxiliares para los datos que no están definidos en los tipos originales
interface ExtendedPrediction extends Prediction {
  confidence?: number;
  direction?: 'up' | 'down' | 'neutral';
  factors?: string[];
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
      id={`prediction-tabpanel-${index}`}
      aria-labelledby={`prediction-tab-${index}`}
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

interface PredictionCardProps {
  symbol: string;
  prediction?: ExtendedPrediction;
}

const PredictionCard: React.FC<PredictionCardProps> = ({ symbol, prediction: initialPrediction }) => {
  const [prediction, setPrediction] = useState<ExtendedPrediction | undefined>(initialPrediction);
  const [strategy, setStrategy] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [predictionHistory, setPredictionHistory] = useState<ExtendedPrediction[]>([]);
  const [tabValue, setTabValue] = useState(0);
  
  // Suscribirse a las predicciones en tiempo real
  const { lastMessage, isConnected } = usePredictionsWebSocket(symbol, (data) => {
    if (data && data.symbol === symbol) {
      const newPrediction = data as ExtendedPrediction;
      setPrediction(newPrediction);
      
      // Añadir a historial
      setPredictionHistory(prev => {
        // Verificar si ya existe una predicción con el mismo timestamp
        const exists = prev.some(p => p.timestamp === newPrediction.timestamp);
        if (exists) return prev;
        
        // Agregar y ordenar por timestamp (más reciente primero)
        const newHistory = [...prev, newPrediction];
        return newHistory
          .sort((a, b) => {
            const dateA = new Date(a.timestamp).getTime();
            const dateB = new Date(b.timestamp).getTime();
            return dateB - dateA; // Orden descendente (más reciente primero)
          })
          .slice(0, 20); // Mantener solo las 20 predicciones más recientes
      });
    }
  });

  // Cargar estrategia de inversión cuando cambia el símbolo
  useEffect(() => {
    const fetchStrategy = async () => {
      setLoading(true);
      try {
        const data = await analysisService.getStrategy(symbol);
        setStrategy(data);
        setError(null);
      } catch (err) {
        console.error(`Error fetching strategy for ${symbol}:`, err);
        setError(`No se pudo obtener la estrategia para ${symbol}`);
      } finally {
        setLoading(false);
      }
    };

    fetchStrategy();
  }, [symbol]);

  // Actualizar predicción del contexto cuando cambia
  useEffect(() => {
    if (initialPrediction) {
      setPrediction(initialPrediction);
      
      // Añadir a historial si es nueva
      setPredictionHistory(prev => {
        const exists = prev.some(p => p.timestamp === initialPrediction.timestamp);
        if (exists) return prev;
        
        const newHistory = [...prev, initialPrediction];
        return newHistory
          .sort((a, b) => {
            const dateA = new Date(a.timestamp).getTime();
            const dateB = new Date(b.timestamp).getTime();
            return dateB - dateA;
          })
          .slice(0, 20);
      });
    }
  }, [initialPrediction]);

  // Manejar cambio de pestaña
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  // Refrescar predicciones manualmente
  const handleRefresh = async () => {
    setLoading(true);
    try {
      const freshPrediction = await analysisService.getPrediction(symbol);
      setPrediction(freshPrediction as ExtendedPrediction);
      
      // Añadir a historial
      setPredictionHistory(prev => {
        const exists = prev.some(p => p.timestamp === freshPrediction.timestamp);
        if (exists) return prev;
        
        const newHistory = [...prev, freshPrediction as ExtendedPrediction];
        return newHistory
          .sort((a, b) => {
            const dateA = new Date(a.timestamp).getTime();
            const dateB = new Date(b.timestamp).getTime();
            return dateB - dateA;
          })
          .slice(0, 20);
      });
      
      setError(null);
    } catch (err) {
      console.error(`Error refreshing prediction for ${symbol}:`, err);
      setError(`No se pudo actualizar la predicción para ${symbol}`);
    } finally {
      setLoading(false);
    }
  };

  // Calcular el valor de predicción usando las predicciones disponibles
  const calculatePredictionValue = (pred: ExtendedPrediction | undefined): number => {
    if (!pred || !pred.predictions) return 0;
    
    // Asumimos que predictions es un objeto con períodos de tiempo como claves
    const predValues = Object.values(pred.predictions);
    return predValues.length > 0 ? predValues[0] : 0;
  };
  
  // Determinar la dirección basada en el valor de predicción
  const determinePredictionDirection = (pred: ExtendedPrediction | undefined): 'up' | 'down' | 'neutral' => {
    if (!pred) return 'neutral';
    // Si el componente ya tiene una dirección definida, usarla
    if (pred.direction) return pred.direction;
    
    // De lo contrario, calcular basándose en el valor de predicción
    const predValue = calculatePredictionValue(pred);
    if (predValue > 0) return 'up';
    if (predValue < 0) return 'down';
    return 'neutral';
  };

  // Formato de los datos
  const formattedPrediction = calculatePredictionValue(prediction);
  const formattedConfidence = prediction?.confidence || 
                             (prediction?.modelMetrics?.accuracy ? prediction.modelMetrics.accuracy / 100 : 0);
  const direction = determinePredictionDirection(prediction);
  
  // Obtener color basado en la dirección
  const getDirectionColor = (dir: string) => {
    switch (dir) {
      case 'up':
        return 'success.main';
      case 'down':
        return 'error.main';
      default:
        return 'text.secondary';
    }
  };

  // Obtener icono basado en la dirección
  const getDirectionIcon = (dir: string) => {
    switch (dir) {
      case 'up':
        return <ArrowUpwardIcon fontSize="small" color="success" />;
      case 'down':
        return <ArrowDownwardIcon fontSize="small" color="error" />;
      default:
        return <RemoveIcon fontSize="small" color="disabled" />;
    }
  };
  
  // Formatear las predicciones multiperiodo si existen
  const formatMultiPeriodPredictions = () => {
    if (!prediction || !prediction.predictions) return [];
    
    return Object.entries(prediction.predictions)
      .sort((a, b) => {
        // Ordenar por periodo (asumimos que son strings como "1d", "7d", "30d")
        const getNumeric = (str: string) => parseInt(str.replace(/\D/g, '')) || 0;
        return getNumeric(a[0]) - getNumeric(b[0]);
      })
      .map(([period, value]) => ({
        period,
        value,
        direction: value > 0 ? 'up' : value < 0 ? 'down' : 'neutral'
      }));
  };

  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
          <Typography variant="h6">
            Predicción: {symbol}
          </Typography>
          <Box>
            <Tooltip title="Refrescar predicciones">
              <IconButton 
                size="small" 
                onClick={handleRefresh} 
                disabled={loading}
              >
                <RefreshIcon />
              </IconButton>
            </Tooltip>
            <Chip
              size="small"
              label={isConnected ? "En vivo" : "Desconectado"}
              color={isConnected ? "success" : "error"}
              variant="outlined"
              sx={{ ml: 1 }}
            />
          </Box>
        </Box>

        {/* Pestañas para diferentes vistas */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange} 
            variant="scrollable"
            scrollButtons="auto"
            aria-label="prediction tabs"
          >
            <Tab label="Resumen" />
            <Tab label="Detalle" />
            <Tab label="Histórico" />
          </Tabs>
        </Box>
        
        {/* Panel de Resumen */}
        <TabPanel value={tabValue} index={0}>
          <Box sx={{ py: 1 }}>
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <CircularProgress />
              </Box>
            ) : error ? (
              <Box sx={{ py: 2 }}>
                <Typography color="error">{error}</Typography>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={handleRefresh}
                  startIcon={<RefreshIcon />}
                  sx={{ mt: 1 }}
                >
                  Reintentar
                </Button>
              </Box>
            ) : (
              <>
                {/* Predicción Principal */}
                <Box 
                  sx={{ 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center',
                    py: 2,
                    backgroundColor: 'background.paper',
                    borderRadius: 1,
                    boxShadow: 1,
                    mb: 2
                  }}
                >
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    Predicción a 5 días (%)
                  </Typography>
                  <Box display="flex" alignItems="center">
                    {getDirectionIcon(direction)}
                    <Typography 
                      variant="h3" 
                      component="div" 
                      color={getDirectionColor(direction)}
                      fontWeight="bold"
                      sx={{ mx: 1 }}
                    >
                      {formattedPrediction > 0 ? '+' : ''}{formattedPrediction.toFixed(2)}%
                    </Typography>
                  </Box>
                  <Box 
                    sx={{ 
                      mt: 1, 
                      display: 'flex', 
                      alignItems: 'center', 
                      bgcolor: formattedConfidence > 0.7 ? 'success.50' :
                               formattedConfidence > 0.5 ? 'warning.50' : 
                               'error.50',
                      px: 2,
                      py: 0.5,
                      borderRadius: 10
                    }}
                  >
                    <Typography variant="body2" color="text.secondary">
                      Confianza:
                    </Typography>
                    <LinearProgress 
                      variant="determinate" 
                      value={formattedConfidence * 100} 
                      sx={{ 
                        width: 100, 
                        ml: 1, 
                        mr: 1, 
                        height: 8, 
                        borderRadius: 5,
                      }}
                      color={
                        formattedConfidence > 0.7 ? "success" :
                        formattedConfidence > 0.5 ? "warning" : 
                        "error"
                      }
                    />
                    <Typography variant="body2" fontWeight="medium">
                      {(formattedConfidence * 100).toFixed(0)}%
                    </Typography>
                  </Box>
                </Box>

                {/* Indicador de recomendación */}
                {strategy && (
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Recomendación
                    </Typography>
                    <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
                      <Grid container alignItems="center" spacing={2}>
                        <Grid item>
                          <Chip
                            label={
                              strategy.recommendation.action === 'comprar' ? 'COMPRAR' :
                              strategy.recommendation.action === 'vender' ? 'VENDER' : 'MANTENER'
                            }
                            color={
                              strategy.recommendation.action === 'comprar' ? 'success' :
                              strategy.recommendation.action === 'vender' ? 'error' : 'default'
                            }
                            size="medium"
                            sx={{ fontWeight: 'bold' }}
                          />
                        </Grid>
                        <Grid item xs>
                          <Typography variant="body2" color="text.secondary">
                            Confianza: {(strategy.recommendation.confidence * 100).toFixed(0)}%
                          </Typography>
                        </Grid>
                        <Grid item>
                          <Typography variant="body2" fontWeight="medium">
                            Precio objetivo: ${strategy.recommendation.price.toFixed(2)}
                          </Typography>
                        </Grid>
                      </Grid>
                    </Paper>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      {strategy.summary}
                    </Typography>
                  </Box>
                )}
              </>
            )}
          </Box>
        </TabPanel>
        
        {/* Panel de Detalle */}
        <TabPanel value={tabValue} index={1}>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
              <CircularProgress />
            </Box>
          ) : prediction ? (
            <Box>
              {/* Predicciones por periodo */}
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Predicciones por periodo
              </Typography>
              <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Período</TableCell>
                      <TableCell align="right">Predicción (%)</TableCell>
                      <TableCell align="right">Dirección</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {formatMultiPeriodPredictions().map((predItem) => (
                      <TableRow key={predItem.period}>
                        <TableCell component="th" scope="row">
                          {predItem.period}
                        </TableCell>
                        <TableCell 
                          align="right"
                          sx={{ color: predItem.value > 0 ? 'success.main' : 
                                        predItem.value < 0 ? 'error.main' : 'text.secondary' }}
                        >
                          <Typography variant="body2" fontWeight="medium">
                            {predItem.value > 0 ? '+' : ''}{predItem.value.toFixed(2)}%
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          {predItem.value > 0 ? (
                            <ArrowUpwardIcon fontSize="small" color="success" />
                          ) : predItem.value < 0 ? (
                            <ArrowDownwardIcon fontSize="small" color="error" />
                          ) : (
                            <RemoveIcon fontSize="small" color="disabled" />
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              
              {/* Métricas del modelo */}
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Métricas del modelo
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Métrica</TableCell>
                      <TableCell align="right">Valor</TableCell>
                      <TableCell align="right">Estado</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {prediction.modelMetrics && (
                      <>
                        <TableRow>
                          <TableCell component="th" scope="row">
                            <Tooltip title="Error porcentual absoluto medio">
                              <Box display="flex" alignItems="center">
                                MAPE
                                <InfoOutlinedIcon fontSize="small" sx={{ ml: 0.5, fontSize: 16 }} />
                              </Box>
                            </Tooltip>
                          </TableCell>
                          <TableCell align="right">
                            {prediction.modelMetrics.MAPE.toFixed(2)}%
                          </TableCell>
                          <TableCell align="right">
                            <Chip 
                              label={prediction.modelMetrics.MAPE < 5 ? "Bueno" : 
                                    prediction.modelMetrics.MAPE < 10 ? "Medio" : "Alto"} 
                              color={prediction.modelMetrics.MAPE < 5 ? "success" : 
                                     prediction.modelMetrics.MAPE < 10 ? "warning" : "error"}
                              size="small"
                              variant="outlined"
                            />
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell component="th" scope="row">
                            <Tooltip title="Raíz del error cuadrático medio">
                              <Box display="flex" alignItems="center">
                                RMSE
                                <InfoOutlinedIcon fontSize="small" sx={{ ml: 0.5, fontSize: 16 }} />
                              </Box>
                            </Tooltip>
                          </TableCell>
                          <TableCell align="right">
                            {prediction.modelMetrics.RMSE.toFixed(2)}
                          </TableCell>
                          <TableCell align="right">
                            <Chip 
                              label={prediction.modelMetrics.RMSE < 3 ? "Bueno" : 
                                    prediction.modelMetrics.RMSE < 6 ? "Medio" : "Alto"} 
                              color={prediction.modelMetrics.RMSE < 3 ? "success" : 
                                     prediction.modelMetrics.RMSE < 6 ? "warning" : "error"}
                              size="small"
                              variant="outlined"
                            />
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell component="th" scope="row">
                            <Tooltip title="Precisión en dirección de predicción">
                              <Box display="flex" alignItems="center">
                                Precisión
                                <InfoOutlinedIcon fontSize="small" sx={{ ml: 0.5, fontSize: 16 }} />
                              </Box>
                            </Tooltip>
                          </TableCell>
                          <TableCell align="right">
                            {(prediction.modelMetrics.accuracy * 100).toFixed(0)}%
                          </TableCell>
                          <TableCell align="right">
                            <Chip 
                              label={prediction.modelMetrics.accuracy > 0.75 ? "Bueno" : 
                                    prediction.modelMetrics.accuracy > 0.6 ? "Medio" : "Bajo"} 
                              color={prediction.modelMetrics.accuracy > 0.75 ? "success" : 
                                     prediction.modelMetrics.accuracy > 0.6 ? "warning" : "error"}
                              size="small"
                              variant="outlined"
                            />
                          </TableCell>
                        </TableRow>
                      </>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
              
              {/* Última actualización */}
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="caption" color="text.secondary">
                  Última actualización: {new Date(prediction.timestamp).toLocaleString()}
                </Typography>
                <Button 
                  variant="text" 
                  size="small" 
                  startIcon={<TimelineIcon />}
                >
                  Ver análisis completo
                </Button>
              </Box>
            </Box>
          ) : (
            <Typography color="text.secondary">
              No hay datos de predicción disponibles.
            </Typography>
          )}
        </TabPanel>
        
        {/* Panel de Histórico */}
        <TabPanel value={tabValue} index={2}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Historial de predicciones
          </Typography>
          {predictionHistory.length > 0 ? (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Fecha</TableCell>
                    <TableCell align="right">Predicción (%)</TableCell>
                    <TableCell align="right">Confianza</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {predictionHistory.slice(0, 10).map((histPred, index) => {
                    const histValue = calculatePredictionValue(histPred);
                    return (
                      <TableRow key={index}>
                        <TableCell component="th" scope="row">
                          {new Date(histPred.timestamp).toLocaleString()}
                        </TableCell>
                        <TableCell 
                          align="right"
                          sx={{ color: histValue > 0 ? 'success.main' : 
                                      histValue < 0 ? 'error.main' : 'text.secondary' }}
                        >
                          <Typography variant="body2" fontWeight="medium">
                            {histValue > 0 ? '+' : ''}{histValue.toFixed(2)}%
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          {histPred.confidence ? 
                            `${(histPred.confidence * 100).toFixed(0)}%` : 
                            histPred.modelMetrics?.accuracy ? 
                            `${(histPred.modelMetrics.accuracy * 100).toFixed(0)}%` : 
                            'N/A'}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Typography color="text.secondary">
              No hay historial de predicciones disponible.
            </Typography>
          )}
        </TabPanel>
      </CardContent>
    </Card>
  );
};

export default PredictionCard;