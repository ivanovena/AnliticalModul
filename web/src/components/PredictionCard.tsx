import React, { useState, useEffect } from 'react';
import { 
  Box, Typography, Card, CardContent, 
  Chip, LinearProgress, Grid, Button,
  CircularProgress, Divider
} from '@mui/material';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import RemoveIcon from '@mui/icons-material/Remove';
import { Prediction } from '../types/api';
import { analysisService } from '../services/api';
import { usePredictionsWebSocket } from '../hooks/useWebSocket';

// Interfaces auxiliares para los datos que no están definidos en los tipos originales
interface ExtendedPrediction extends Prediction {
  confidence?: number;
  direction?: 'up' | 'down' | 'neutral';
  factors?: string[];
}

interface PredictionCardProps {
  symbol: string;
  prediction?: ExtendedPrediction;
}

const PredictionCard: React.FC<PredictionCardProps> = ({ symbol, prediction: initialPrediction }) => {
  const [prediction, setPrediction] = useState<ExtendedPrediction | undefined>(initialPrediction);
  const [strategy, setStrategy] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // Suscribirse a las predicciones en tiempo real
  const { lastMessage, isConnected } = usePredictionsWebSocket(symbol, (data) => {
    if (data && data.symbol === symbol) {
      setPrediction(data as ExtendedPrediction);
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
    }
  }, [initialPrediction]);

  // Calcular el valor de predicción usando las predicciones disponibles
  const calculatePredictionValue = (pred: ExtendedPrediction | undefined): number => {
    if (!pred || !pred.predictions) return 0;
    // Asumimos que predictions es un objeto con períodos de tiempo como claves
    // y tomamos el primer valor (generalmente el más inmediato)
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

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Predicciones y Análisis
      </Typography>

      <Grid container spacing={2}>
        {/* Tarjeta de predicción */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Predicción para {symbol}
                {isConnected && (
                  <Chip 
                    size="small" 
                    label="En vivo" 
                    color="success" 
                    sx={{ ml: 1 }} 
                  />
                )}
              </Typography>

              {prediction ? (
                <>
                  <Box display="flex" alignItems="center" mt={2}>
                    <Typography variant="h4" color={getDirectionColor(direction)}>
                      {formattedPrediction > 0 ? '+' : ''}{formattedPrediction.toFixed(2)}%
                    </Typography>
                    {getDirectionIcon(direction)}
                  </Box>
                  
                  <Typography variant="body2" color="text.secondary" mt={1}>
                    Variación prevista para próxima sesión
                  </Typography>
                  
                  <Box mt={2}>
                    <Typography variant="subtitle2" gutterBottom>
                      Confianza: {(formattedConfidence * 100).toFixed(0)}%
                    </Typography>
                    <LinearProgress 
                      variant="determinate" 
                      value={formattedConfidence * 100} 
                      color={
                        formattedConfidence > 0.7 ? "success" :
                        formattedConfidence > 0.4 ? "info" : "warning"
                      }
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                  </Box>
                  
                  {prediction.factors && prediction.factors.length > 0 && (
                    <Box mt={2}>
                      <Typography variant="subtitle2" gutterBottom>
                        Factores relevantes:
                      </Typography>
                      <Box mt={1}>
                        {prediction.factors.map((factor: string, index: number) => (
                          <Chip
                            key={index}
                            label={factor}
                            size="small"
                            variant="outlined"
                            sx={{ mr: 1, mb: 1 }}
                          />
                        ))}
                      </Box>
                    </Box>
                  )}
                  
                  <Typography variant="caption" color="text.secondary" display="block" mt={2}>
                    Última actualización: {prediction.timestamp ? new Date(prediction.timestamp).toLocaleString() : 'N/A'}
                  </Typography>
                </>
              ) : (
                <Box display="flex" justifyContent="center" alignItems="center" height="150px">
                  {loading ? (
                    <CircularProgress size={30} />
                  ) : (
                    <Typography color="text.secondary">
                      No hay predicciones disponibles para {symbol}
                    </Typography>
                  )}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Tarjeta de estrategia */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Estrategia de Inversión
              </Typography>

              {loading ? (
                <Box display="flex" justifyContent="center" alignItems="center" height="150px">
                  <CircularProgress size={30} />
                </Box>
              ) : error ? (
                <Typography color="error" sx={{ py: 2 }}>
                  {error}
                </Typography>
              ) : strategy ? (
                <>
                  <Box mt={2} display="flex" justifyContent="space-between" alignItems="center">
                    <Typography variant="body1">
                      Recomendación:
                    </Typography>
                    <Chip 
                      label={strategy.recommendation.action.toUpperCase()} 
                      color={
                        strategy.recommendation.action === 'comprar' ? 'success' :
                        strategy.recommendation.action === 'vender' ? 'error' : 'default'
                      }
                    />
                  </Box>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Grid container spacing={1}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Precio objetivo:
                      </Typography>
                      <Typography variant="body1">
                        ${strategy.recommendation.price.toFixed(2)}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Confianza:
                      </Typography>
                      <Typography variant="body1">
                        {strategy.recommendation.confidence}%
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Stop Loss:
                      </Typography>
                      <Typography variant="body1">
                        ${strategy.recommendation.stopLoss.toFixed(2)}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Take Profit:
                      </Typography>
                      <Typography variant="body1">
                        ${strategy.recommendation.takeProfit.toFixed(2)}
                      </Typography>
                    </Grid>
                  </Grid>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Typography variant="body2" noWrap>
                    {strategy.summary}
                  </Typography>
                  
                  <Button 
                    variant="outlined" 
                    fullWidth 
                    size="small" 
                    sx={{ mt: 2 }}
                    href="#full-analysis"
                  >
                    Ver análisis completo
                  </Button>
                </>
              ) : (
                <Typography color="text.secondary" sx={{ py: 2 }}>
                  No hay estrategia disponible para {symbol}
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Análisis completo (opcional, depende del diseño) */}
        {strategy && strategy.analysis && (
          <Grid item xs={12} id="full-analysis">
            <Card>
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>
                  Análisis Técnico Completo
                </Typography>
                <Typography variant="body2" sx={{ whiteSpace: 'pre-line' }}>
                  {strategy.analysis}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default PredictionCard;