import React, { useState, useEffect } from 'react';
import {
  Box, Typography, Card, CardContent, Grid, Chip,
  Alert, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Paper, Button, CircularProgress
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import TrendingFlatIcon from '@mui/icons-material/TrendingFlat';
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

const TradingRecommendations: React.FC = () => {
  const [recommendations, setRecommendations] = useState<TradingRecommendation[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const { selectedSymbol, portfolio } = useTradingContext();

  // Suscribirse a las recomendaciones en tiempo real
  const { isConnected } = useRecommendationsWebSocket(selectedSymbol, (data) => {
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
    }
  });

  // Cargar recomendaciones iniciales
  useEffect(() => {
    const fetchRecommendations = async () => {
      setLoading(true);
      try {
        // Obtener recomendaciones iniciales
        const data = await analysisService.getStrategy(selectedSymbol);
        
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
  }, [selectedSymbol]);

  // Verificar si hay posición existente para el símbolo seleccionado
  const hasPosition = portfolio?.positions && selectedSymbol in portfolio.positions;
  const position = hasPosition ? portfolio.positions[selectedSymbol] : null;

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="300px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">{error}</Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
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
      
      {recommendations.length === 0 ? (
        <Alert severity="info">
          No hay recomendaciones disponibles para {selectedSymbol}
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
                    disabled={recommendations[0].action !== 'comprar'}
                  >
                    Comprar
                  </Button>
                  <Button 
                    variant="contained" 
                    color="error" 
                    size="small"
                    disabled={recommendations[0].action !== 'vender' || !hasPosition}
                  >
                    Vender
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
    </Box>
  );
};

export default TradingRecommendations;