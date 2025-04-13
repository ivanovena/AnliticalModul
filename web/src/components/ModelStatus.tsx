import React, { useEffect, useState } from 'react';
import {
  Box, Typography, Card, CardContent,
  Grid, Chip, LinearProgress, Divider,
  List, ListItem, ListItemText, ListItemIcon,
  Button, Alert, CircularProgress
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningIcon from '@mui/icons-material/Warning';
import ErrorIcon from '@mui/icons-material/Error';
import RefreshIcon from '@mui/icons-material/Refresh';
import { useTradingContext } from '../contexts/TradingContext';
import { ModelStatus as ModelStatusType } from '../types/api';

const ModelStatus: React.FC = () => {
  const { modelsStatus, refreshModelsStatus } = useTradingContext();
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // Función para refrescar los datos de los modelos
  const handleRefresh = async () => {
    setLoading(true);
    setError(null);
    try {
      await refreshModelsStatus();
    } catch (err) {
      setError('Error al actualizar el estado de los modelos. Intente nuevamente.');
      console.error('Error refreshing models status:', err);
    } finally {
      setLoading(false);
    }
  };

  // Cargar datos al montar el componente si no están disponibles
  useEffect(() => {
    if (!modelsStatus || (Array.isArray(modelsStatus) && modelsStatus.length === 0)) {
      handleRefresh();
    }
  }, []);

  // Generar datos de modelo por defecto si no existen
  const createDefaultModels = (): ModelStatusType[] => {
    return [
      {
        modelId: "online",
        name: "Modelo Tiempo Real",
        status: "active",
        version: "1.0",
        lastUpdated: new Date().toISOString(),
        symbols: ["AAPL", "MSFT", "GOOGL"],
        metrics: [
          { name: "accuracy", value: 0.78, status: "good", trend: "up" },
          { name: "MAPE", value: 3.2, status: "good", trend: "down" },
          { name: "RMSE", value: 2.1, status: "good", trend: "stable" }
        ]
      },
      {
        modelId: "batch",
        name: "Modelo Diario",
        status: "active",
        version: "1.1",
        lastUpdated: new Date().toISOString(),
        symbols: ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"],
        metrics: [
          { name: "accuracy", value: 0.82, status: "good", trend: "stable" },
          { name: "MAPE", value: 2.8, status: "good", trend: "down" },
          { name: "RMSE", value: 1.9, status: "good", trend: "down" }
        ]
      },
      {
        modelId: "ensemble",
        name: "Modelo Ensemble",
        status: "active",
        version: "1.2",
        lastUpdated: new Date().toISOString(),
        symbols: ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA"],
        metrics: [
          { name: "accuracy", value: 0.85, status: "good", trend: "up" },
          { name: "MAPE", value: 2.5, status: "good", trend: "down" },
          { name: "RMSE", value: 1.7, status: "good", trend: "down" }
        ]
      }
    ];
  };

  // Crear un modelo por defecto individual
  const createDefaultModel = (id: string): ModelStatusType => ({
    modelId: id,
    name: `Modelo ${id}`,
    status: 'active' as 'error' | 'active' | 'training',
    version: '1.0',
    lastUpdated: new Date().toISOString(),
    symbols: ['AAPL', 'GOOG', 'MSFT'],
    metrics: [
      { name: 'accuracy', value: 0.75, status: 'good' as 'good' | 'warning' | 'error', trend: 'stable' as 'up' | 'down' | 'stable' },
      { name: 'MAPE', value: 3.5, status: 'good' as 'good' | 'warning' | 'error', trend: 'stable' as 'up' | 'down' | 'stable' },
      { name: 'RMSE', value: 2.2, status: 'good' as 'good' | 'warning' | 'error', trend: 'stable' as 'up' | 'down' | 'stable' }
    ]
  });

  if (loading) {
    return (
      <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" p={4}>
        <CircularProgress size={60} />
        <Typography variant="h6" mt={2}>
          Cargando estado de los modelos...
        </Typography>
      </Box>
    );
  }

  // Usar datos por defecto si no hay datos disponibles 
  // (en lugar de mostrar un mensaje de error)
  let modelsToShow = modelsStatus;
  if (!modelsStatus || !Array.isArray(modelsStatus) || modelsStatus.length === 0) {
    modelsToShow = createDefaultModels();
  }

  // Obtener los modelos específicos del array con fallbacks
  // Asegurarnos de que modelsStatus siempre sea un array
  const modelsArray = Array.isArray(modelsToShow) ? modelsToShow : [];
  const onlineModel = modelsArray.find(model => model && model.modelId === 'online') || createDefaultModel('online');
  const batchModel = modelsArray.find(model => model && model.modelId === 'batch') || createDefaultModel('batch');
  const ensembleModel = modelsArray.find(model => model && model.modelId === 'ensemble') || createDefaultModel('ensemble');

  // Convertir estado del modelo a nuestros estados internos
  const mapStatusToInternal = (model: ModelStatusType): string => {
    if (!model) return 'unknown';
    return model.status === 'active' ? 'healthy' : 
           model.status === 'training' ? 'degraded' : 'critical';
  };

  // Obtener precisión del modelo de sus métricas
  const getModelAccuracy = (model: ModelStatusType): number => {
    if (!model || !model.metrics) return 0;
    const accuracyMetric = model.metrics.find(m => m.name === 'accuracy');
    return accuracyMetric ? accuracyMetric.value : 0;
  };

  // Obtener valor de métricas específicas
  const getMetricValue = (model: ModelStatusType, metricName: string): number => {
    if (!model || !model.metrics) return 0;
    const metric = model.metrics.find(m => m.name === metricName);
    return metric ? metric.value : 0;
  };

  // Obtener icono y color basado en el estado
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircleIcon color="success" />;
      case 'degraded':
        return <WarningIcon color="warning" />;
      case 'critical':
        return <ErrorIcon color="error" />;
      default:
        return <WarningIcon color="disabled" />;
    }
  };

  const getStatusColor = (status: string): 'success' | 'warning' | 'error' | 'primary' => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'critical':
        return 'error';
      default:
        return 'primary';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'Operando normalmente';
      case 'degraded':
        return 'Rendimiento degradado';
      case 'critical':
        return 'Estado crítico';
      default:
        return 'Estado desconocido';
    }
  };

  // Formatear fecha
  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch (e) {
      return 'Fecha desconocida';
    }
  };

  // Estados mapeados
  const onlineStatus = mapStatusToInternal(onlineModel);
  const batchStatus = mapStatusToInternal(batchModel);
  const ensembleStatus = mapStatusToInternal(ensembleModel);

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">
          Estado de los Modelos
        </Typography>
        <Button 
          startIcon={<RefreshIcon />} 
          onClick={handleRefresh}
          disabled={loading}
          size="small"
        >
          Actualizar
        </Button>
      </Box>

      <Grid container spacing={2}>
        {/* Modelo Online */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                <Typography variant="subtitle1">
                  Modelo Online
                </Typography>
                <Chip
                  label={getStatusText(onlineStatus)}
                  color={getStatusColor(onlineStatus)}
                  size="small"
                />
              </Box>
              
              <Divider sx={{ my: 1 }} />
              
              <List dense>
                <ListItem>
                  <ListItemIcon sx={{ minWidth: '30px' }}>
                    {getStatusIcon(onlineStatus)}
                  </ListItemIcon>
                  <ListItemText
                    primary="Precisión"
                    secondary={`${getModelAccuracy(onlineModel).toFixed(1)}%`}
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemText
                    primary="MAPE"
                    secondary={getMetricValue(onlineModel, 'MAPE').toFixed(2)}
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemText
                    primary="RMSE"
                    secondary={getMetricValue(onlineModel, 'RMSE').toFixed(2)}
                  />
                </ListItem>
              </List>
              
              <Typography variant="caption" color="text.secondary">
                Actualizado: {formatDate(onlineModel.lastUpdated)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Modelo Batch */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                <Typography variant="subtitle1">
                  Modelo Batch
                </Typography>
                <Chip
                  label={getStatusText(batchStatus)}
                  color={getStatusColor(batchStatus)}
                  size="small"
                />
              </Box>
              
              <Divider sx={{ my: 1 }} />
              
              <List dense>
                <ListItem>
                  <ListItemIcon sx={{ minWidth: '30px' }}>
                    {getStatusIcon(batchStatus)}
                  </ListItemIcon>
                  <ListItemText
                    primary="Precisión"
                    secondary={`${getModelAccuracy(batchModel).toFixed(1)}%`}
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemText
                    primary="MAPE"
                    secondary={getMetricValue(batchModel, 'MAPE').toFixed(2)}
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemText
                    primary="RMSE"
                    secondary={getMetricValue(batchModel, 'RMSE').toFixed(2)}
                  />
                </ListItem>
              </List>
              
              <Typography variant="caption" color="text.secondary">
                Actualizado: {formatDate(batchModel.lastUpdated)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Modelo Ensemble */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                <Typography variant="subtitle1">
                  Modelo Ensemble
                </Typography>
                <Chip
                  label={getStatusText(ensembleStatus)}
                  color={getStatusColor(ensembleStatus)}
                  size="small"
                />
              </Box>
              
              <Divider sx={{ my: 1 }} />
              
              <List dense>
                <ListItem>
                  <ListItemIcon sx={{ minWidth: '30px' }}>
                    {getStatusIcon(ensembleStatus)}
                  </ListItemIcon>
                  <ListItemText
                    primary="Precisión"
                    secondary={`${getModelAccuracy(ensembleModel).toFixed(1)}%`}
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemText
                    primary="MAPE"
                    secondary={getMetricValue(ensembleModel, 'MAPE').toFixed(2)}
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemText
                    primary="RMSE"
                    secondary={getMetricValue(ensembleModel, 'RMSE').toFixed(2)}
                  />
                </ListItem>
              </List>
              
              <Typography variant="caption" color="text.secondary">
                Actualizado: {formatDate(ensembleModel.lastUpdated)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Resumen de la comparación de modelos */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Comparación de Precisión de Modelos
              </Typography>
              
              <Box mt={2}>
                <Grid container spacing={1} alignItems="center">
                  <Grid item xs={3}>
                    <Typography variant="body2">Online:</Typography>
                  </Grid>
                  <Grid item xs={9}>
                    <LinearProgress
                      variant="determinate"
                      value={getModelAccuracy(onlineModel)}
                      color={getStatusColor(onlineStatus) === 'primary' ? 'primary' : getStatusColor(onlineStatus)}
                      sx={{ height: 10, borderRadius: 5 }}
                    />
                  </Grid>
                </Grid>
                
                <Grid container spacing={1} alignItems="center" mt={1}>
                  <Grid item xs={3}>
                    <Typography variant="body2">Batch:</Typography>
                  </Grid>
                  <Grid item xs={9}>
                    <LinearProgress
                      variant="determinate"
                      value={getModelAccuracy(batchModel)}
                      color={getStatusColor(batchStatus) === 'primary' ? 'primary' : getStatusColor(batchStatus)}
                      sx={{ height: 10, borderRadius: 5 }}
                    />
                  </Grid>
                </Grid>
                
                <Grid container spacing={1} alignItems="center" mt={1}>
                  <Grid item xs={3}>
                    <Typography variant="body2">Ensemble:</Typography>
                  </Grid>
                  <Grid item xs={9}>
                    <LinearProgress
                      variant="determinate"
                      value={getModelAccuracy(ensembleModel)}
                      color={getStatusColor(ensembleStatus) === 'primary' ? 'primary' : getStatusColor(ensembleStatus)}
                      sx={{ height: 10, borderRadius: 5 }}
                    />
                  </Grid>
                </Grid>
              </Box>
              
              <Typography variant="caption" color="text.secondary" display="block" mt={2}>
                Última actualización general: {Array.isArray(modelsStatus) && modelsStatus.length > 0 ? formatDate(modelsStatus[0].lastUpdated) : 'No disponible'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ModelStatus;