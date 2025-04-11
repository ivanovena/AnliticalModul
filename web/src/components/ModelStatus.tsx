import React from 'react';
import {
  Box, Typography, Card, CardContent,
  Grid, Chip, LinearProgress, Divider,
  List, ListItem, ListItemText, ListItemIcon
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningIcon from '@mui/icons-material/Warning';
import ErrorIcon from '@mui/icons-material/Error';
import { useTradingContext } from '../contexts/TradingContext';
import { ModelStatus as ModelStatusType } from '../types/api';

const ModelStatus: React.FC = () => {
  const { modelsStatus } = useTradingContext();

  if (!modelsStatus || modelsStatus.length === 0) {
    return (
      <Box>
        <Typography variant="h6" gutterBottom>
          Estado de los Modelos
        </Typography>
        <Typography color="text.secondary">
          Cargando información de los modelos...
        </Typography>
      </Box>
    );
  }

  // Obtener los modelos específicos del array
  const onlineModel = modelsStatus.find(model => model.modelId === 'online') || modelsStatus[0];
  const batchModel = modelsStatus.find(model => model.modelId === 'batch') || modelsStatus[0];
  const ensembleModel = modelsStatus.find(model => model.modelId === 'ensemble') || modelsStatus[0];

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

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'critical':
        return 'error';
      default:
        return 'default';
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
      <Typography variant="h6" gutterBottom>
        Estado de los Modelos
      </Typography>

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
                  color={getStatusColor(onlineStatus) as any}
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
                  color={getStatusColor(batchStatus) as any}
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
                  color={getStatusColor(ensembleStatus) as any}
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
                      color={getStatusColor(onlineStatus) as any}
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
                      color={getStatusColor(batchStatus) as any}
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
                      color={getStatusColor(ensembleStatus) as any}
                      sx={{ height: 10, borderRadius: 5 }}
                    />
                  </Grid>
                </Grid>
              </Box>
              
              <Typography variant="caption" color="text.secondary" display="block" mt={2}>
                Última actualización general: {formatDate(modelsStatus[0].lastUpdated)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ModelStatus;