import React, { useState } from 'react';

const PredictionTable = ({ symbol, predictions, verificationResults }) => {
  const [selectedTab, setSelectedTab] = useState('current'); // 'current' o 'history'
  
  // Formatear resultados de verificación
  const horizons = predictions ? Object.keys(predictions) : [];
  
  // Función para determinar el color según el error
  const getErrorColor = (errorPct) => {
    if (errorPct < 1) return 'text-green-600';
    if (errorPct < 2) return 'text-green-500';
    if (errorPct < 3) return 'text-blue-500';
    if (errorPct < 5) return 'text-yellow-500';
    return 'text-red-500';
  };
  
  // Formatear timestamp
  const formatTime = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold">Predicciones {symbol}</h2>
        <div className="flex space-x-2">
          <button
            className={`px-3 py-1 text-sm rounded-md 
                      ${selectedTab === 'current' ? 'bg-indigo-600 text-white' : 'bg-gray-200'}`}
            onClick={() => setSelectedTab('current')}
          >
            Predicciones
          </button>
          <button
            className={`px-3 py-1 text-sm rounded-md 
                      ${selectedTab === 'history' ? 'bg-indigo-600 text-white' : 'bg-gray-200'}`}
            onClick={() => setSelectedTab('history')}
          >
            Verificación
          </button>
        </div>
      </div>
      
      {selectedTab === 'current' ? (
        <div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Horizonte
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Precio Predicho
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Cambio %
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Precisión Histórica
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {horizons.length > 0 ? (
                  horizons.map(horizon => {
                    const predictedPrice = predictions[horizon];
                    const avg_error = verificationResults?.[horizon]?.avgError || null;
                    
                    // No tenemos el precio actual, así que usamos un valor aproximado
                    const basePrice = Object.values(predictions)[0] || 0; // Usa el primer valor como referencia
                    const change = ((predictedPrice - basePrice) / basePrice) * 100;
                    
                    return (
                      <tr key={horizon}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {horizon}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          ${predictedPrice?.toFixed(2)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          <span className={change >= 0 ? 'text-green-600' : 'text-red-600'}>
                            {change >= 0 ? '+' : ''}{change.toFixed(2)}%
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          {avg_error !== null ? (
                            <span className={getErrorColor(avg_error)}>
                              Error: {avg_error.toFixed(2)}%
                            </span>
                          ) : (
                            <span className="text-gray-400">Sin datos</span>
                          )}
                        </td>
                      </tr>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan="4" className="px-6 py-4 text-center text-sm text-gray-500">
                      No hay predicciones disponibles
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="mt-2 text-xs text-gray-500 text-right">
            Ciclo de verificación: 1 hora
          </div>
        </div>
      ) : (
        <div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Horizonte
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Predicción
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Real
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Error %
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Hora
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {verificationResults && Object.entries(verificationResults).some(([_, data]) => data?.records?.length > 0) ? (
                  Object.entries(verificationResults)
                    .filter(([_, data]) => data?.records?.length > 0)
                    .flatMap(([horizon, data]) => 
                      data.records.slice(0, 3).map((record, index) => (
                        <tr key={`${horizon}-${index}`}>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                            {horizon}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            ${record.predictedPrice.toFixed(2)}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            ${record.actualPrice.toFixed(2)}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm">
                            <span className={getErrorColor(record.errorPct)}>
                              {record.errorPct.toFixed(2)}%
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {formatTime(record.timestamp)}
                          </td>
                        </tr>
                      ))
                    )
                ) : (
                  <tr>
                    <td colSpan="5" className="px-6 py-4 text-center text-sm text-gray-500">
                      No hay verificaciones disponibles
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="mt-2 text-xs text-gray-500 flex justify-between">
            <div>
              <span className="font-medium">MAPE Total:</span> {verificationResults?.mape ? `${verificationResults.mape.toFixed(2)}%` : 'N/A'}
            </div>
            <div>
              Mostrando las últimas verificaciones por horizonte
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionTable;
