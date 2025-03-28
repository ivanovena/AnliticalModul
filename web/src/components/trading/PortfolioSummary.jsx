import React from 'react';

const PortfolioSummary = ({ portfolio, metrics }) => {
  // Calcular valores
  const totalPositionsValue = Object.values(portfolio.positions).reduce(
    (sum, position) => sum + position.quantity * position.currentPrice, 
    0
  );
  
  const cashPercentage = (portfolio.cash / portfolio.totalValue) * 100;
  const positionsPercentage = (totalPositionsValue / portfolio.totalValue) * 100;
  
  return (
    <div className="h-full">
      <h2 className="text-lg font-semibold mb-2">Mi Cartera</h2>
      
      <div className="grid grid-cols-2 gap-2 mb-3">
        <div className="col-span-2">
          <div className="text-sm text-gray-500">Valor Total</div>
          <div className="text-xl font-bold">${portfolio.totalValue.toLocaleString('es-ES', { minimumFractionDigits: 2 })}</div>
        </div>
        
        <div>
          <div className="text-sm text-gray-500">Efectivo</div>
          <div className="text-lg font-semibold">${portfolio.cash.toLocaleString('es-ES', { minimumFractionDigits: 2 })}</div>
        </div>
        
        <div>
          <div className="text-sm text-gray-500">Posiciones</div>
          <div className="text-lg font-semibold">${totalPositionsValue.toLocaleString('es-ES', { minimumFractionDigits: 2 })}</div>
        </div>
      </div>
      
      {/* Rentabilidad */}
      <div className="mb-3">
        <div className="flex justify-between items-center">
          <div className="text-sm text-gray-500">Rentabilidad</div>
          <div className={`font-semibold ${metrics.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {metrics.totalReturn >= 0 ? '+' : ''}{metrics.totalReturn.toFixed(2)}%
          </div>
        </div>
        
        <div className="flex justify-between items-center text-xs">
          <div className="text-gray-500">Hoy:</div>
          <div className={metrics.dailyReturn >= 0 ? 'text-green-600' : 'text-red-600'}>
            {metrics.dailyReturn >= 0 ? '+' : ''}{metrics.dailyReturn.toFixed(2)}%
          </div>
        </div>
        
        <div className="flex justify-between items-center text-xs">
          <div className="text-gray-500">7 días:</div>
          <div className={metrics.weeklyReturn >= 0 ? 'text-green-600' : 'text-red-600'}>
            {metrics.weeklyReturn >= 0 ? '+' : ''}{metrics.weeklyReturn.toFixed(2)}%
          </div>
        </div>
      </div>
      
      {/* Distribución de la cartera */}
      <div className="mb-3">
        <div className="text-sm text-gray-500 mb-1">Distribución</div>
        <div className="w-full bg-gray-200 rounded-full h-2 mb-1">
          <div 
            className="bg-indigo-600 h-2 rounded-full" 
            style={{ width: `${positionsPercentage}%` }}
          ></div>
        </div>
        <div className="flex justify-between text-xs">
          <div>Posiciones: {positionsPercentage.toFixed(1)}%</div>
          <div>Efectivo: {cashPercentage.toFixed(1)}%</div>
        </div>
      </div>
      
      {/* Posiciones */}
      <div>
        <div className="text-sm text-gray-500 mb-1">Posiciones</div>
        <div className="space-y-1 max-h-24 overflow-y-auto">
          {Object.keys(portfolio.positions).length === 0 ? (
            <div className="text-sm text-gray-400 italic">Sin posiciones abiertas</div>
          ) : (
            Object.entries(portfolio.positions).map(([symbol, position]) => {
              const value = position.quantity * position.currentPrice;
              const percentChange = ((position.currentPrice / position.avgCost) - 1) * 100;
              
              return (
                <div key={symbol} className="flex justify-between items-center text-sm">
                  <div className="font-medium">{symbol}</div>
                  <div className="flex flex-col items-end">
                    <div>${value.toLocaleString('es-ES', { minimumFractionDigits: 2 })}</div>
                    <div className={`text-xs ${percentChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {percentChange >= 0 ? '+' : ''}{percentChange.toFixed(2)}%
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};

export default PortfolioSummary;
