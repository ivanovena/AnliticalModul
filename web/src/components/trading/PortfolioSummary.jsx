import React from 'react';

const PortfolioSummary = ({ portfolio, metrics }) => {
  // Calcular datos de forma segura
  const positions = portfolio?.positions ?? {};
  const totalValue = portfolio?.totalValue ?? 0;
  const initialCash = portfolio?.initialCash ?? 0;
  const cash = portfolio?.cash ?? 0;
  
  const investedValue = Object.values(positions).reduce(
    (total, pos) => total + ((pos?.quantity ?? 0) * (pos?.avgCost ?? 0)), 0
  );
  
  const currentPositionsValue = Object.values(positions).reduce(
    (total, pos) => total + ((pos?.quantity ?? 0) * (pos?.currentPrice ?? 0)), 0
  );
  
  // Calcular ganancias/pérdidas realizadas e irrealizadas
  const unrealizedPL = currentPositionsValue - investedValue;
  const unrealizedPLPercent = investedValue > 0 ? (unrealizedPL / investedValue) * 100 : 0;
  
  // Calcular ganancias/pérdidas totales (incluye las realizadas)
  const totalPL = totalValue - initialCash;
  const totalPLPercent = initialCash > 0 ? (totalPL / initialCash) * 100 : 0;
  
  // Métricas seguras
  const dailyReturn = metrics?.dailyReturn ?? 0;
  const weeklyReturn = metrics?.weeklyReturn ?? 0;
  
  return (
    <div className="rounded-lg border p-4 h-full flex flex-col">
      <h3 className="font-semibold mb-4">Resumen de Cartera</h3>
      
      {/* Valor Total */}
      <div className="mb-4">
        <div className="text-sm text-gray-500">Valor Total</div>
        <div className="text-xl font-bold">${totalValue.toLocaleString('es-ES', { minimumFractionDigits: 2 })}</div>
        <div className={`text-xs ${totalPL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
          {totalPL >= 0 ? '+' : ''}{totalPL.toFixed(2)} ({totalPLPercent.toFixed(2)}%)
        </div>
      </div>
      
      {/* Efectivo vs Invertido */}
      <div className="mb-4">
        <div className="flex justify-between items-center mb-1">
          <div className="text-sm text-gray-500">Efectivo Disponible</div>
          <div className="text-sm font-medium">${cash.toLocaleString('es-ES', { minimumFractionDigits: 2 })}</div>
        </div>
        <div className="flex justify-between items-center">
          <div className="text-sm text-gray-500">Invertido</div>
          <div className="text-sm font-medium">${investedValue.toLocaleString('es-ES', { minimumFractionDigits: 2 })}</div>
        </div>
        
        {/* Barra de progreso para visualizar la distribución */}
        <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
          <div 
            className="h-full bg-blue-500" 
            style={{ width: `${totalValue > 0 ? (investedValue / totalValue) * 100 : 0}%` }}
          ></div>
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>Efectivo: {totalValue > 0 ? ((cash / totalValue) * 100).toFixed(1) : 0}%</span>
          <span>Invertido: {totalValue > 0 ? ((investedValue / totalValue) * 100).toFixed(1) : 0}%</span>
        </div>
      </div>
      
      {/* Ganancias/Pérdidas */}
      <div className="mb-4">
        <div className="text-sm text-gray-500 mb-1">Rendimiento No Realizado</div>
        <div className="flex justify-between items-center">
          <div className={`text-sm ${unrealizedPL >= 0 ? 'text-green-600' : 'text-red-600'} font-medium`}>
            {unrealizedPL >= 0 ? '+' : ''}{unrealizedPL.toFixed(2)}
          </div>
          <div className={`text-sm ${unrealizedPL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            ({unrealizedPLPercent.toFixed(2)}%)
          </div>
        </div>
      </div>
      
      {/* Posiciones activas */}
      <div className="mb-4">
        <div className="text-sm text-gray-500 mb-1">Posiciones Activas</div>
        <div className="flex justify-between items-center">
          <div className="text-sm font-medium">{Object.keys(positions).length}</div>
        </div>
      </div>
      
      {/* Métricas adicionales */}
      <div className="text-xs text-gray-500 mt-auto">
        <div className="flex justify-between mb-1">
          <span>Rendimiento diario:</span>
          <span className={dailyReturn >= 0 ? 'text-green-600' : 'text-red-600'}>
            {dailyReturn >= 0 ? '+' : ''}{dailyReturn.toFixed(2)}%
          </span>
        </div>
        <div className="flex justify-between">
          <span>Rendimiento semanal:</span>
          <span className={weeklyReturn >= 0 ? 'text-green-600' : 'text-red-600'}>
            {weeklyReturn >= 0 ? '+' : ''}{weeklyReturn.toFixed(2)}%
          </span>
        </div>
      </div>
    </div>
  );
};

export default PortfolioSummary;
