import React, { useState } from 'react';

const TransactionHistory = ({ transactions, selectedSymbol = null, compact = false }) => {
  const [filter, setFilter] = useState(selectedSymbol ? 'symbol' : 'all'); // 'all', 'buys', 'sells', 'symbol'
  
  // Filtrar transacciones
  const filteredTransactions = transactions.filter(transaction => {
    if (filter === 'all') return true;
    if (filter === 'buys') return transaction.action === 'BUY';
    if (filter === 'sells') return transaction.action === 'SELL';
    if (filter === 'symbol') return transaction.symbol === selectedSymbol;
    return true;
  });
  
  // Ordenar por fecha (más reciente primero)
  const sortedTransactions = [...filteredTransactions].sort(
    (a, b) => new Date(b.timestamp) - new Date(a.timestamp)
  );
  
  // Limitar si es compacto
  const displayTransactions = compact ? sortedTransactions.slice(0, 5) : sortedTransactions;
  
  // Formatear fecha
  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString([], { day: '2-digit', month: '2-digit', year: '2-digit' }) + 
           ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold">
          {compact ? 'Últimas Operaciones' : 'Historial de Operaciones'}
        </h2>
        
        {!compact && (
          <div className="flex space-x-1">
            <button
              className={`px-2 py-1 text-xs rounded-md ${filter === 'all' ? 'bg-indigo-600 text-white' : 'bg-gray-200'}`}
              onClick={() => setFilter('all')}
            >
              Todas
            </button>
            <button
              className={`px-2 py-1 text-xs rounded-md ${filter === 'buys' ? 'bg-green-600 text-white' : 'bg-gray-200'}`}
              onClick={() => setFilter('buys')}
            >
              Compras
            </button>
            <button
              className={`px-2 py-1 text-xs rounded-md ${filter === 'sells' ? 'bg-red-600 text-white' : 'bg-gray-200'}`}
              onClick={() => setFilter('sells')}
            >
              Ventas
            </button>
            {selectedSymbol && (
              <button
                className={`px-2 py-1 text-xs rounded-md ${filter === 'symbol' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
                onClick={() => setFilter('symbol')}
              >
                {selectedSymbol}
              </button>
            )}
          </div>
        )}
      </div>
      
      <div className="overflow-x-auto">
        <table className={`min-w-full divide-y divide-gray-200 ${compact ? 'text-xs' : 'text-sm'}`}>
          <thead className="bg-gray-50">
            <tr>
              <th className={`${compact ? 'px-2 py-2' : 'px-6 py-3'} text-left text-xs font-medium text-gray-500 uppercase tracking-wider`}>
                Fecha
              </th>
              <th className={`${compact ? 'px-2 py-2' : 'px-6 py-3'} text-left text-xs font-medium text-gray-500 uppercase tracking-wider`}>
                Símbolo
              </th>
              <th className={`${compact ? 'px-2 py-2' : 'px-6 py-3'} text-left text-xs font-medium text-gray-500 uppercase tracking-wider`}>
                Operación
              </th>
              <th className={`${compact ? 'px-2 py-2' : 'px-6 py-3'} text-left text-xs font-medium text-gray-500 uppercase tracking-wider`}>
                Cantidad
              </th>
              <th className={`${compact ? 'px-2 py-2' : 'px-6 py-3'} text-left text-xs font-medium text-gray-500 uppercase tracking-wider`}>
                Precio
              </th>
              <th className={`${compact ? 'px-2 py-2' : 'px-6 py-3'} text-left text-xs font-medium text-gray-500 uppercase tracking-wider`}>
                Total
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {displayTransactions.length > 0 ? (
              displayTransactions.map((transaction) => (
                <tr key={transaction.id}>
                  <td className={`${compact ? 'px-2 py-2' : 'px-6 py-4'} whitespace-nowrap`}>
                    {!compact 
                      ? formatDate(transaction.timestamp)
                      : new Date(transaction.timestamp).toLocaleDateString([], { day: '2-digit', month: '2-digit' })}
                  </td>
                  <td className={`${compact ? 'px-2 py-2' : 'px-6 py-4'} whitespace-nowrap font-medium`}>
                    {transaction.symbol}
                  </td>
                  <td className={`${compact ? 'px-2 py-2' : 'px-6 py-4'} whitespace-nowrap`}>
                    <span className={`px-2 py-1 rounded-full text-xs
                                      ${transaction.action === 'BUY' 
                                        ? 'bg-green-100 text-green-800' 
                                        : 'bg-red-100 text-red-800'}`}
                    >
                      {transaction.action === 'BUY' ? 'Compra' : 'Venta'}
                    </span>
                  </td>
                  <td className={`${compact ? 'px-2 py-2' : 'px-6 py-4'} whitespace-nowrap`}>
                    {transaction.quantity}
                  </td>
                  <td className={`${compact ? 'px-2 py-2' : 'px-6 py-4'} whitespace-nowrap`}>
                    ${transaction.price.toFixed(2)}
                  </td>
                  <td className={`${compact ? 'px-2 py-2' : 'px-6 py-4'} whitespace-nowrap font-semibold`}>
                    ${transaction.totalValue.toFixed(2)}
                    {transaction.profit && (
                      <span className={transaction.profit >= 0 ? 'text-green-600 ml-1' : 'text-red-600 ml-1'}>
                        ({transaction.profit >= 0 ? '+' : ''}{transaction.profit.toFixed(2)})
                      </span>
                    )}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="6" className={`${compact ? 'px-2 py-4' : 'px-6 py-4'} text-center text-gray-500`}>
                  No hay operaciones para mostrar
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      
      {compact && transactions.length > 5 && (
        <div className="mt-2 text-center">
          <button className="text-indigo-600 text-xs hover:underline">
            Ver todas las operaciones
          </button>
        </div>
      )}
    </div>
  );
};

export default TransactionHistory;
