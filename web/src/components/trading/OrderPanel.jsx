import React, { useState, useEffect, useCallback } from 'react';
import { usePortfolio } from '../../contexts/PortfolioContext';

const OrderPanel = ({ symbol, currentPrice, position, cash }) => {
  const { placeOrder, loading } = usePortfolio();
  
  const [orderType, setOrderType] = useState('market'); // market o limit
  const [action, setAction] = useState('buy'); // buy o sell
  const [quantity, setQuantity] = useState(1);
  const [limitPrice, setLimitPrice] = useState(currentPrice);
  const [totalCost, setTotalCost] = useState(0);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  
  // Actualizar precio límite cuando cambia el precio actual
  useEffect(() => {
    setLimitPrice(currentPrice);
  }, [currentPrice]);
  
  // Calcular costo total
  useEffect(() => {
    const price = orderType === 'market' ? currentPrice : limitPrice;
    setTotalCost(price * quantity);
  }, [quantity, limitPrice, orderType, currentPrice]);
  
  // Determinar cantidad máxima que se puede comprar/vender
  const maxBuyQty = Math.floor(cash / currentPrice);
  const maxSellQty = position ? position.quantity : 0;
  
  // Manejar cambio de acción (compra/venta)
  const handleActionChange = (newAction) => {
    setAction(newAction);
    // Resetear cantidad a valores razonables
    if (newAction === 'buy') {
      setQuantity(1);
    } else {
      setQuantity(Math.min(1, maxSellQty));
    }
  };
  
  // Manejar cambio de cantidad
  const handleQuantityChange = (e) => {
    const value = parseInt(e.target.value, 10) || 0;
    if (action === 'buy') {
      setQuantity(Math.min(value, maxBuyQty));
    } else {
      setQuantity(Math.min(value, maxSellQty));
    }
  };
  
  // Manejar cambio de precio límite
  const handleLimitPriceChange = (e) => {
    const value = parseFloat(e.target.value) || 0;
    setLimitPrice(Math.max(0.01, value));
  };
  
  // Manejar envío de orden
  const handleSubmitOrder = async () => {
    setError(null);
    setSuccess(null);
    
    if (quantity <= 0) {
      setError('La cantidad debe ser mayor que cero');
      return;
    }
    
    try {
      const orderData = {
        symbol,
        action,
        quantity,
        price: orderType === 'market' ? currentPrice : limitPrice,
        orderType
      };
      
      await placeOrder(orderData);
      setSuccess(`Orden de ${action === 'buy' ? 'compra' : 'venta'} ejecutada con éxito`);
      
      // Resetear formulario
      if (action === 'buy') {
        setQuantity(1);
      } else {
        setQuantity(0);
      }
    } catch (err) {
      setError(err.message || 'Error al ejecutar la orden');
    }
  };
  
  // Función para usar un porcentaje del máximo disponible
  const handlePercentageClick = (percent) => {
    if (action === 'buy') {
      const maxQuantity = Math.floor(cash / (orderType === 'market' ? currentPrice : limitPrice || currentPrice));
      setQuantity(Math.floor(maxQuantity * (percent / 100)));
    } else if (action === 'sell' && position) {
      setQuantity(Math.floor(position.quantity * (percent / 100)));
    }
  };
  
  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">Nueva Orden</h2>
      
      {/* Selección de tipo de operación */}
      <div className="flex mb-4">
        <button
          className={`flex-1 py-2 ${action === 'buy' ? 'bg-green-600 text-white' : 'bg-gray-200 text-gray-700'}`}
          onClick={() => handleActionChange('buy')}
        >
          Comprar
        </button>
        <button
          className={`flex-1 py-2 ${action === 'sell' ? 'bg-red-600 text-white' : 'bg-gray-200 text-gray-700'}`}
          onClick={() => handleActionChange('sell')}
          disabled={maxSellQty === 0}
        >
          Vender
        </button>
      </div>
      
      {/* Información del símbolo y precio */}
      <div className="flex justify-between mb-4">
        <div>
          <div className="text-sm text-gray-500">Símbolo</div>
          <div className="font-medium">{symbol}</div>
        </div>
        <div className="text-right">
          <div className="text-sm text-gray-500">Precio Actual</div>
          <div className="font-medium">${currentPrice.toFixed(2)}</div>
        </div>
      </div>
      
      {/* Tipo de orden */}
      <div className="mb-4">
        <label className="block text-sm text-gray-500 mb-1">Tipo de Orden</label>
        <div className="flex">
          <button
            className={`flex-1 py-1 text-sm ${orderType === 'market' ? 'bg-indigo-100 text-indigo-800 font-medium border-b-2 border-indigo-500' : 'bg-gray-100'}`}
            onClick={() => setOrderType('market')}
          >
            Mercado
          </button>
          <button
            className={`flex-1 py-1 text-sm ${orderType === 'limit' ? 'bg-indigo-100 text-indigo-800 font-medium border-b-2 border-indigo-500' : 'bg-gray-100'}`}
            onClick={() => setOrderType('limit')}
          >
            Límite
          </button>
        </div>
      </div>
      
      {/* Cantidad */}
      <div className="mb-4">
        <div className="flex justify-between items-center mb-1">
          <label className="text-sm text-gray-500">Cantidad</label>
          <div className="text-xs text-gray-500">
            Máx: {action === 'buy' ? maxBuyQty : maxSellQty}
          </div>
        </div>
        <input
          type="number"
          className="w-full p-2 border rounded"
          value={quantity}
          onChange={handleQuantityChange}
          min="1"
          max={action === 'buy' ? maxBuyQty : maxSellQty}
        />
        
        {/* Botones de porcentaje */}
        <div className="flex justify-between mt-1">
          {[25, 50, 75, 100].map((percent) => (
            <button
              key={percent}
              className="text-xs py-1 px-2 bg-gray-100 hover:bg-gray-200 rounded"
              onClick={() => handlePercentageClick(percent)}
            >
              {percent}%
            </button>
          ))}
        </div>
      </div>
      
      {/* Precio límite (si aplica) */}
      {orderType === 'limit' && (
        <div className="mb-4">
          <label className="block text-sm text-gray-500 mb-1">Precio Límite</label>
          <input
            type="number"
            className="w-full p-2 border rounded"
            value={limitPrice}
            onChange={handleLimitPriceChange}
            step="0.01"
            min="0.01"
          />
        </div>
      )}
      
      {/* Total de la operación */}
      <div className="flex justify-between py-2 px-4 bg-gray-100 rounded mb-4">
        <div className="font-medium">Total:</div>
        <div className="font-bold">${totalCost.toFixed(2)}</div>
      </div>
      
      {/* Información adicional */}
      <div className="text-xs text-gray-500 mb-4">
        {action === 'buy' ? (
          <div>Efectivo disponible: ${cash.toFixed(2)}</div>
        ) : (
          <div>Posición actual: {position ? position.quantity : 0} acciones</div>
        )}
      </div>
      
      {/* Mensajes de error/éxito */}
      {error && (
        <div className="p-2 mb-4 bg-red-100 text-red-800 text-sm rounded">
          {error}
        </div>
      )}
      
      {success && (
        <div className="p-2 mb-4 bg-green-100 text-green-800 text-sm rounded">
          {success}
        </div>
      )}
      
      {/* Botón de envío */}
      <button
        className={`w-full py-2 text-white font-medium rounded
                   ${action === 'buy' 
                     ? 'bg-green-600 hover:bg-green-700' 
                     : 'bg-red-600 hover:bg-red-700'}`}
        onClick={handleSubmitOrder}
        disabled={
          loading || 
          quantity <= 0 || 
          (action === 'buy' && totalCost > cash) ||
          (action === 'sell' && (!position || quantity > position.quantity))
        }
      >
        {loading 
          ? 'Procesando...' 
          : `${action === 'buy' ? 'Comprar' : 'Vender'} ${quantity} ${symbol}`}
      </button>
    </div>
  );
};

export default OrderPanel;
