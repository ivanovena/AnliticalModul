import React, { useEffect, useState } from 'react';
import axios from 'axios';

function Orders() {
  const [orders, setOrders] = useState([]);

  useEffect(() => {
    fetchOrders();
  }, []);

  const fetchOrders = async () => {
    try {
      const response = await axios.get('http://localhost:8000/orders');
      setOrders(response.data);
    } catch (error) {
      console.error("Error fetching orders", error);
    }
  };

  return (
    <div className="orders">
      <h2>Órdenes</h2>
      <table>
        <thead>
          <tr>
            <th>Símbolo</th>
            <th>Acción</th>
            <th>Cantidad</th>
            <th>Precio</th>
            <th>Fuente</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody>
          {orders.map((order, index) => (
            <tr key={index}>
              <td>{order.symbol}</td>
              <td>{order.action}</td>
              <td>{order.quantity}</td>
              <td>{order.price}</td>
              <td>{order.source}</td>
              <td>{new Date(order.timestamp * 1000).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default Orders;
