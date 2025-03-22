import React, { useEffect, useState } from 'react';
import axios from 'axios';

function Portfolio() {
  const [portfolio, setPortfolio] = useState({});

  useEffect(() => {
    fetchPortfolio();
  }, []);

  const fetchPortfolio = async () => {
    try {
      const response = await axios.get('http://localhost:8000/portfolio');
      setPortfolio(response.data);
    } catch (error) {
      console.error("Error fetching portfolio", error);
    }
  };

  return (
    <div className="portfolio">
      <h2>Portafolio</h2>
      <p>Efectivo: ${portfolio.cash}</p>
      <h3>Posiciones:</h3>
      <ul>
        {portfolio.positions && Object.entries(portfolio.positions).map(([symbol, qty]) => (
          <li key={symbol}>{symbol}: {qty}</li>
        ))}
      </ul>
    </div>
  );
}

export default Portfolio;
