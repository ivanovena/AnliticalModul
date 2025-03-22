import React, { useState } from 'react';
import axios from 'axios';

function PredictionsTable() {
  const [predictions, setPredictions] = useState({});
  const [plan, setPlan] = useState({});

  const handleInputChange = (symbol, value) => {
    setPredictions(prev => ({ ...prev, [symbol]: parseFloat(value) }));
  };

  const generatePlan = async () => {
    try {
      const response = await axios.post('http://localhost:8000/plan', { predictions });
      setPlan(response.data.investment_plan);
    } catch (error) {
      console.error("Error generating plan", error);
    }
  };

  return (
    <div className="predictions-table">
      <h2>Predicciones y Estrategias</h2>
      <table>
        <thead>
          <tr>
            <th>Símbolo</th>
            <th>Ganancia Esperada (%)</th>
          </tr>
        </thead>
        <tbody>
          {["AAPL", "GOOGL", "MSFT"].map(symbol => (
            <tr key={symbol}>
              <td>{symbol}</td>
              <td>
                <input
                  type="number"
                  onChange={(e) => handleInputChange(symbol, e.target.value)}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <button onClick={generatePlan}>Generar Plan de Inversión</button>
      {Object.keys(plan).length > 0 && (
        <div>
          <h3>Plan de Inversión</h3>
          <ul>
            {Object.entries(plan).map(([symbol, action]) => (
              <li key={symbol}>{symbol}: {action}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default PredictionsTable;
