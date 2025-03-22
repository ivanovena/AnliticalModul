import React, { useEffect, useState } from 'react';
import axios from 'axios';

function Strategies() {
  const [params, setParams] = useState({});

  useEffect(() => {
    fetchLearnedParams();
  }, []);

  const fetchLearnedParams = async () => {
    try {
      const response = await axios.get('http://localhost:8000/learned_params');
      setParams(response.data);
    } catch (error) {
      console.error("Error fetching learned parameters", error);
    }
  };

  return (
    <div className="strategies">
      <h2>Estrategias y Parámetros Aprendidos</h2>
      <table>
        <tbody>
          {Object.entries(params).map(([key, value]) => (
            <tr key={key}>
              <td>{key}</td>
              <td>{value.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default Strategies;
