import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import axios from 'axios';
import { Chart as ChartJS, LineElement, CategoryScale, LinearScale, PointElement } from 'chart.js';

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement);

function RealTimeChart() {
  const [chartData, setChartData] = useState({
    labels: [],
    datasets: [{
      label: 'AAPL',
      data: [],
      borderColor: 'rgba(75,192,192,1)',
      fill: false,
    }]
  });

  useEffect(() => {
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      const response = await axios.get('http://localhost:8000/fmp_quote/AAPL');
      const price = response.data.price;
      const currentTime = new Date().toLocaleTimeString();
      setChartData(prevData => ({
        labels: [...prevData.labels, currentTime].slice(-20),
        datasets: [{
          ...prevData.datasets[0],
          data: [...prevData.datasets[0].data, price].slice(-20),
        }],
      }));
    } catch (error) {
      console.error("Error fetching real-time data", error);
    }
  };

  return (
    <div className="chart-container">
      <h2>Gráfico en Tiempo Real - AAPL</h2>
      <Line data={chartData} />
    </div>
  );
}

export default RealTimeChart;
