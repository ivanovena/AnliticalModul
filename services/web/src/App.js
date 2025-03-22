import React from 'react';
import Navbar from './components/Navbar';
import Portfolio from './components/Portfolio';
import Orders from './components/Orders';
import RealTimeChart from './components/RealTimeChart';
import PredictionsTable from './components/PredictionsTable';
import Strategies from './components/Strategies';
import Chat from './components/Chat';
import './App.css';

function App() {
  return (
    <div className="App">
      <Navbar />
      <div className="dashboard">
        <Portfolio />
        <Orders />
      </div>
      <div className="main-content">
        <RealTimeChart />
        <PredictionsTable />
        <Strategies />
        <Chat />
      </div>
    </div>
  );
}

export default App;
