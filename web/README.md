# Stock Market Model Web Dashboard

## Overview
This is the web interface for the Stock Market Model project, providing a comprehensive dashboard for financial data analysis and real-time stock monitoring.

## Features
- Stock search and selection
- Historical stock data visualization
- Machine learning model predictions
- Real-time stock updates
- Model performance metrics

## Prerequisites
- Node.js (v16+)
- npm or yarn

## Installation
1. Clone the project
2. Navigate to the web directory
3. Run `npm install`
4. Create a `.env` file with:
   ```
   REACT_APP_API_BASE_URL=http://localhost:8000
   REACT_APP_WEBSOCKET_URL=http://localhost:8001
   ```

## Running the Application
- Development: `npm start`
- Production build: `npm run build`

## Architecture
- React for frontend
- Recharts for data visualization
- Axios for API calls
- Socket.IO for real-time updates

## Environment Configuration
Ensure backend services are running and accessible at the URLs specified in the `.env` file.