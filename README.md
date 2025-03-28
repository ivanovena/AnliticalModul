# Stock Market Prediction and Trading Platform

A comprehensive platform for real-time market data ingestion, algorithmic trading, and machine learning-based market predictions.

## Architecture Overview

The platform is built with a microservices architecture:

![Architecture Diagram](https://via.placeholder.com/800x500?text=Architecture+Diagram)

### Services

1. **Ingestion Service**: Collects market data from Financial Modeling Prep API
2. **Streaming Service**: Processes data in real-time using online learning algorithms
3. **Batch Service**: Trains sophisticated ML models for market prediction
4. **Broker Service**: Manages portfolio, executes trades, and provides a trading API
5. **Web Service**: Web interface for monitoring and interacting with the system

### Technologies

- **Backend**: Python, FastAPI, SQLAlchemy, Kafka, River ML
- **Machine Learning**: Scikit-learn, PyTorch, Transformers, FAISS
- **Frontend**: React, Chart.js, Recharts
- **Infrastructure**: Docker, Kubernetes, Prometheus, Grafana
- **Data Storage**: PostgreSQL, Kafka

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Financial Modeling Prep API key
- (Optional) Telegram Bot Token for notifications

### Required API Keys

#### Financial Modeling Prep (FMP) API
The platform requires a Financial Modeling Prep API key to fetch market data. You can:
1. Register at [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs/)
2. Choose a plan based on your needs (even the free tier works for testing)
3. Generate an API key from your dashboard
4. Add the key to your `.env` file as `FMP_API_KEY=your-key-here`

#### Telegram Bot (Optional)
For trade notifications and interaction via Telegram:
1. Create a new bot through [BotFather](https://t.me/botfather)
2. Get the bot token
3. Add it to your `.env` file as `TELEGRAM_BOT_TOKEN=your-token-here`

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/market-model.git
   cd market-model
   ```

2. Run the setup script:
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. Update your API credentials in the `.env` file:
   ```bash
   nano .env
   ```

4. Verify your environment:
   ```bash
   python scripts/verify_environment.py
   ```

5. Clean up unnecessary files (if needed):
   ```bash
   chmod +x scripts/cleanup.sh
   ./scripts/cleanup.sh
   ```

6. Start the services:
   ```bash
   docker-compose up -d
   ```

7. Access the web interface at http://localhost

### Configuration

The system can be configured through environment variables in the `.env` file or through Kubernetes ConfigMaps when running in a cluster.

Key configuration options:
- `FMP_API_KEY`: Financial Modeling Prep API key
- `TELEGRAM_BOT_TOKEN`: Telegram Bot Token for notifications
- `INITIAL_CASH`: Starting cash balance for the trading portfolio
- `USE_DATALAKE`: Toggle between using FMP API directly or cached data

## Services

### Ingestion Service

The Ingestion Service pulls market data from the Financial Modeling Prep API and stores it in the PostgreSQL database. It also publishes events to Kafka for real-time processing.

#### Features:
- Configurable data sources and symbols
- Exponential backoff for API requests
- Automatic data validation and cleaning
- Support for multiple timeframes

### Streaming Service

The Streaming Service processes market data in real-time using online learning algorithms from the River ML library.

#### Features:
- Online learning for real-time adaptation
- Feature extraction and engineering
- Real-time prediction publishing via Kafka
- Performance metrics tracking

### Batch Service

The Batch Service performs periodic training of sophisticated machine learning models using historical data.

#### Features:
- Ensemble modeling with cross-validation
- Hyperparameter optimization
- Feature importance analysis
- Model evaluation and comparison
- Automated model deployment

### Broker Service

The Broker Service manages the trading portfolio, executes trades, and provides a REST API for client applications.

#### Features:
- Portfolio management
- Order execution
- Risk management
- AI-powered trading advisor
- WebSocket API for real-time updates
- REST API for client applications
- Telegram bot integration

### Web Service

The Web Service provides a user interface for monitoring and interacting with the system.

#### Features:
- Real-time market data visualization
- Portfolio monitoring
- Trade execution interface
- Model performance dashboards
- AI advisor chat interface

## API Documentation

### Broker API

The Broker Service exposes a REST API at port 8000.

#### Endpoints:

- `GET /portfolio`: Get current portfolio state
- `GET /orders`: Get order history
- `POST /order`: Place a new order
- `GET /metrics`: Get portfolio metrics and risk analysis
- `POST /chat`: Chat with the AI trading advisor
- `POST /feedback`: Submit feedback on AI recommendations
- `GET /model/performance`: Get ML model performance metrics
- `POST /plan`: Generate investment plan based on predictions
- `GET /health`: Health check endpoint

### WebSocket API

Real-time updates are available through WebSocket on port 8001.

#### Topics:

- `real-time-stock-update`: Real-time market data updates
- `predictions-{symbol}`: Real-time predictions for a specific symbol
- `portfolio-update`: Portfolio updates
- `order-update`: Order status updates

## Monitoring

The platform includes comprehensive monitoring with Prometheus and Grafana.

### Metrics:

- Service health and performance
- Database performance
- Kafka performance
- Model performance (accuracy, latency)
- Trading performance (returns, risk metrics)

### Dashboards:

- System Overview
- Market Data
- Model Performance
- Trading Performance
- Risk Analysis

## Development

### Local Development

1. Clone the repository
2. Run the setup script
3. Start the development environment:
   ```bash
   docker-compose up -d
   ```

### Running Tests

```bash
# Run unit tests
pytest

# Run with coverage
pytest --cov=services tests/
```

### CI/CD Pipeline

The project includes a GitHub Actions CI/CD pipeline that:
1. Runs linting and tests
2. Builds Docker images
3. Pushes images to Docker Hub
4. Deploys to Kubernetes clusters

## Deployment

### Kubernetes Deployment

The `k8s` directory contains Kubernetes manifests for deploying the platform to a Kubernetes cluster.

#### Deploying to a Kubernetes cluster:

```bash
kubectl apply -f k8s/
```

## Performance Tuning

### Database Optimization

The PostgreSQL database is optimized with:
- Appropriate indexes on frequently queried columns
- Connection pooling
- Query optimization
- Partitioning for historical data

### Kafka Optimization

The Kafka cluster is configured for optimal performance with:
- Appropriate partition count for topics
- Replication factor for fault tolerance
- Producer and consumer tuning

## Troubleshooting

### Common Issues

#### Missing API Keys
If you get environment variable errors:
1. Check your `.env` file has all required keys
2. Run `python scripts/verify_environment.py` to confirm
3. Restart services with `docker-compose restart`

#### Connection Issues
If services can't connect:
1. Check if Kafka and PostgreSQL are running
2. Verify network configuration in Docker Compose
3. Check service logs with `docker-compose logs <service_name>`

#### Data Not Updating
If market data isn't updating:
1. Verify your FMP API key is valid 
2. Check ingestion service logs for API rate limits
3. Confirm PostgreSQL has the market_data table

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
