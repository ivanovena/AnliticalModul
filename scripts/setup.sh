#!/bin/bash

# Color codes for output
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

echo -e "${GREEN}Starting project setup...${NC}"

# Create required directories
echo -e "${YELLOW}Creating required directories...${NC}"
mkdir -p logs/{ingestion,broker,streaming,batch} models monitoring/{prometheus,grafana/provisioning/{datasources,dashboards}} 2>/dev/null

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file from example...${NC}"
    cp .env.example .env
    echo -e "${RED}Please update the .env file with your API keys and other configurations!${NC}"
fi

# Create prometheus config
echo -e "${YELLOW}Setting up Prometheus configuration...${NC}"
mkdir -p monitoring/prometheus
cat > monitoring/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  - job_name: "broker"
    static_configs:
      - targets: ["broker:8000"]

  - job_name: "kafka"
    static_configs:
      - targets: ["kafka-exporter:9308"]

  - job_name: "postgres"
    static_configs:
      - targets: ["postgres-exporter:9187"]
EOF

# Create Grafana datasource
echo -e "${YELLOW}Setting up Grafana configuration...${NC}"
mkdir -p monitoring/grafana/provisioning/datasources
cat > monitoring/grafana/provisioning/datasources/datasource.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

# Run database migrations
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Update the .env file with your API keys and database credentials"
echo -e "2. Run the services with 'docker-compose up -d'"
echo -e "3. Access the web interface at http://localhost"
echo -e "4. Access Grafana at http://localhost:3000 (admin/admin)"
echo -e "5. Access Prometheus at http://localhost:9090"
