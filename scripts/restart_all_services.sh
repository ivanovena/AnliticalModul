#!/bin/bash

echo "Restarting all services in the correct order..."

# Stop all running containers
echo "Stopping all running containers..."
docker-compose down

# Build the containers
echo "Building containers..."
docker-compose build --no-cache postgres ingestion streaming broker frontend

# Start the infrastructure services first
echo "Starting infrastructure services (Postgres, Zookeeper, Kafka)..."
docker-compose up -d postgres zookeeper kafka

# Wait for them to be ready
echo "Waiting for infrastructure services to be ready..."
sleep 30

# Start the backend services
echo "Starting backend services (Ingestion, Streaming, Broker)..."
docker-compose up -d ingestion streaming broker

# Wait for them to be ready
echo "Waiting for backend services to be ready..."
sleep 15

# Start the frontend
echo "Starting frontend service..."
docker-compose up -d frontend

# Check the status
echo "Checking service status..."
docker-compose ps

echo "Done! Services have been restarted."
echo "Monitor the logs with: docker-compose logs -f"
