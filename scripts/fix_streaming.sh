#!/bin/bash

echo "Starting streaming service fix..."

# Stop and remove the container
echo "Stopping streaming service..."
docker-compose stop streaming
docker-compose rm -f streaming

# Rebuild the container
echo "Rebuilding streaming service..."
docker-compose build streaming

# Start the service
echo "Starting streaming service..."
docker-compose up -d streaming

# Wait a moment
echo "Waiting for service to initialize..."
sleep 10

# Check if it's running correctly
echo "Checking status..."
docker-compose ps streaming

# Check the logs
echo "Checking logs (last 10 lines)..."
docker-compose logs --tail=10 streaming

echo "Fix completed. If the service is still restarting, check the logs with:"
echo "docker-compose logs -f streaming"
