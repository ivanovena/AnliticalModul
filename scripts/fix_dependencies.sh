#!/bin/bash

echo "Installing required system dependencies..."

# Function to check and install dependencies in docker containers
install_dependencies() {
  local container=$1
  echo "Installing dependencies for $container container..."
  
  # Execute command in container
  docker-compose exec -u root $container sh -c 'apt-get update && apt-get install -y --no-install-recommends curl netcat-openbsd'
  
  echo "Dependencies installed for $container."
}

# Create a fix for init_ingestion.sh in case it doesn't exist
fix_init_scripts() {
  echo "Fixing initialization scripts..."
  
  # Check if init_ingestion.sh exists and has correct permissions
  docker-compose exec -u root ingestion sh -c '
    if [ ! -f /app/init_ingestion.sh ]; then
      echo "#!/bin/sh" > /app/init_ingestion.sh
      echo "set -e" >> /app/init_ingestion.sh
      echo "" >> /app/init_ingestion.sh
      echo "echo \"Starting ingestion service...\"" >> /app/init_ingestion.sh
      echo "python /app/app.py" >> /app/init_ingestion.sh
      chmod +x /app/init_ingestion.sh
    else
      chmod +x /app/init_ingestion.sh
    fi
  '
  
  # Check if init_streaming.sh exists and has correct permissions
  docker-compose exec -u root streaming sh -c '
    if [ ! -f /app/init_streaming.sh ]; then
      echo "#!/bin/sh" > /app/init_streaming.sh
      echo "set -e" >> /app/init_streaming.sh
      echo "" >> /app/init_streaming.sh
      echo "echo \"Starting streaming service...\"" >> /app/init_streaming.sh
      echo "python /app/app.py" >> /app/init_streaming.sh
      chmod +x /app/init_streaming.sh
    else
      chmod +x /app/init_streaming.sh
    fi
  '
  
  echo "Initialization scripts fixed."
}

# Install missing Python dependencies
install_python_deps() {
  echo "Installing missing Python dependencies..."
  
  # Install Flask in ingestion service
  docker-compose exec -u root ingestion pip install flask requests python-dotenv kafka-python
  
  # Install dependencies in streaming service
  docker-compose exec -u root streaming pip install flask requests python-dotenv kafka-python river
  
  echo "Python dependencies installed."
}

# Install dependencies in containers
install_dependencies ingestion
install_dependencies streaming

# Fix initialization scripts
fix_init_scripts

# Install missing Python dependencies
install_python_deps

echo "All dependencies installed. Restarting services..."
docker-compose restart ingestion streaming

echo "Done!"
