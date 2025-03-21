#!/bin/bash
set -e

# Check if .env file exists, if not, create from environment variables
if [ ! -f /app/.env ]; then
    echo "Creating .env file from environment variables..."
    
    # Create .env file with required variables
    echo "API_KEY=${API_KEY}" > /app/.env
    echo "PORT=${PORT}" >> /app/.env
    echo "DEBUG_MODE=${DEBUG_MODE:-False}" >> /app/.env
    echo "MODEL_PATH=${MODEL_PATH:-/app/models/segmentation_model.pth}" >> /app/.env
    
    # Add optional variables if they exist
    if [ ! -z "${AWS_ACCESS_KEY_ID}" ]; then
        echo "AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}" >> /app/.env
        echo "AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}" >> /app/.env
        echo "AWS_REGION=${AWS_REGION}" >> /app/.env
    fi
    
    if [ ! -z "${DB_USERNAME}" ]; then
        echo "DB_USERNAME=${DB_USERNAME}" >> /app/.env
        echo "DB_PASSWORD=${DB_PASSWORD}" >> /app/.env
        echo "DB_HOST=${DB_HOST}" >> /app/.env
        echo "DB_PORT=${DB_PORT}" >> /app/.env
        echo "DB_NAME=${DB_NAME}" >> /app/.env
    fi
    
    echo ".env file created successfully."
else
    echo "Using existing .env file."
fi

# Run the application
exec python app.py