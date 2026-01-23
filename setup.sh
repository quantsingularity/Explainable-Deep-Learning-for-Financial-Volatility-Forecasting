#!/bin/bash
# Production Setup Script
# Automates the setup and deployment of the volatility forecasting system

set -e  # Exit on error

echo "=================================================="
echo "  Volatility Forecasting System - Setup Script"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${NC}ℹ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Docker
    if command -v docker &> /dev/null; then
        print_success "Docker is installed ($(docker --version))"
    else
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        print_success "Docker Compose is installed ($(docker-compose --version))"
    else
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Python (for local development)
    if command -v python3 &> /dev/null; then
        print_success "Python 3 is installed ($(python3 --version))"
    else
        print_warning "Python 3 is not installed. Docker-only deployment will be used."
    fi
    
    echo ""
}

# Create necessary directories
create_directories() {
    print_info "Creating necessary directories..."
    
    mkdir -p data models figures tables logs mlruns checkpoints monitoring
    
    print_success "Directories created"
    echo ""
}

# Setup monitoring configuration
setup_monitoring() {
    print_info "Setting up monitoring configuration..."
    
    # Create Prometheus config
    cat > monitoring/prometheus.yml <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'api-server'
    static_configs:
      - targets: ['api-server:8000']
    metrics_path: '/metrics'
EOF
    
    # Create Grafana datasources
    cat > monitoring/grafana-datasources.yml <<EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF
    
    print_success "Monitoring configuration created"
    echo ""
}

# Setup environment file
setup_environment() {
    print_info "Setting up environment configuration..."
    
    if [ ! -f .env ]; then
        cat > .env <<EOF
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:mlflow_password@postgres:5432/mlflow_db

# PostgreSQL Configuration
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow_password
POSTGRES_DB=mlflow_db

# Redis Configuration
REDIS_URL=redis://redis:6379

# API Configuration
PORT=8000

# TensorFlow Configuration
TF_CPP_MIN_LOG_LEVEL=2
TF_FORCE_GPU_ALLOW_GROWTH=true

# Model Configuration
MODEL_PATH=/app/models/lstm_attention_model.h5

# Logging
LOG_LEVEL=INFO
EOF
        print_success "Environment file created (.env)"
    else
        print_warning "Environment file already exists (.env)"
    fi
    echo ""
}

# Build Docker images
build_images() {
    print_info "Building Docker images..."
    
    read -p "Build GPU version? (y/N): " build_gpu
    
    if [[ $build_gpu =~ ^[Yy]$ ]]; then
        print_info "Building GPU images..."
        docker-compose build training-gpu
        print_success "GPU images built"
    else
        print_info "Building CPU images..."
        docker-compose build training-cpu
        print_success "CPU images built"
    fi
    
    # Build API image
    docker-compose build api-server
    print_success "API server image built"
    
    echo ""
}

# Start core services
start_core_services() {
    print_info "Starting core services (PostgreSQL, MLflow)..."
    
    docker-compose up -d postgres mlflow
    
    print_info "Waiting for services to be ready..."
    sleep 10
    
    # Check if services are healthy
    if docker-compose ps | grep -q "healthy"; then
        print_success "Core services are running"
    else
        print_warning "Some services may not be fully ready yet"
    fi
    
    echo ""
}

# Download sample data
download_data() {
    print_info "Checking for sample data..."
    
    if [ ! -f data/synthetic_data.csv ]; then
        print_info "Generating synthetic data..."
        
        if command -v python3 &> /dev/null; then
            python3 code/data_generator.py
            print_success "Synthetic data generated"
        else
            print_warning "Python not available. Will generate data in container during training."
        fi
    else
        print_success "Sample data already exists"
    fi
    
    echo ""
}

# Train initial model
train_initial_model() {
    print_info "Training initial model..."
    
    read -p "Train initial model now? (y/N): " train_now
    
    if [[ $train_now =~ ^[Yy]$ ]]; then
        read -p "Use GPU? (y/N): " use_gpu
        
        if [[ $use_gpu =~ ^[Yy]$ ]]; then
            print_info "Starting GPU training..."
            docker-compose --profile training-gpu up
        else
            print_info "Starting CPU training..."
            docker-compose --profile training-cpu up
        fi
        
        print_success "Training completed"
    else
        print_info "Skipping initial training"
    fi
    
    echo ""
}

# Print access information
print_access_info() {
    echo ""
    echo "=================================================="
    echo "  Setup Complete! Access Information:"
    echo "=================================================="
    echo ""
    echo "🔬 MLflow Tracking UI:     http://localhost:5000"
    echo "🚀 API Server:             http://localhost:8000"
    echo "📊 API Documentation:      http://localhost:8000/docs"
    echo "📈 Prometheus:             http://localhost:9090"
    echo "📊 Grafana:                http://localhost:3000"
    echo "   └─ Default credentials: admin / admin"
    echo "📓 Jupyter Notebook:       http://localhost:8888"
    echo ""
    echo "=================================================="
    echo "  Useful Commands:"
    echo "=================================================="
    echo ""
    echo "# Start API server:"
    echo "docker-compose --profile api up -d"
    echo ""
    echo "# Start training (CPU):"
    echo "docker-compose --profile training-cpu up"
    echo ""
    echo "# Start monitoring:"
    echo "docker-compose --profile monitoring up -d"
    echo ""
    echo "# View logs:"
    echo "docker-compose logs -f [service-name]"
    echo ""
    echo "# Stop all services:"
    echo "docker-compose down"
    echo ""
    echo "# Run ablation study:"
    echo "python code/ablation_study.py"
    echo ""
    echo "# Run trading backtest:"
    echo "python code/trading_backtest.py"
    echo ""
}

# Main setup flow
main() {
    echo "Starting setup process..."
    echo ""
    
    check_prerequisites
    create_directories
    setup_monitoring
    setup_environment
    
    read -p "Build Docker images? (Y/n): " build_choice
    if [[ ! $build_choice =~ ^[Nn]$ ]]; then
        build_images
    fi
    
    read -p "Start core services (PostgreSQL, MLflow)? (Y/n): " services_choice
    if [[ ! $services_choice =~ ^[Nn]$ ]]; then
        start_core_services
    fi
    
    download_data
    train_initial_model
    
    print_access_info
    
    print_success "Setup complete! Your volatility forecasting system is ready."
}

# Run main setup
main
