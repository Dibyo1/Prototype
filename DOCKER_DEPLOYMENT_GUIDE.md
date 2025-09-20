# Docker Deployment Guide for TruthGuard

This guide explains how to containerize and deploy the TruthGuard fake news detection application using Docker with enhanced features including automatic cache cleanup, rate limiting, and optimized performance.

## 📋 Prerequisites

- Docker installed on your system (version 20.10+)
- Docker Compose v2.0+ (for easier management)
- 4GB+ RAM recommended (8GB+ for ML features)
- Internet connection for downloading dependencies
- Windows 10/11, macOS, or Linux

## 🚀 Quick Start

### Option 1: Using Build Scripts (Recommended)

**For Windows:**
```cmd
# Production mode
docker-build.bat

# Development mode
docker-build.bat --mode development

# With Redis caching
docker-build.bat --mode redis
```

**For Linux/macOS:**
```bash
# Make script executable
chmod +x docker-build.sh

# Production mode
./docker-build.sh

# Development mode
./docker-build.sh --mode development

# With Redis caching
./docker-build.sh --mode redis
```

### Option 2: Using Docker Compose Directly

1. **Set up environment variables**
   ```bash
   # Copy template and edit with your API keys
   cp .env.template .env
   ```

2. **Build and run in production mode**
   ```bash
   docker-compose build
   docker-compose --profile production up -d
   ```

3. **Access the application**
   - Application: `http://localhost:80` (via Nginx)
   - Direct access: `http://localhost:5000`
   - Health check: `http://localhost:5000/health`

### Option 3: Using Docker directly

1. **Build the optimized Docker image**
   ```bash
   docker build -t truthguard-app .
   ```

2. **Run with enhanced features**
   ```bash
   docker run -d \
     --name truthguard \
     -p 5000:5000 \
     --memory=2g \
     --cpus=1.0 \
     --restart=unless-stopped \
     truthguard-app
   ```

## 🔧 Configuration Options

### Environment Variables

The application supports comprehensive environment variables for enhanced functionality:

```bash
# API Keys (Optional but recommended for full features)
VIRUSTOTAL_API_KEY=your_virustotal_api_key
GOOGLE_GENERATIVE_AI_API_KEY=your_google_ai_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key

# Application Settings
FLASK_ENV=production
FLASK_DEBUG=false
PORT=5000
SECRET_KEY=your-secure-secret-key

# Database Configuration
DATABASE_URL=sqlite:///app/truthguard.db
# For PostgreSQL: postgresql://user:password@localhost/dbname

# Cache Configuration (Enhanced Features)
REDIS_URL=redis://redis:6379/0
# Leave empty to use SQLite caching

# Logging
LOG_LEVEL=INFO

# Rate Limiting
RATE_LIMIT=100
```

### Enhanced Features Configuration

The updated application includes several enhanced features:

- **Automatic Cache Cleanup**: Intelligent cache management
- **Rate Limiting**: Prevents abuse and ensures fair usage
- **Advanced Caching**: Redis support for better performance
- **Health Monitoring**: Comprehensive health checks
- **Security Features**: Enhanced security measures

### Running with Environment Variables

**Using Docker Compose:**
```bash
# Set environment variables in .env file, then:
docker-compose up
```

**Using Docker directly:**
```bash
docker run -p 5000:5000 \
  -e VIRUSTOTAL_API_KEY=your_key \
  -e GOOGLE_GENERATIVE_AI_API_KEY=your_key \
  --name truthguard \
  truthguard-app
```

## 📁 Volume Mounting

### Models Directory
If you have pre-trained models, mount them to the container:

```bash
docker run -p 5000:5000 \
  -v $(pwd)/models:/app/models:ro \
  --name truthguard \
  truthguard-app
```

### Logs Directory
To persist logs outside the container:

```bash
docker run -p 5000:5000 \
  -v $(pwd)/logs:/app/logs \
  --name truthguard \
  truthguard-app
```

## 🌐 Production Deployment

### With Nginx Reverse Proxy

Use the production profile in docker-compose:

```bash
# Create nginx configuration
cp nginx.conf.template nginx.conf

# Start with nginx reverse proxy
docker-compose --profile production up -d
```

### Resource Limits

For production, set appropriate resource limits:

```bash
docker run -p 5000:5000 \
  --memory=2g \
  --cpus=1.0 \
  --restart=unless-stopped \
  --name truthguard \
  truthguard-app
```

## 🔍 Health Monitoring

The container includes a built-in health check:

```bash
# Check container health
docker ps
docker inspect truthguard --format='{{.State.Health.Status}}'

# View health check logs
docker inspect truthguard --format='{{range .State.Health.Log}}{{.Output}}{{end}}'
```

## 🛠️ Development with Docker

### Development Mode with Live Reload

For development, mount the source code and enable debug mode:

```bash
docker run -p 5000:5000 \
  -v $(pwd):/app \
  -e FLASK_ENV=development \
  -e FLASK_DEBUG=true \
  --name truthguard-dev \
  truthguard-app
```

### Debugging Inside Container

Access the container shell for debugging:

```bash
# Start a bash session in running container
docker exec -it truthguard bash

# Or run a new container with bash
docker run -it --entrypoint bash truthguard-app
```

## 📊 Performance Optimization

### Multi-stage Build (Advanced)

For smaller production images, consider a multi-stage Dockerfile:

```dockerfile
# Build stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
CMD ["python", "app.py"]
```

### Image Size Optimization

Current image size optimizations:
- Uses `python:3.11-slim` base image
- Removes package cache with `--no-cache-dir`
- Combines RUN commands to reduce layers
- Uses `.dockerignore` to exclude unnecessary files

## 🚨 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Change the host port
   docker run -p 8080:5000 truthguard-app
   ```

2. **Memory issues**
   ```bash
   # Increase memory limit
   docker run --memory=4g truthguard-app
   ```

3. **Permission errors**
   ```bash
   # Run as root (not recommended for production)
   docker run --user root truthguard-app
   ```

4. **NLTK data download fails**
   ```bash
   # Pre-download NLTK data in Dockerfile
   RUN python -c "import nltk; nltk.download('punkt')"
   ```

### Viewing Logs

```bash
# View container logs
docker logs truthguard

# Follow logs in real-time
docker logs -f truthguard

# View last 100 lines
docker logs --tail 100 truthguard
```

### Container Management

```bash
# Stop the container
docker stop truthguard

# Start stopped container
docker start truthguard

# Restart container
docker restart truthguard

# Remove container
docker rm truthguard

# Remove image
docker rmi truthguard-app
```

## 🔐 Security Considerations

### Production Security

1. **Non-root user**: The Dockerfile creates and uses a non-root user
2. **Environment variables**: Store sensitive data in environment variables
3. **Network isolation**: Use Docker networks for service isolation
4. **Resource limits**: Set memory and CPU limits
5. **Health checks**: Monitor container health
6. **Log management**: Configure proper logging

### API Keys Security

Never include API keys in the Docker image:

```bash
# Good: Use environment variables
docker run -e VIRUSTOTAL_API_KEY=secret_key truthguard-app

# Bad: Don't hardcode in Dockerfile
ENV VIRUSTOTAL_API_KEY=secret_key  # DON'T DO THIS
```

## 📈 Scaling

### Docker Swarm

For horizontal scaling with Docker Swarm:

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml truthguard-stack

# Scale service
docker service scale truthguard-stack_fake-news-detector=3
```

### Kubernetes

For Kubernetes deployment, convert Docker Compose to Kubernetes manifests:

```bash
# Using kompose
kompose convert -f docker-compose.yml
```

## 🎯 Next Steps

1. **Monitoring**: Add monitoring with Prometheus/Grafana
2. **Logging**: Centralized logging with ELK stack
3. **CI/CD**: Automated builds with GitHub Actions
4. **Registry**: Push images to Docker Hub or private registry
5. **Orchestration**: Deploy to Kubernetes or Docker Swarm

## 📞 Support

If you encounter issues:

1. Check the container logs: `docker logs truthguard`
2. Verify environment variables are set correctly
3. Ensure required ports are not in use
4. Check Docker and system resources
5. Review the application's README.md for additional guidance