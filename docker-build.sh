#!/bin/bash

# TruthGuard Docker Build and Run Script
# This script helps build and run the TruthGuard application in different modes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODE="production"
BUILD_CACHE=true
FORCE_REBUILD=false

# Help function
show_help() {
    echo -e "${BLUE}TruthGuard Docker Build Script${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --mode MODE        Deployment mode: production, development, redis (default: production)"
    echo "  -r, --rebuild          Force rebuild without cache"
    echo "  -c, --clean            Clean all containers and volumes before building"
    echo "  -s, --stop             Stop all running containers"
    echo "  -l, --logs             Show logs from running containers"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Build and run in production mode"
    echo "  $0 --mode development                 # Run in development mode"
    echo "  $0 --mode production --rebuild        # Force rebuild and run in production"
    echo "  $0 --mode redis                       # Run with Redis caching enabled"
    echo "  $0 --clean                           # Clean and rebuild everything"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -r|--rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        -c|--clean)
            echo -e "${YELLOW}Cleaning all containers and volumes...${NC}"
            docker-compose down -v --remove-orphans
            docker system prune -f
            FORCE_REBUILD=true
            shift
            ;;
        -s|--stop)
            echo -e "${YELLOW}Stopping all containers...${NC}"
            docker-compose down
            exit 0
            ;;
        -l|--logs)
            echo -e "${BLUE}Showing logs...${NC}"
            docker-compose logs -f
            exit 0
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Validate mode
case $MODE in
    production|development|redis)
        ;;
    *)
        echo -e "${RED}Invalid mode: $MODE${NC}"
        echo "Valid modes: production, development, redis"
        exit 1
        ;;
esac

echo -e "${BLUE}🐳 TruthGuard Docker Deployment${NC}"
echo -e "${YELLOW}Mode: $MODE${NC}"
echo ""

# Check if .env file exists
if [[ ! -f .env ]]; then
    echo -e "${YELLOW}⚠️  No .env file found. Copying from .env.template...${NC}"
    if [[ -f .env.template ]]; then
        cp .env.template .env
        echo -e "${GREEN}✅ Created .env file from template${NC}"
    else
        echo -e "${RED}❌ No .env.template found. Please create a .env file manually.${NC}"
        exit 1
    fi
fi

# Build arguments
BUILD_ARGS=""
if [[ "$FORCE_REBUILD" == true ]]; then
    BUILD_ARGS="--no-cache --force-rm"
fi

# Build and run based on mode
case $MODE in
    production)
        echo -e "${GREEN}🚀 Building and starting in production mode...${NC}"
        docker-compose build $BUILD_ARGS
        docker-compose --profile production up -d
        ;;
    development)
        echo -e "${GREEN}🔧 Building and starting in development mode...${NC}"
        docker-compose build $BUILD_ARGS
        docker-compose --profile development up
        ;;
    redis)
        echo -e "${GREEN}🗄️  Building and starting with Redis caching...${NC}"
        docker-compose build $BUILD_ARGS
        docker-compose --profile redis up -d
        ;;
esac

# Show status
echo ""
echo -e "${GREEN}📋 Container Status:${NC}"
docker-compose ps

echo ""
echo -e "${BLUE}🌐 Application URLs:${NC}"
if [[ "$MODE" == "production" ]]; then
    echo "  • Application: http://localhost:80 (via Nginx)"
    echo "  • Direct App:  http://localhost:5000"
else
    echo "  • Application: http://localhost:5000"
fi
echo "  • Health Check: http://localhost:5000/health"

echo ""
echo -e "${YELLOW}💡 Useful commands:${NC}"
echo "  • View logs:        docker-compose logs -f"
echo "  • Stop containers:  docker-compose down"
echo "  • Restart:          docker-compose restart"
echo "  • Shell access:     docker exec -it truthguard-app /bin/bash"
echo ""
echo -e "${GREEN}✅ Deployment complete!${NC}"