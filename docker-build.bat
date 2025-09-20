@echo off
setlocal enabledelayedexpansion

REM TruthGuard Docker Build and Run Script for Windows
REM This script helps build and run the TruthGuard application in different modes

set MODE=production
set FORCE_REBUILD=false

:parse_args
if \"%1\"==\"\" goto :start_deployment
if \"%1\"==\"--mode\" (
    set MODE=%2
    shift
    shift
    goto :parse_args
)
if \"%1\"==\"-m\" (
    set MODE=%2
    shift
    shift
    goto :parse_args
)
if \"%1\"==\"--rebuild\" (
    set FORCE_REBUILD=true
    shift
    goto :parse_args
)
if \"%1\"==\"-r\" (
    set FORCE_REBUILD=true
    shift
    goto :parse_args
)
if \"%1\"==\"--clean\" (
    echo Cleaning all containers and volumes...
    docker-compose down -v --remove-orphans
    docker system prune -f
    set FORCE_REBUILD=true
    shift
    goto :parse_args
)
if \"%1\"==\"--stop\" (
    echo Stopping all containers...
    docker-compose down
    goto :end
)
if \"%1\"==\"--logs\" (
    echo Showing logs...
    docker-compose logs -f
    goto :end
)
if \"%1\"==\"--help\" (
    goto :show_help
)
if \"%1\"==\"-h\" (
    goto :show_help
)
echo Unknown option: %1
goto :show_help

:show_help
echo.
echo TruthGuard Docker Build Script for Windows
echo.
echo Usage: %0 [OPTIONS]
echo.
echo Options:
echo   -m, --mode MODE        Deployment mode: production, development, redis (default: production)
echo   -r, --rebuild          Force rebuild without cache
echo   -c, --clean            Clean all containers and volumes before building
echo   -s, --stop             Stop all running containers
echo   -l, --logs             Show logs from running containers
echo   -h, --help             Show this help message
echo.
echo Examples:
echo   %0                                    # Build and run in production mode
echo   %0 --mode development                 # Run in development mode
echo   %0 --mode production --rebuild        # Force rebuild and run in production
echo   %0 --mode redis                       # Run with Redis caching enabled
echo   %0 --clean                           # Clean and rebuild everything
echo.
goto :end

:start_deployment
echo.
echo ================================================================
echo                    TruthGuard Docker Deployment
echo ================================================================
echo Mode: %MODE%
echo.

REM Validate mode
if \"%MODE%\"==\"production\" goto :mode_valid
if \"%MODE%\"==\"development\" goto :mode_valid
if \"%MODE%\"==\"redis\" goto :mode_valid
echo Invalid mode: %MODE%
echo Valid modes: production, development, redis
goto :end

:mode_valid

REM Check if .env file exists
if not exist \".env\" (
    echo Warning: No .env file found. 
    if exist \".env.template\" (
        echo Copying from .env.template...
        copy \".env.template\" \".env\"
        echo Created .env file from template
    ) else (
        echo ERROR: No .env.template found. Please create a .env file manually.
        goto :end
    )
)

REM Build arguments
set BUILD_ARGS=
if \"%FORCE_REBUILD%\"==\"true\" (
    set BUILD_ARGS=--no-cache --force-rm
)

REM Build and run based on mode
if \"%MODE%\"==\"production\" (
    echo Building and starting in production mode...
    docker-compose build %BUILD_ARGS%
    docker-compose --profile production up -d
) else if \"%MODE%\"==\"development\" (
    echo Building and starting in development mode...
    docker-compose build %BUILD_ARGS%
    docker-compose --profile development up
) else if \"%MODE%\"==\"redis\" (
    echo Building and starting with Redis caching...
    docker-compose build %BUILD_ARGS%
    docker-compose --profile redis up -d
)

REM Show status
echo.
echo Container Status:
docker-compose ps

echo.
echo Application URLs:
if \"%MODE%\"==\"production\" (
    echo   • Application: http://localhost:80 (via Nginx)
    echo   • Direct App:  http://localhost:5000
) else (
    echo   • Application: http://localhost:5000
)
echo   • Health Check: http://localhost:5000/health

echo.
echo Useful commands:
echo   • View logs:        docker-compose logs -f
echo   • Stop containers:  docker-compose down
echo   • Restart:          docker-compose restart
echo   • Shell access:     docker exec -it truthguard-app /bin/bash
echo.
echo Deployment complete!

:end
pause