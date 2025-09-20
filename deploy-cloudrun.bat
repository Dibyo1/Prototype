@echo off
REM Cloud Run Deployment Script for TruthGuard (Windows)
REM Make sure you have gcloud CLI installed and authenticated

setlocal enabledelayedexpansion

REM Configuration
set PROJECT_ID=your-project-id
set SERVICE_NAME=truthguard-app
set REGION=us-central1
set IMAGE_NAME=gcr.io/%PROJECT_ID%/%SERVICE_NAME%

echo 🚀 Starting Cloud Run deployment for TruthGuard...

REM Check if gcloud is installed
where gcloud >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ gcloud CLI not found. Please install Google Cloud SDK first.
    exit /b 1
)

REM Check if Docker is installed
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker not found. Please install Docker first.
    exit /b 1
)

REM Set the project
echo 📋 Setting project to %PROJECT_ID%...
gcloud config set project %PROJECT_ID%

REM Enable required APIs
echo 🔧 Enabling required APIs...
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com

REM Configure Docker for Google Cloud
echo 🐳 Configuring Docker authentication...
gcloud auth configure-docker

REM Build and push the image
echo 🏗️ Building Docker image...
docker build -t %IMAGE_NAME% .

echo 📤 Pushing image to Google Container Registry...
docker push %IMAGE_NAME%

REM Deploy to Cloud Run
echo ☁️ Deploying to Cloud Run...
gcloud run deploy %SERVICE_NAME% ^
    --image %IMAGE_NAME% ^
    --platform managed ^
    --region %REGION% ^
    --allow-unauthenticated ^
    --memory 2Gi ^
    --cpu 1 ^
    --timeout 300 ^
    --max-instances 10 ^
    --set-env-vars="FLASK_ENV=production,FLASK_DEBUG=false" ^
    --port 5000

REM Get the service URL
for /f "delims=" %%i in ('gcloud run services describe %SERVICE_NAME% --platform managed --region %REGION% --format "value(status.url)"') do set SERVICE_URL=%%i

echo ✅ Deployment completed!
echo 🌐 Your TruthGuard app is available at: %SERVICE_URL%
echo.
echo Next steps:
echo 1. Test your deployment: curl %SERVICE_URL%/health
echo 2. Set up environment variables for API keys:
echo    gcloud run services update %SERVICE_NAME% --set-env-vars="VIRUSTOTAL_API_KEY=your_key"
echo 3. Configure custom domain (optional)
echo.
echo 🛠️ To update your app in the future:
echo    1. Make your changes
echo    2. Run this script again

pause