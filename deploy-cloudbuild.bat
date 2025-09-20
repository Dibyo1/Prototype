@echo off
REM Cloud Build + Cloud Run Deployment Script for TruthGuard (Windows)
REM This script uses Google Cloud Build instead of local Docker

setlocal enabledelayedexpansion

REM Configuration
set PROJECT_ID=truthguard-472613
set SERVICE_NAME=prototype
set REGION=us-central1

echo 🚀 Starting Cloud Build + Cloud Run deployment for TruthGuard...

REM Test the app locally first
echo 🧪 Testing app locally before deployment...
python test_startup.py
if %errorlevel% neq 0 (
    echo ❌ Local app test failed. Please fix the issues before deploying.
    exit /b 1
)
echo ✅ Local test passed!

REM Check if gcloud is installed
where gcloud >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ gcloud CLI not found. Please install Google Cloud SDK first.
    exit /b 1
)

REM Set the project
echo 📋 Setting project to %PROJECT_ID%...
gcloud config set project %PROJECT_ID%

REM Enable required APIs
echo 🔧 Enabling required APIs...
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

REM Submit build to Cloud Build
echo 🏗️ Building and deploying with Cloud Build...
gcloud builds submit --config cloudbuild.yaml .

if %errorlevel% neq 0 (
    echo ❌ Build failed. Please check the logs for details.
    exit /b 1
)

REM Get the service URL
echo 🔍 Getting service URL...
for /f "delims=" %%i in ('gcloud run services describe %SERVICE_NAME% --platform managed --region %REGION% --format "value(status.url)"') do set SERVICE_URL=%%i

echo ✅ Deployment completed!
echo 🌐 Your TruthGuard app is available at: %SERVICE_URL%
echo.
echo 🧪 Testing the deployed service...
timeout 10 >nul
curl -f %SERVICE_URL%/health
if %errorlevel% equ 0 (
    echo ✅ Health check passed!
) else (
    echo ⚠️ Health check failed. Checking logs...
    gcloud run services logs read %SERVICE_NAME% --limit=20
)
echo.
echo Next steps:
echo 1. Check logs: gcloud run services logs tail %SERVICE_NAME%
echo 2. Set up environment variables for API keys:
echo    gcloud run services update %SERVICE_NAME% --set-env-vars="VIRUSTOTAL_API_KEY=your_key"
echo 3. Configure custom domain (optional)
echo.
echo 🛠️ To update your app in the future:
echo    1. Make your changes
echo    2. Run this script again

pause