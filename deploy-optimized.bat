@echo off
echo 🚀 TruthGuard Cloud Run Deployment Script
echo ==========================================

REM Check if gcloud is installed
gcloud version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Google Cloud SDK not found. Please install gcloud CLI.
    pause
    exit /b 1
)

REM Get project ID
echo 📋 Getting current project ID...
for /f "tokens=*" %%i in ('gcloud config get-value project 2^>nul') do set PROJECT_ID=%%i

if "%PROJECT_ID%"=="" (
    echo ❌ No Google Cloud project set. Please run: gcloud config set project YOUR_PROJECT_ID
    pause
    exit /b 1
)

echo ✅ Using project: %PROJECT_ID%

REM Pre-deployment verification
echo.
echo 🧪 Running pre-deployment tests...
python test_deployment.py
if %errorlevel% neq 0 (
    echo ❌ Pre-deployment tests failed. Please fix issues before deploying.
    pause
    exit /b 1
)

echo ✅ Pre-deployment tests passed!

REM Build and deploy with optimized settings
echo.
echo 🏗️ Building and deploying to Cloud Run...
echo This may take 10-15 minutes due to ML dependencies...

gcloud run deploy truthguard-app ^
  --source . ^
  --platform managed ^
  --region us-central1 ^
  --allow-unauthenticated ^
  --memory 1Gi ^
  --cpu 1 ^
  --timeout 900 ^
  --concurrency 80 ^
  --max-instances 2 ^
  --set-env-vars="FLASK_ENV=production,FLASK_DEBUG=false" ^
  --port 5000

if %errorlevel% equ 0 (
    echo.
    echo 🎉 Deployment successful!
    echo 📱 Your app should be running at:
    gcloud run services describe truthguard-app --region=us-central1 --format="value(status.url)"
    echo.
    echo 🔍 Monitor logs with:
    echo gcloud logs tail --follow --format="value(textPayload)" --filter="resource.type=cloud_run_revision AND resource.labels.service_name=truthguard-app"
) else (
    echo.
    echo ❌ Deployment failed. Check the error messages above.
    echo 📋 Common issues:
    echo   - Quota exceeded (increase Cloud Run quotas)
    echo   - Build timeout (dependencies too large)
    echo   - Memory limits (reduce memory usage)
    echo.
    echo 🔍 Check logs with:
    echo gcloud logs tail --format="value(textPayload)" --filter="resource.type=cloud_run_revision AND resource.labels.service_name=truthguard-app"
)

echo.
pause