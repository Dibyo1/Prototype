@echo off
echo 🚀 TruthGuard High-Performance Cloud Run Deployment
echo ==================================================
echo.
echo Configuration:
echo - Memory: 8GB
echo - CPU: 2 cores  
echo - Max Instances: 1
echo - Timeout: 900s
echo.

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

echo.
echo 🏗️ Deploying TruthGuard with High-Performance Configuration...
echo This may take 15-20 minutes due to ML dependencies and fresh build...

REM Deploy with high-performance configuration
gcloud run deploy truthguard-app ^
  --source . ^
  --platform managed ^
  --region asia-south1 ^
  --allow-unauthenticated ^
  --memory 8Gi ^
  --cpu 2 ^
  --timeout 900 ^
  --concurrency 80 ^
  --max-instances 1 ^
  --set-env-vars="FLASK_ENV=production,FLASK_DEBUG=false" ^
  --port 5000

if %errorlevel% equ 0 (
    echo.
    echo 🎉 High-Performance Deployment Successful!
    echo 📊 Configuration:
    echo    - Memory: 8GB
    echo    - CPU: 2 cores
    echo    - Max Instances: 1
    echo    - Timeout: 15 minutes
    echo.
    echo 📱 Your app should be running at:
    gcloud run services describe truthguard-app --region=asia-south1 --format="value(status.url)"
    echo.
    echo 🔍 Monitor logs with:
    echo gcloud logs tail --follow --format="value(textPayload)" --filter="resource.type=cloud_run_revision AND resource.labels.service_name=truthguard-app"
) else (
    echo.
    echo ❌ High-Performance Deployment failed. Check the error messages above.
    echo.
    echo 💡 Alternative: Force fresh build with Cloud Build
    echo gcloud builds submit --config cloudbuild-8gb.yaml .
    echo.
    echo 🔍 Check logs with:
    echo gcloud logs tail --format="value(textPayload)" --filter="resource.type=cloud_run_revision AND resource.labels.service_name=truthguard-app"
)

echo.
pause