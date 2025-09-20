# Cloud Run Troubleshooting Guide - Updated

## ❌ Current Issue: Container Failed to Start

**Error Message**: "The user-provided container failed to start and listen on the port defined provided by the PORT=5000 environment variable within the allocated timeout."

### 🔍 Root Cause Analysis

This error occurs when:
1. **Container startup takes too long** (Cloud Run has timeout limits)
2. **Application doesn't bind to correct port** (must use PORT environment variable)
3. **Health check fails** (application not responding on /health endpoint)
4. **Memory/CPU insufficient** for ML model loading
5. **Dependencies loading slowly** (PyTorch, transformers, etc.)

### ✅ Applied Fixes

#### 1. **Extended Timeouts & Health Checks**
```yaml
# In cloudrun-service.yaml
run.googleapis.com/timeout: "900s"  # 15 minutes
startupProbe:
  initialDelaySeconds: 20
  timeoutSeconds: 10
  failureThreshold: 6  # Allow more failures
livenessProbe:
  initialDelaySeconds: 60  # Wait longer before first check
```

#### 2. **Optimized Memory Usage**
```yaml
# Reduced memory to stay within quota limits
run.googleapis.com/memory: "1Gi"
autoscaling.knative.dev/maxScale: "2"
```

#### 3. **Enhanced Gunicorn Configuration**
```dockerfile
# Extended timeouts for ML model loading
CMD gunicorn \
  --bind 0.0.0.0:$PORT \
  --timeout 600 \
  --graceful-timeout 30 \
  --workers 2 \
  --preload \
  app:app
```

#### 4. **Application Startup Logging**
```python
# Enhanced logging for debugging startup issues
print(f"🚀 Flask app initialized for Cloud Run")
print(f"🌐 Port from environment: {os.environ.get('PORT', '5000')}")
print(f"🏗️ Running in: {'Cloud Run' if os.environ.get('K_SERVICE') else 'Local/Other'}")
```

This guide helps you troubleshoot common Cloud Run deployment issues for the TruthGuard application.

## Common Issues and Solutions

### 1. Container Failed to Start and Listen on Port

**Error Message:**
```
The user-provided container failed to start and listen on the port defined provided by the PORT=5000 environment variable within the allocated timeout.
```

**Causes and Solutions:**

#### A. Port Configuration Issues
- **Problem**: App not listening on the PORT environment variable
- **Solution**: Ensure your app reads `PORT` from environment variables
```python
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)
```

#### B. Binding to Wrong Interface
- **Problem**: App binding to localhost (127.0.0.1) instead of 0.0.0.0
- **Solution**: Always bind to `0.0.0.0` in Cloud Run
```python
app.run(host='0.0.0.0', port=port)  # ✅ Correct
app.run(host='127.0.0.1', port=port)  # ❌ Wrong
```

#### C. Startup Timeout
- **Problem**: App takes too long to start
- **Solution**: 
  - Increase startup timeout in service configuration
  - Optimize dependency loading
  - Remove unnecessary ML model downloads at startup

### 2. Memory and CPU Issues

**Symptoms:**
- Container killed due to memory limits
- Slow response times
- Timeout errors

**Solutions:**
```yaml
# In cloudrun-service.yaml
resources:
  limits:
    cpu: "2"      # Increase from 1
    memory: "4Gi" # Increase from 2Gi
```

### 3. Dependencies Installation Issues

**Problem**: Missing or incompatible dependencies

**Solution:**
1. Use CPU-only versions of ML libraries:
```txt
torch --index-url https://download.pytorch.org/whl/cpu
```

2. Remove problematic dependencies:
```txt
# Remove opencv-python if causing issues
# Use Pillow for image processing instead
```

### 4. Environment Variables Not Set

**Problem**: API keys or configuration not available

**Solution:**
```bash
# Set environment variables during deployment
gcloud run services update truthguard-app \
  --set-env-vars="VIRUSTOTAL_API_KEY=your_key,GOOGLE_GENERATIVE_AI_KEY=your_key"
```

### 5. Health Check Failures

**Problem**: Health check endpoint returning errors

**Solution:**
1. Verify health endpoint works:
```python
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': time.time()})
```

2. Test locally:
```bash
curl http://localhost:5000/health
```

## Debugging Commands

### View Logs
```bash
# Real-time logs
gcloud run services logs tail truthguard-app

# Recent logs
gcloud run services logs read truthguard-app --limit=100
```

### Check Service Status
```bash
# Service details
gcloud run services describe truthguard-app

# List all revisions
gcloud run revisions list --service=truthguard-app
```

### Test Deployment Locally
```bash
# Build and test locally first
docker build -t truthguard-local .
docker run -p 5000:5000 -e PORT=5000 truthguard-local

# Test health endpoint
curl http://localhost:5000/health
```

## Performance Optimization

### 1. Reduce Image Size
- Use `.dockerignore` to exclude unnecessary files
- Remove unused dependencies
- Use multi-stage builds if needed

### 2. Optimize Startup Time
```python
# Lazy load heavy dependencies
def get_ml_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model
```

### 3. Configure Autoscaling
```yaml
annotations:
  autoscaling.knative.dev/minScale: "1"  # Keep 1 instance warm
  autoscaling.knative.dev/maxScale: "10"
```

## Step-by-Step Deployment Checklist

1. **Pre-deployment Checks:**
   - [ ] Test locally with Docker
   - [ ] Verify health endpoint works
   - [ ] Check all environment variables
   - [ ] Ensure port configuration is correct

2. **Build and Push:**
   ```bash
   docker build -t gcr.io/PROJECT_ID/truthguard-app .
   docker push gcr.io/PROJECT_ID/truthguard-app
   ```

3. **Deploy:**
   ```bash
   gcloud run deploy truthguard-app \
     --image gcr.io/PROJECT_ID/truthguard-app \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 1 \
     --timeout 300 \
     --port 5000
   ```

4. **Post-deployment Verification:**
   ```bash
   # Test health endpoint
   curl https://YOUR_SERVICE_URL/health
   
   # Check logs
   gcloud run services logs tail truthguard-app
   ```

## Environment Variables Setup

Set these environment variables for full functionality:

```bash
gcloud run services update truthguard-app \
  --set-env-vars="
    FLASK_ENV=production,
    FLASK_DEBUG=false,
    VIRUSTOTAL_API_KEY=your_virustotal_key,
    GOOGLE_GENERATIVE_AI_KEY=your_google_ai_key
  "
```

## Monitoring and Alerts

### Set up monitoring:
1. Go to Cloud Monitoring in Google Cloud Console
2. Create alerts for:
   - High error rates
   - Memory usage > 80%
   - Response time > 10 seconds

### Key metrics to monitor:
- Request count
- Error rate
- Memory utilization
- CPU utilization
- Cold start frequency

## Getting Help

If you continue experiencing issues:

1. **Check Cloud Logging:**
   - Go to Cloud Logging in Google Cloud Console
   - Filter by your Cloud Run service

2. **Community Support:**
   - Stack Overflow with `google-cloud-run` tag
   - Google Cloud Community on Reddit

3. **Google Cloud Support:**
   - File a support ticket if you have a support plan

## Quick Fix Commands

```bash
# Redeploy with increased resources
gcloud run deploy truthguard-app \
  --image gcr.io/PROJECT_ID/truthguard-app \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600

# Check current configuration
gcloud run services describe truthguard-app \
  --format="export" > current-config.yaml

# Rollback to previous revision
gcloud run services update-traffic truthguard-app \
  --to-revisions=PREVIOUS_REVISION=100
```

Remember: Always test locally before deploying to Cloud Run!

## 🚀 **High-Performance Deployment (8GB + 2 CPUs)**

**Updated Deployment Command**:
```bash
gcloud run deploy truthguard-app \
  --source . \
  --platform managed \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 2 \
  --timeout 900 \
  --concurrency 80 \
  --max-instances 1 \
  --set-env-vars="FLASK_ENV=production,FLASK_DEBUG=false" \
  --port 5000
```

**Resource Configuration**:
- **Memory**: 8GB (sufficient for ML models + PyTorch)
- **CPU**: 2 cores (faster model loading and processing)  
- **Max Instances**: 1 (to respect quota limits)
- **Timeout**: 900s (15 minutes for startup)
- **Concurrency**: 80 requests per instance

**Cloud Build Alternative** (for cache issues):
```bash
# Force fresh build with no cache
gcloud builds submit --config cloudbuild-8gb.yaml .
```

**Memory Usage Breakdown**:
- PyTorch models: ~2-3GB
- Transformers cache: ~1-2GB  
- Application runtime: ~1GB
- System overhead: ~1GB
- **Total**: ~5-7GB (8GB provides buffer)

**Benefits of Higher Resources**:
- ✅ Faster model loading (2 CPUs)
- ✅ Better concurrent request handling
- ✅ Reduced cold start times
- ✅ More stable under load