# Google Cloud Platform Deployment Guide

This guide will help you deploy the TruthGuard application to Google Cloud Platform using App Engine.

## Prerequisites

1. **Google Cloud Account**: Sign up at [cloud.google.com](https://cloud.google.com)
2. **Google Cloud SDK**: Install from [cloud.google.com/sdk](https://cloud.google.com/sdk)
3. **Project Setup**: Create a new project in Google Cloud Console

## Step-by-Step Deployment

### 1. Install Google Cloud SDK

Download and install the Google Cloud SDK for your operating system:
- **Windows**: Download the installer from the official website
- **Mac**: `brew install google-cloud-sdk`
- **Linux**: Follow the official installation guide

### 2. Initialize gcloud

```bash
gcloud init
```

This will:
- Authenticate your Google account
- Set up your default project
- Configure default settings

### 3. Create a New Project (Optional)

```bash
# Create a new project
gcloud projects create truthguard-app --name="TruthGuard"

# Set the project as default
gcloud config set project truthguard-app
```

### 4. Enable Required APIs

```bash
# Enable App Engine API
gcloud services enable appengine.googleapis.com

# Enable Cloud Build API (for deployment)
gcloud services enable cloudbuild.googleapis.com
```

### 5. Initialize App Engine

```bash
# Initialize App Engine in your project
gcloud app create --region=us-central1
```

**Note**: Choose a region close to your users. Popular options:
- `us-central1` (Iowa, USA)
- `europe-west1` (Belgium, Europe)
- `asia-northeast1` (Tokyo, Asia)

### 6. Configure Environment Variables

In the Google Cloud Console:

1. Go to **App Engine > Settings > Environment Variables**
2. Add the following variables:

| Variable Name | Value | Description |
|---------------|-------|-------------|
| `VIRUSTOTAL_API_KEY` | `your_virustotal_api_key` | VirusTotal API key for malicious URL detection |
| `GOOGLE_GENERATIVE_AI_KEY` | `your_google_ai_api_key` | Google AI API key for enhanced analysis |
| `FLASK_ENV` | `production` | Flask environment setting |

### 7. Deploy the Application

Navigate to your project directory and run:

```bash
# Deploy to App Engine
gcloud app deploy

# Deploy with specific version (optional)
gcloud app deploy --version=v1
```

### 8. Open Your Application

```bash
# Open the deployed application in your browser
gcloud app browse
```

## Configuration Files

### app.yaml
The `app.yaml` file configures your App Engine deployment:
- **Runtime**: Python 3.11
- **Scaling**: Automatic scaling (0-10 instances)
- **Instance Class**: F2 (suitable for ML workloads)
- **Timeout**: 300 seconds for ML processing

### requirements.txt
Updated with all necessary dependencies including `gunicorn` for production deployment.

## Monitoring and Management

### View Logs
```bash
# View application logs
gcloud app logs tail -s default

# View logs in browser
gcloud app logs read
```

### Check Application Status
```bash
# Get app information
gcloud app describe

# List versions
gcloud app versions list
```

### Update Application
```bash
# Deploy new version
gcloud app deploy

# Set traffic to new version
gcloud app services set-traffic default --splits=v2=1
```

## Cost Optimization

### Free Tier Limits
App Engine provides:
- 28 instance hours per day
- 1 GB outbound data transfer per day
- 1 GB inbound data transfer per day

### Scaling Configuration
The current configuration:
- **Min instances**: 0 (scales to zero when not in use)
- **Max instances**: 10 (prevents runaway costs)
- **CPU utilization**: 60% (efficient scaling trigger)

## Security Best Practices

1. **API Keys**: Store in environment variables, not in code
2. **HTTPS**: App Engine provides automatic HTTPS
3. **Access Control**: Configure IAM roles as needed
4. **Monitoring**: Enable Cloud Monitoring for alerts

## Troubleshooting

### Common Issues

1. **Deployment Timeout**
   - Solution: Increase timeout in `app.yaml`
   - Check for large dependencies

2. **Memory Limits**
   - Solution: Use higher instance class (F4, F4_1G)
   - Optimize ML model loading

3. **Cold Starts**
   - Solution: Set min_instances > 0
   - Use Cloud Scheduler for warmup requests

### Debug Commands
```bash
# Check deployment status
gcloud app operations list

# View detailed logs
gcloud logging read "resource.type=gae_app"

# Check quotas
gcloud compute project-info describe
```

## Custom Domain (Optional)

1. **Verify Domain**: In Google Cloud Console
2. **Map Domain**: App Engine > Settings > Custom Domains
3. **SSL Certificate**: Automatically provisioned

## Backup and Restore

```bash
# Download source code
gcloud app versions download v1 --version=v1

# List all versions
gcloud app versions list
```

## Support and Resources

- **Documentation**: [cloud.google.com/appengine/docs](https://cloud.google.com/appengine/docs)
- **Pricing**: [cloud.google.com/appengine/pricing](https://cloud.google.com/appengine/pricing)
- **Support**: Google Cloud Support or Stack Overflow

## Quick Commands Reference

```bash
# Deploy application
gcloud app deploy

# View logs
gcloud app logs tail -s default

# Open application
gcloud app browse

# Check status
gcloud app describe

# Update environment variables
# (Use Google Cloud Console GUI)
```

---

Your TruthGuard application is now ready for Google Cloud deployment! 🚀