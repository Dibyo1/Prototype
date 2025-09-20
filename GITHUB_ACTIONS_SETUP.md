# GitHub Actions Deployment Setup

This guide explains how to set up automated deployment to Google Cloud Run using GitHub Actions.

## Prerequisites

1. **GitHub Repository**: Your code must be in a GitHub repository
2. **Google Cloud Project**: Project ID `intense-zoo-472518-r6`
3. **Cloud Run Service**: Service name `fake-news-detector`
4. **Service Account**: Google Cloud service account with deployment permissions

## Setup Steps

### 1. Create Service Account

```bash
# Create a service account
gcloud iam service-accounts create github-actions \
    --description="Service account for GitHub Actions deployment" \
    --display-name="GitHub Actions"

# Grant necessary roles
gcloud projects add-iam-policy-binding intense-zoo-472518-r6 \
    --member="serviceAccount:github-actions@intense-zoo-472518-r6.iam.gserviceaccount.com" \
    --role="roles/run.developer"

gcloud projects add-iam-policy-binding intense-zoo-472518-r6 \
    --member="serviceAccount:github-actions@intense-zoo-472518-r6.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding intense-zoo-472518-r6 \
    --member="serviceAccount:github-actions@intense-zoo-472518-r6.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.admin"

# Create and download service account key
gcloud iam service-accounts keys create github-actions-key.json \
    --iam-account=github-actions@intense-zoo-472518-r6.iam.gserviceaccount.com
```

### 2. Configure GitHub Secrets

1. Go to your GitHub repository
2. Navigate to **Settings > Secrets and variables > Actions**
3. Click **New repository secret**
4. Add the following secret:

   - **Name**: `GCP_SA_KEY`
   - **Value**: Copy the entire content of `github-actions-key.json`

### 3. Optional: Add API Keys (if needed)

Add these secrets for enhanced functionality:

- **Name**: `VIRUSTOTAL_API_KEY`
  - **Value**: Your VirusTotal API key

- **Name**: `GOOGLE_GENERATIVE_AI_KEY`
  - **Value**: Your Google AI API key

### 4. Deployment Configuration

The workflow is configured with:
- **Region**: `asia-south1` (Mumbai, India)
- **Memory**: 2Gi
- **Timeout**: 900 seconds (15 minutes)
- **Max Instances**: 2
- **Min Instances**: 0
- **Port**: 5000 (Flask standard)

### 5. Trigger Deployment

Push to the `main` branch to trigger automatic deployment:

```bash
git add .
git commit -m "Deploy to Cloud Run"
git push origin main
```

## Workflow Features

- ✅ **Automatic deployment** on push to main branch
- ✅ **Optimized builds** with .gcloudignore
- ✅ **Proper scaling** configuration
- ✅ **Environment variables** setup
- ✅ **URL output** after successful deployment

## Monitoring

After deployment, you can:

1. **Check deployment status** in GitHub Actions tab
2. **View logs** in Google Cloud Console
3. **Monitor service** in Cloud Run dashboard
4. **Access your app** at the deployed URL

## Troubleshooting

### Common Issues:

1. **Permission denied**: Ensure service account has correct roles
2. **Build timeout**: Increase timeout in workflow file
3. **Memory issues**: Adjust memory allocation in workflow
4. **Secret not found**: Verify `GCP_SA_KEY` secret is properly set

### Useful Commands:

```bash
# Check service status
gcloud run services list --region=asia-south1

# View service details
gcloud run services describe fake-news-detector --region=asia-south1

# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=fake-news-detector" --limit=50
```

## Security Notes

- Service account key should never be committed to repository
- Use GitHub secrets for all sensitive information
- Regularly rotate service account keys
- Monitor access logs for suspicious activity