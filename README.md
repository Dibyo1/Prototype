# TruthGuard - Advanced Information Verification Platform

TruthGuard is a comprehensive Flask-based web application that combines multiple verification technologies to combat misinformation. The platform integrates machine learning, external security APIs, and steganography tools to provide a complete information integrity solution.

## 🚀 Core Features

### 📰 **Fact Checking**
- AI-powered claim analysis using machine learning models
- Real-time fact verification against trusted sources
- Confidence scoring and detailed explanations
- Support for multiple content types (text, articles, statements)

### 🔗 **Link Verification** 
- URL safety assessment using multiple security databases
- Malicious website detection and categorization
- Real-time threat intelligence integration
- Comprehensive safety reports with risk levels

### 🛡️ **Malicious URL Detection**
- Advanced threat detection using VirusTotal API
- Multi-engine scanning for comprehensive coverage
- Categorized threat classification
- Historical threat data analysis

### 🖼️ **Image Steganography Tools**
- **Message Encoding**: Hide secret messages within images using LSB (Least Significant Bit) technique
- **Message Decoding**: Extract hidden messages from steganographic images
- **Integrity Verification**: Ensure message authenticity and completeness
- **Support for multiple image formats**: PNG, JPEG, BMP
- **Base64 image processing**: Direct browser integration without file uploads

### 🔧 **REST API**
- Comprehensive JSON API endpoints for all features
- Easy integration with external applications
- Standardized response formats
- Robust error handling and validation

## Deployment Options

### 🌐 Google Cloud Platform (Recommended)

Deploy to Google Cloud App Engine for production use:

1. **Quick Deployment**: See [`GCP_DEPLOYMENT_GUIDE.md`](GCP_DEPLOYMENT_GUIDE.md) for detailed instructions
2. **Install Google Cloud SDK**: [cloud.google.com/sdk](https://cloud.google.com/sdk)
3. **Deploy**: `gcloud app deploy`
4. **Access**: `gcloud app browse`

**Benefits**:
- Automatic scaling (0-10 instances)
- HTTPS included
- 28 free instance hours per day
- Integrated monitoring and logging
- Custom domain support

### 💻 Local Development

### Quick Start Options

#### Option 1: Using Startup Scripts (Recommended)

**For Windows:**
```bash
start.bat
```

**For Mac/Linux:**
```bash
chmod +x start.sh
./start.sh
```

#### Option 2: Manual Setup

1. **Clone** this repository to your local machine
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run the application**: `python app.py`
4. **Visit** `http://localhost:5000` in your browser

### Requirements

- Python 3.7 or higher
- pip package manager
- Internet connection for external API features (optional)

### Optional Configuration

For enhanced features, you can set environment variables:

- `VIRUSTOTAL_API_KEY`: Your VirusTotal API key (for enhanced URL scanning)
- `GOOGLE_GENERATIVE_AI_KEY`: Your Google AI API key (for advanced analysis)

## 🌐 API Endpoints

### Fact Checking
- `POST /api/fact-check`: Analyze claims for accuracy and provide verification results
  - **Input**: `{"claim": "statement to verify"}`
  - **Output**: Verification status, confidence score, and detailed analysis

### Link Security
- `POST /api/verify-link`: Comprehensive URL safety verification
  - **Input**: `{"url": "https://example.com"}`
  - **Output**: Safety status, threat categories, and security assessment

- `POST /api/malicious-detection`: Advanced malicious URL detection
  - **Input**: `{"url": "https://suspicious-site.com"}`
  - **Output**: Threat detection results from multiple security engines

### Steganography
- `POST /api/steganography/encode`: Hide messages within images
  - **Input**: `{"image_data": "base64_image", "message": "secret_text"}`
  - **Output**: Steganographic image with embedded message

- `POST /api/steganography/decode`: Extract hidden messages from images
  - **Input**: `{"image_data": "base64_steganographic_image"}`
  - **Output**: Extracted message and verification status

### Image Analysis
- `POST /api/analyze-image`: General image analysis capabilities
  - **Input**: `{"image_data": "base64_image"}`
  - **Output**: Image metadata and analysis results

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd TruthGuard
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the application**: Open your browser and go to `http://localhost:5000`

## 📁 Project Structure

```
TruthGuard/
├── app.py                          # Main Flask application with all API endpoints
├── index.html                      # Complete web interface with integrated CSS/JS
├── misinfo_ir_ml_pipeline.py       # Machine learning pipeline for fact-checking
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python runtime specification
├── app.yaml                        # Google Cloud App Engine configuration
├── .gcloudignore                   # Google Cloud deployment exclusions
├── GCP_DEPLOYMENT_GUIDE.md         # Detailed Google Cloud deployment guide
├── start.bat                       # Windows startup script
├── start.sh                        # Unix/Linux startup script
├── .gitignore                      # Git ignore rules
└── models/                         # Pre-trained ML models
    ├── aux_classifier.joblib       # Auxiliary classifier model (313KB)
    ├── aux_label_encoder.joblib    # Label encoder for classification (0.5KB)
    └── tfidf_vectorizer.joblib     # TF-IDF vectorizer for text processing (184KB)
```

### 📋 Key Files Description

- **`app.py`** (45KB): Core Flask application containing:
  - All API route handlers
  - Steganography encoding/decoding algorithms
  - Machine learning model integration
  - Error handling and logging
  - Security middleware

- **`index.html`** (74KB): Single-page application featuring:
  - Responsive design with modern UI components
  - Interactive steganography tools with tabbed interface
  - Real-time API integration
  - Custom cursor animations
  - Mobile-optimized layouts

- **`misinfo_ir_ml_pipeline.py`** (44KB): Advanced ML pipeline including:
  - Text preprocessing and feature extraction
  - Multi-model ensemble for fact-checking
  - Confidence scoring algorithms
  - Model training and evaluation utilities

## 🛠️ Technical Implementation

### Machine Learning Stack
- **Scikit-learn**: Core ML framework for fact-checking models
- **TF-IDF Vectorization**: Text feature extraction and representation
- **Ensemble Classification**: Multiple models for improved accuracy
- **Graceful Fallbacks**: Mock responses when models are unavailable

### Steganography Technology
- **LSB Encoding**: Least Significant Bit technique for message hiding
- **PIL/Pillow**: Image processing and manipulation
- **Base64 Processing**: Direct browser-to-server image handling
- **Lossless Compression**: Maintains image quality while embedding data

### Security Integration
- **VirusTotal API**: Real-time malicious URL detection
- **Multi-engine Scanning**: Comprehensive threat assessment
- **Rate Limiting**: API abuse prevention
- **Input Validation**: Robust data sanitization

### Frontend Technologies
- **Vanilla JavaScript**: Lightweight, framework-free implementation
- **CSS3 Animations**: Smooth interactions and transitions
- **Responsive Design**: Mobile-first approach
- **Progressive Enhancement**: Graceful degradation for older browsers

## 🚦 Current Status

✅ **Fully Implemented**:
- Fact-checking with ML models
- Link verification and malicious URL detection
- Complete steganography encode/decode functionality
- Responsive web interface
- REST API endpoints
- Local development environment
- Google Cloud App Engine configuration

🔄 **Production Ready**:
- Optimized for Google Cloud Platform deployment
- App Engine configuration with auto-scaling
- Environment-based configuration for security
- Comprehensive error handling and logging
- Performance optimizations for cloud deployment
- HTTPS-ready configuration
- Clean project structure with deployment guides

## 🎯 Use Cases

1. **Journalists**: Verify claims and check source credibility
2. **Researchers**: Analyze information integrity and detect manipulation
3. **Security Teams**: Identify malicious links and threats
4. **Digital Forensics**: Extract hidden information from images
5. **Educators**: Teach information literacy and verification techniques
6. **Organizations**: Integrate verification capabilities into existing workflows

## 🔒 Privacy & Security

- No permanent storage of user data
- Secure API key management
- Input sanitization and validation
- Rate limiting and abuse prevention
- HTTPS-ready configuration
