#!/usr/bin/env python3
"""
Quick startup verification for TruthGuard before Cloud Run deployment
This script tests if the application can start and respond to health checks
"""

import os
import sys
import time
import subprocess
import requests
import signal
from pathlib import Path

def test_app_startup():
    """Test if the app starts correctly with Gunicorn"""
    print("🧪 Testing TruthGuard startup with Gunicorn...")
    
    # Set environment variables for testing
    env = os.environ.copy()
    env.update({
        'PORT': '8080',
        'FLASK_ENV': 'production',
        'FLASK_DEBUG': 'false'
    })
    
    # Start the app with Gunicorn (same as Cloud Run)
    print("🚀 Starting Gunicorn server...")
    process = subprocess.Popen([
        'gunicorn',
        '--bind', '0.0.0.0:8080',
        '--workers', '1',
        '--worker-class', 'gevent',
        '--timeout', '60',
        '--access-logfile', '-',
        '--error-logfile', '-',
        '--log-level', 'info',
        'app:app'
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Wait for startup
    print("⏳ Waiting for server to start...")
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get('http://localhost:8080/health', timeout=5)
            if response.status_code == 200:
                print(f"✅ Server started successfully! Health check: {response.status_code}")
                print(f"📊 Response: {response.json()}")
                
                # Test main page
                try:
                    main_response = requests.get('http://localhost:8080/', timeout=5)
                    print(f"✅ Main page accessible: {main_response.status_code}")
                except Exception as e:
                    print(f"⚠️ Main page test failed: {e}")
                
                # Cleanup
                process.terminate()
                process.wait(timeout=10)
                print("✅ Test completed successfully!")
                return True
                
        except requests.exceptions.ConnectionError:
            time.sleep(1)
            continue
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            break
    
    # If we get here, startup failed
    print("❌ Server failed to start within 30 seconds")
    
    # Get process output for debugging
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=10)
    
    stdout, stderr = process.communicate()
    if stdout:
        print(f"📋 Process output:\n{stdout}")
    
    return False

def test_import():
    """Test if the app can be imported without errors"""
    print("📦 Testing application import...")
    try:
        import app
        print("✅ Application imported successfully")
        print(f"📊 Flask app: {app.app}")
        print(f"🤖 ML Available: {app.ML_AVAILABLE}")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all startup tests"""
    print("🔍 TruthGuard Startup Verification")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('app.py').exists():
        print("❌ app.py not found. Please run this script from the project directory.")
        return False
    
    # Test 1: Import test
    if not test_import():
        print("❌ Import test failed. Fix import errors before deploying.")
        return False
    
    print("\n" + "=" * 50)
    
    # Test 2: Startup test
    if not test_app_startup():
        print("❌ Startup test failed. Fix startup issues before deploying.")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! Ready for Cloud Run deployment.")
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)