#!/usr/bin/env python3
"""
Simple startup test to verify the app can be imported and initialized properly.
This helps identify issues before deployment to Cloud Run.
"""

import sys
import os

def test_app_import():
    """Test that the app can be imported without errors"""
    try:
        print("🧪 Testing app import...")
        
        # Set environment variables for testing
        os.environ['FLASK_ENV'] = 'production'
        os.environ['PORT'] = '5000'
        
        # Try to import the app
        from app import app, application
        
        print("✅ App imported successfully")
        print(f"📱 App name: {app.name}")
        print(f"🔧 Debug mode: {app.debug}")
        print(f"🎯 App config keys: {list(app.config.keys())}")
        
        # Test that application variable exists (needed for Gunicorn)
        if application:
            print("✅ Application variable defined for WSGI")
        else:
            print("❌ Application variable not defined")
            return False
        
        # Test health endpoint
        print("🏥 Testing health endpoint...")
        with app.test_client() as client:
            response = client.get('/health')
            if response.status_code == 200:
                print("✅ Health endpoint working")
                print(f"📊 Health response: {response.get_json()}")
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                return False
        
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_app_import()
    sys.exit(0 if success else 1)