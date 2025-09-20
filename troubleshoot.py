#!/usr/bin/env python3
"""
TruthGuard Troubleshooting Script
Diagnoses and fixes common connectivity issues
"""

import requests
import sys
import json
import time
import subprocess
import socket
from urllib.parse import urlparse

def print_banner():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                🔍 TruthGuard Troubleshooter 🔍               ║
    ║                                                              ║
    ║    Diagnoses and fixes common connectivity issues            ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def check_port_availability(port=5000):
    """Check if port is available or already in use"""
    print(f"🔌 Checking port {port} availability...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        
        if result == 0:
            print(f"✅ Port {port} is in use (likely by Flask server)")
            return True
        else:
            print(f"❌ Port {port} is not in use")
            return False
    except Exception as e:
        print(f"❌ Error checking port: {e}")
        return False

def test_health_endpoint():
    """Test the Flask health endpoint"""
    print("🏥 Testing health endpoint...")
    
    urls_to_try = [
        'http://localhost:5000/health',
        'http://127.0.0.1:5000/health',
        'http://0.0.0.0:5000/health'
    ]
    
    for url in urls_to_try:
        try:
            print(f"   Trying: {url}")
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Health check passed!")
                print(f"   Status: {result.get('status')}")
                print(f"   Timestamp: {result.get('timestamp')}")
                return True
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection refused - server not running")
        except requests.exceptions.Timeout:
            print(f"❌ Request timed out")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return False

def test_api_endpoints():
    """Test main API endpoints"""
    print("🔗 Testing API endpoints...")
    
    endpoints = [
        ('/api/fact-check', 'POST', {'claim': 'Test claim'}),
        ('/api/verify-link', 'POST', {'url': 'https://example.com'}),
        ('/health', 'GET', None)
    ]
    
    base_url = 'http://localhost:5000'
    working_endpoints = []
    
    for endpoint, method, data in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            print(f"   Testing: {method} {url}")
            
            if method == 'GET':
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, json=data, timeout=10)
            
            if response.status_code in [200, 400, 500]:  # Any response is good
                print(f"✅ {endpoint} responding (HTTP {response.status_code})")
                working_endpoints.append(endpoint)
            else:
                print(f"❌ {endpoint} unexpected response (HTTP {response.status_code})")
                
        except Exception as e:
            print(f"❌ {endpoint} failed: {e}")
    
    return working_endpoints

def check_dependencies():
    """Check if required Python packages are installed"""
    print("📦 Checking Python dependencies...")
    
    required_packages = [
        'flask',
        'flask_cors',
        'requests',
        'nltk',
        'numpy',
        'PIL',
        'PyPDF2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n🚨 Missing packages detected!")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_firewall_and_antivirus():
    """Check for common firewall/antivirus issues"""
    print("🛡️ Firewall and Security Check...")
    
    # Test if we can make outbound connections
    try:
        response = requests.get('https://httpbin.org/get', timeout=5)
        if response.status_code == 200:
            print("✅ Outbound internet connection working")
        else:
            print("❌ Outbound connection issues detected")
    except:
        print("❌ Outbound internet connection blocked")
    
    # Test localhost connectivity
    try:
        response = requests.get('http://localhost:5000', timeout=2)
        print("✅ Localhost connections allowed")
    except requests.exceptions.ConnectionError:
        print("⚠️ Cannot connect to localhost:5000 (expected if server not running)")
    except Exception as e:
        print(f"❌ Localhost connection issue: {e}")

def provide_solutions(issues_found):
    """Provide solutions based on detected issues"""
    print("\n" + "="*60)
    print("🔧 SOLUTIONS & NEXT STEPS")
    print("="*60)
    
    if not issues_found['port_available']:
        print("""
🚀 START THE FLASK SERVER:
   1. Open terminal/command prompt
   2. Navigate to your project directory:
      cd "c:\\Users\\dibya\\OneDrive\\Desktop\\Main is good\\Main\\modified Fake news\\Fake news"
   3. Run the server:
      python app.py
   4. Look for this message:
      "* Running on http://127.0.0.1:5000"
        """)
    
    if not issues_found['dependencies_ok']:
        print("""
📦 INSTALL MISSING DEPENDENCIES:
   Run this command:
   pip install flask flask-cors requests nltk numpy Pillow PyPDF2
        """)
    
    if not issues_found['health_ok']:
        print("""
🔄 IF SERVER IS RUNNING BUT HEALTH CHECK FAILS:
   1. Check if another service is using port 5000:
      netstat -an | findstr :5000
   2. Try a different port:
      set PORT=8080 && python app.py
   3. Check Windows Firewall settings
   4. Try running as administrator
        """)
    
    print("""
🌐 BROWSER TROUBLESHOOTING:
   1. Clear browser cache and cookies
   2. Try accessing: http://localhost:5000
   3. Check browser console for JavaScript errors (F12)
   4. Try a different browser
   5. Disable browser extensions temporarily
    """)
    
    print("""
🔍 ADVANCED TROUBLESHOOTING:
   1. Check app.py console for error messages
   2. Verify all files are in the correct directory
   3. Test with a simple claim: "Water boils at 100 degrees"
   4. Check network tab in browser dev tools (F12)
    """)

def main():
    print_banner()
    
    issues_found = {
        'port_available': False,
        'health_ok': False,
        'api_ok': False,
        'dependencies_ok': False
    }
    
    # Run diagnostics
    issues_found['dependencies_ok'] = check_dependencies()
    issues_found['port_available'] = check_port_availability()
    issues_found['health_ok'] = test_health_endpoint()
    
    if issues_found['health_ok']:
        working_endpoints = test_api_endpoints()
        issues_found['api_ok'] = len(working_endpoints) > 0
    
    check_firewall_and_antivirus()
    
    # Summary
    print("\n" + "="*60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*60)
    
    status_icon = lambda x: "✅" if x else "❌"
    print(f"{status_icon(issues_found['dependencies_ok'])} Python Dependencies")
    print(f"{status_icon(issues_found['port_available'])} Port 5000 Available")
    print(f"{status_icon(issues_found['health_ok'])} Health Endpoint")
    print(f"{status_icon(issues_found['api_ok'])} API Endpoints")
    
    if all(issues_found.values()):
        print(f"\n🎉 ALL SYSTEMS GO! Your TruthGuard server is working correctly!")
        print(f"   Access your application at: http://localhost:5000")
    else:
        provide_solutions(issues_found)

if __name__ == "__main__":
    main()