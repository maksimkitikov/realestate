#!/usr/bin/env python3
"""
Simple test file for Render deployment
"""

import os
import sys

def test_imports():
    """Test basic imports"""
    try:
        import dash
        print("✅ Dash imported successfully")
    except ImportError as e:
        print(f"❌ Dash import failed: {e}")
        return False
    
    try:
        import pandas
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import plotly
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    return True

def test_app():
    """Test app creation"""
    try:
        from app import server
        print("✅ App server created successfully")
        return True
    except Exception as e:
        print(f"❌ App creation failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Render deployment...")
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    if test_imports() and test_app():
        print("🎉 All tests passed! Ready for Render deployment.")
    else:
        print("❌ Some tests failed.")
        sys.exit(1)
