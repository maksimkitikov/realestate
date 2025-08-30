#!/usr/bin/env python3
"""
US Real Estate Analytics Dashboard - Production Ready for Render
"""

# Import the main dashboard application
from dashboard_advanced import app

# Export server for Gunicorn
server = app.server

if __name__ == '__main__':
    import os
    print("🚀 Starting US Real Estate Analytics Dashboard...")
    print("🌐 Open http://localhost:8050 in your browser")
    print("🗺️ Features: Interactive US States Map, Real-time Analytics")
    print("📊 Production-ready analytics platform")
    app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))
