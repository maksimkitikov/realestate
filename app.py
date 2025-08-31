#!/usr/bin/env python3
"""
Render deployment entry point for Real Estate Dashboard
Developed by Maksim Kitikov - Upside Analytics
"""

import os
import sys

# Try to import dashboards in order of complexity
try:
    from dashboard_simple_render import app
    print("✅ Loaded simple render dashboard")
except ImportError as e:
    print(f"⚠️ Simple render dashboard failed to load: {e}")
    try:
        from dashboard_advanced import app
        print("✅ Loaded advanced dashboard")
    except ImportError as e2:
        print(f"⚠️ Advanced dashboard failed to load: {e2}")
        try:
            from dashboard import app
            print("✅ Loaded basic dashboard")
        except ImportError as e3:
            print(f"❌ All dashboards failed to load: {e3}")
            sys.exit(1)

# Export server for Gunicorn
server = app.server

# For Render deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    print(f"🚀 Starting dashboard on port {port}")
    app.run_server(debug=False, host='0.0.0.0', port=port)
