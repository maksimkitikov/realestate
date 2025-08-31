#!/usr/bin/env python3
"""
Render deployment entry point for Real Estate Dashboard
Developed by Maksim Kitikov - Upside Analytics
"""

import os
import sys
import traceback

def create_fallback_app():
    """Create a minimal working app as last resort"""
    import dash
    from dash import html
    
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("🏠 Real Estate Dashboard - Maksim Kitikov", style={'textAlign': 'center'}),
        html.P("Dashboard is starting up... Please refresh in a moment.", style={'textAlign': 'center'}),
        html.P("Developer: Maksim Kitikov - Upside Analytics", style={'textAlign': 'center'})
    ])
    return app

# Try to import dashboards in order of preference
app = None
print("🚀 Starting dashboard import process...")

try:
    print("📊 Attempting to load API-only dashboard...")
    from dashboard_api_only import app
    print("✅ Successfully loaded API-only dashboard (recommended for Render)")
except Exception as e:
    print(f"⚠️ API-only dashboard failed to load: {e}")
    print(f"Error details: {traceback.format_exc()}")
    
    try:
        print("📊 Attempting to load advanced dashboard...")
        from dashboard_advanced import app
        print("✅ Successfully loaded advanced dashboard (full version)")
    except Exception as e2:
        print(f"⚠️ Advanced dashboard failed to load: {e2}")
        
        try:
            print("📊 Attempting to load basic dashboard...")
            from dashboard import app
            print("✅ Successfully loaded basic dashboard")
        except Exception as e3:
            print(f"⚠️ Basic dashboard failed to load: {e3}")
            
            try:
                print("📊 Attempting to load simple render dashboard...")
                from dashboard_simple_render import app
                print("✅ Successfully loaded simple render dashboard (fallback)")
            except Exception as e4:
                print(f"⚠️ Simple render dashboard failed to load: {e4}")
                print("🔧 Creating minimal fallback app...")
                app = create_fallback_app()
                print("✅ Created minimal fallback app")

if app is None:
    print("❌ Critical error: No app created")
    sys.exit(1)

# Export server for Gunicorn
server = app.server
print("✅ Server exported successfully")

# For Render deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    print(f"🚀 Starting dashboard on port {port}")
    print(f"📍 Host: 0.0.0.0")
    app.run_server(debug=False, host='0.0.0.0', port=port)
