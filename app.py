#!/usr/bin/env python3
"""
Render deployment entry point for Real Estate Dashboard
Developed by Maksim Kitikov - Upside Analytics
"""

import os
import sys
from dashboard_advanced import app, server

# For Render deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=False, host='0.0.0.0', port=port)
