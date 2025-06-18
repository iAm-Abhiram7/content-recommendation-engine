#!/usr/bin/env python3
"""
Streamlit Demo Launcher
Properly configured launcher for the Streamlit demo application
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get project root
    project_root = Path(__file__).parent
    
    # Change to project directory
    os.chdir(project_root)
    
    # Set PYTHONPATH to include project root
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    # Launch Streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        'src/demo/streamlit_app.py',
        '--server.port', '8503',
        '--server.address', '0.0.0.0'
    ]
    
    print("🚀 Launching Streamlit Demo...")
    print("📍 URL: http://localhost:8503")
    print("🔧 Project Root:", project_root)
    
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    main()
