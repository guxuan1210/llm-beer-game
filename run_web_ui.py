#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Beer Game Web UI Launcher

Run this script directly to launch the web UI.
"""

import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Launch Streamlit Web UI"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Launch LLM Beer Game Web UI')
    parser.add_argument('--port', type=int, default=8501, help='Web server port (default: 8501)')
    args = parser.parse_args()

    try:
        # Check if streamlit is installed
        import streamlit

        # Launch streamlit app
        streamlit_script = project_root / "llm_beer_game" / "ui" / "streamlit_app.py"

        print("🍺 Launching LLM Beer Game Web UI...")
        print(f"📁 Script path: {streamlit_script}")
        print(f"🌐 Web UI will launch on port {args.port}")
        print("⏹️  Press Ctrl+C to stop")
        print("-" * 50)

        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(streamlit_script),
            "--server.address", "localhost",
            "--server.port", str(args.port),
            "--browser.gatherUsageStats", "false"
        ])

    except ImportError:
        print("❌ Error: streamlit not installed")
        print("📦 Run: pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Web UI stopped")
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()