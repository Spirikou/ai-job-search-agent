#!/usr/bin/env python3
"""
AI Job Search Agent Launcher
Simple script to launch the streamlined interface
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Main launcher function."""
    print("🤖 AI Job Search Agent - Streamlined Workflow")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("src/streamlit_ui.py").exists():
        print("❌ Error: streamlit_ui.py not found!")
        print("   Please run this script from the Job_search_agent directory")
        return
    
    print("🚀 Launching streamlined Streamlit interface...")
    print("\n📝 Workflow:")
    print("   1. Upload Resume (any PDF filename)")
    print("   2. Review & Modify Preferences")
    print("   3. Review & Modify Job Search Prompt")
    print("   4. Perform Job Search")
    print("   5. View Results")
    print("   6. Chat & Customize CVs")
    print("\n🎨 Features:")
    print("   - Modern night mode interface")
    print("   - Enter key to send chat messages")
    print("   - PDF download for customized CVs")
    print("   - Clean workflow progress tracking")
    print("   - Flexible resume upload (any filename)")
    print("\n🌐 The interface will open in your default web browser")
    print("   If it doesn't open automatically, go to: http://localhost:8501")
    print("\n" + "=" * 60)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/streamlit_ui.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Thanks for using AI Job Search Agent!")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        print("\n💡 Try running manually:")
        print("   streamlit run src/streamlit_ui.py")

if __name__ == "__main__":
    main()