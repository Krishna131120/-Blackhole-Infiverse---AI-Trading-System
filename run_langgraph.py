#!/usr/bin/env python3
"""
LangGraph Workflow Runner
Automatically starts API server and runs LangGraph workflow
"""

import subprocess
import time
import sys
import os
import signal
import threading
from pathlib import Path

def start_api_server():
    """Start the API server in a separate process"""
    print("🚀 Starting API server...")
    try:
        process = subprocess.Popen([
            sys.executable, "api/server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        if process.poll() is None:
            print("✅ API server started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start API server:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return None
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return None

def run_langgraph():
    """Run the LangGraph workflow"""
    print("🔄 Running LangGraph workflow...")
    try:
        result = subprocess.run([
            sys.executable, "langgraph_workflow.py"
        ], capture_output=True, text=True)
        
        print("📊 LangGraph Output:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  LangGraph Errors:")
            print(result.stderr)
            
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running LangGraph: {e}")
        return False

def main():
    """Main execution function"""
    print("=" * 60)
    print("🤖 LangGraph Workflow Runner")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("api/server.py").exists():
        print("❌ Error: api/server.py not found")
        print("💡 Make sure you're in the project root directory")
        return 1
    
    if not Path("langgraph_workflow.py").exists():
        print("❌ Error: langgraph_workflow.py not found")
        print("💡 Make sure you're in the project root directory")
        return 1
    
    # Start API server
    server_process = start_api_server()
    if not server_process:
        print("❌ Cannot proceed without API server")
        return 1
    
    try:
        # Run LangGraph workflow
        success = run_langgraph()
        
        if success:
            print("✅ LangGraph workflow completed successfully")
        else:
            print("❌ LangGraph workflow failed")
            
    finally:
        # Clean up: stop the API server
        print("🛑 Stopping API server...")
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
            print("✅ API server stopped")
        except subprocess.TimeoutExpired:
            print("⚠️  Force killing API server...")
            server_process.kill()
        except Exception as e:
            print(f"⚠️  Error stopping server: {e}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
