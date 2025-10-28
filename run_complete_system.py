#!/usr/bin/env python3
"""
Complete System Setup and Run Script
Automates the entire process from data ingestion to API server startup
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def run_command(command, description, check=True):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return False
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_file_exists(file_path, description):
    """Check if a file exists"""
    if Path(file_path).exists():
        print(f"✅ {description} - EXISTS")
        return True
    else:
        print(f"❌ {description} - MISSING")
        return False

def main():
    print("\n" + "="*80)
    print("🚀 BLACKHOLE INFIVERSE - COMPLETE SYSTEM SETUP")
    print("="*80)
    
    # Step 1: Check prerequisites
    print("\n[1/8] Checking prerequisites...")
    if not check_file_exists("fetch_more_data.py", "Data fetching script"):
        print("❌ Missing fetch_more_data.py")
        return False
    
    if not check_file_exists("core/enhanced_features.py", "Enhanced feature pipeline"):
        print("❌ Missing core/enhanced_features.py")
        return False
    
    if not check_file_exists("core/models/enhanced_lightgbm.py", "Enhanced LightGBM model"):
        print("❌ Missing enhanced_lightgbm.py")
        return False
    
    if not check_file_exists("api/server.py", "API server"):
        print("❌ Missing api/server.py")
        return False
    
    print("✅ All prerequisites found!")
    
    # Step 2: Fetch market data
    print("\n[2/8] Fetching market data...")
    if not run_command("python fetch_more_data.py", "Data fetching"):
        print("❌ Data fetching failed. Stopping setup.")
        return False
    
    # Step 3: Generate features
    print("\n[3/8] Generating features...")
    if not run_command("python update_enhanced_features.py", "Enhanced feature generation"):
        print("❌ Feature generation failed. Stopping setup.")
        return False
    
    # Step 4: Train Enhanced LightGBM model
    print("\n[4/8] Training Enhanced LightGBM model...")
    if not run_command("python core/models/enhanced_lightgbm.py", "Enhanced LightGBM training"):
        print("❌ Model training failed. Stopping setup.")
        return False
    
    # Step 5: Train RL agent
    print("\n[5/8] Training RL agent...")
    if not run_command("python train_rl_agent.py", "RL agent training"):
        print("❌ RL agent training failed. Stopping setup.")
        return False
    
    # Step 6: Verify models are created
    print("\n[6/8] Verifying models...")
    required_models = [
        "models/enhanced-lightgbm-v2.pkl",
        "models/enhanced-lightgbm-v2_scaler.pkl", 
        "models/enhanced-lightgbm-v2_features.pkl",
        "models/enhanced-lightgbm-v2_metadata.json",
        "models/linucb_agent.pkl"
    ]
    
    all_models_exist = True
    for model in required_models:
        if not check_file_exists(model, f"Model: {model}"):
            all_models_exist = False
    
    if not all_models_exist:
        print("❌ Some models are missing. Stopping setup.")
        return False
    
    print("✅ All models created successfully!")
    
    # Step 7: Start API server
    print("\n[7/8] Starting API server...")
    print("🚀 Starting server in background...")
    print("📝 Server logs will be saved to logs/api_server.log")
    print("🌐 API will be available at: http://localhost:8000")
    print("📚 API docs will be available at: http://localhost:8000/docs")
    
    # Start server in background
    try:
        with open("logs/api_server.log", "w") as log_file:
            process = subprocess.Popen(
                ["python", "api/server.py"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        print(f"✅ API server started with PID: {process.pid}")
        
        # Wait a bit for server to start
        print("⏳ Waiting for server to initialize...")
        time.sleep(10)
        
        # Check if server is running
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ API server is running and healthy!")
            else:
                print("⚠️ API server started but health check failed")
        except Exception as e:
            print(f"⚠️ Could not verify server health: {e}")
        
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return False
    
    # Step 8: Run demo test
    print("\n[8/8] Running demo test...")
    print("🧪 Testing complete system...")
    
    if not run_command("python demo_system.py", "Demo system test", check=False):
        print("⚠️ Demo test had issues, but server is running")
    
    # Final status
    print("\n" + "="*80)
    print("🎉 SETUP COMPLETE!")
    print("="*80)
    print("✅ Data fetched and processed")
    print("✅ Features generated")
    print("✅ Enhanced LightGBM model trained")
    print("✅ RL agent trained")
    print("✅ API server started")
    print("✅ Demo test completed")
    print("")
    print("🌐 API Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("📊 Dashboard: http://localhost:8000/data/dashboard")
    print("📈 Live Feed: http://localhost:8000/feed/live")
    print("")
    print("🔧 To stop the server: Press Ctrl+C in the terminal")
    print("📝 Server logs: logs/api_server.log")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!")
        print("🚀 Your system is ready for integration!")
