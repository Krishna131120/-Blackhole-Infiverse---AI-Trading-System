#!/usr/bin/env python3
"""
Complete System Setup Script
Runs all necessary components in the correct order with better error handling
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def run_command(command, description, timeout=300):
    """Run a command and handle errors with timeout"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*60}")
    
    try:
        # Use Popen for better control
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        output_lines = []
        error_lines = []
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                output_lines.append(output.strip())
        
        # Get any remaining stderr
        stderr = process.stderr.read()
        if stderr:
            print(f"STDERR: {stderr}")
            error_lines.append(stderr)
        
        return_code = process.poll()
        
        if return_code == 0:
            print(f"SUCCESS: {description}")
            return True
        else:
            print(f"ERROR: {description} (Exit code: {return_code})")
            if error_lines:
                print("Error details:", "\n".join(error_lines[-5:]))  # Show last 5 errors
            return False
            
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {description} took too long")
        process.kill()
        return False
    except Exception as e:
        print(f"EXCEPTION: {description}")
        print(f"Error: {e}")
        return False

def check_file_exists(file_path, description):
    """Check if a file exists"""
    if Path(file_path).exists():
        print(f"EXISTS: {description}")
        return True
    else:
        print(f"MISSING: {description}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\n[0] Checking Python dependencies...")
    try:
        import pandas
        import numpy
        import lightgbm
        import requests
        import fastapi
        print("SUCCESS: All required packages are installed")
        return True
    except ImportError as e:
        print(f"ERROR: Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def main():
    print("BLACKHOLE INFIVERSE - COMPLETE SYSTEM SETUP")
    print("="*60)
    
    # Check dependencies first
    if not check_dependencies():
        print("\nSTOPPING: Please install dependencies first")
        return
    
    # Check prerequisites
    print("\n[1] Checking prerequisites...")
    prerequisites = [
        ("universe.txt", "Universe file"),
        ("core/data_ingest.py", "Data ingestion module"),
        ("core/enhanced_features.py", "Enhanced feature pipeline"),
        ("core/models/enhanced_lightgbm.py", "LightGBM model"),
        ("train_rl_agent.py", "RL training script"),
        ("api/server.py", "API server")
    ]
    
    all_prereqs_ok = True
    for file_path, description in prerequisites:
        if not check_file_exists(file_path, description):
            all_prereqs_ok = False
    
    if not all_prereqs_ok:
        print("\nMISSING: Prerequisites. Please check the files above.")
        return
    
    print("\nSUCCESS: All prerequisites found!")
    
    # Step 1: Fetch data (with longer timeout)
    print("\n[2] Starting data fetch (this may take 5-10 minutes)...")
    if not run_command("python fetch_more_data.py", "Fetch market data", timeout=600):
        print("\nERROR: Data fetching failed. You can try running it manually:")
        print("python fetch_more_data.py")
        print("\nContinuing with next steps...")
    
    # Step 2: Generate features
    print("\n[3] Generating enhanced features...")
    if not run_command("python update_enhanced_features.py", "Generate enhanced features", timeout=300):
        print("\nERROR: Feature generation failed. Stopping setup.")
        return
    
    # Step 3: Train LightGBM model
    print("\n[4] Training LightGBM model...")
    if not run_command("python core/models/enhanced_lightgbm.py", "Train LightGBM model", timeout=300):
        print("\nERROR: LightGBM training failed. Stopping setup.")
        return
    
    # Step 4: Train RL agent
    print("\n[5] Training RL agent...")
    if not run_command("python train_rl_agent.py", "Train RL agent", timeout=300):
        print("\nERROR: RL agent training failed. Stopping setup.")
        return
    
    # Step 5: Verify models
    print(f"\n{'='*60}")
    print("VERIFYING TRAINED MODELS")
    print(f"{'='*60}")
    
    model_files = [
        ("models/enhanced-lightgbm-v2.pkl", "LightGBM model"),
        ("models/linucb_agent.pkl", "LinUCB agent"),
        ("data/features/feature_store.parquet", "Feature store")
    ]
    
    all_models_ok = True
    for file_path, description in model_files:
        if not check_file_exists(file_path, description):
            all_models_ok = False
    
    if not all_models_ok:
        print("\nWARNING: Some models are missing. You may need to run training manually.")
        print("Check the training steps above for errors.")
    else:
        print("\nSUCCESS: All models trained successfully!")
    
    # Final instructions
    print(f"\n{'='*60}")
    print("SETUP COMPLETE!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Start the API server:")
    print("   python api/server.py")
    print("\n2. Test the API:")
    print("   - Open http://localhost:8000/docs")
    print("   - Get token: POST /auth/token")
    print("   - Test predictions: POST /prediction_agent/tools/predict")
    print("\n3. Run comprehensive test:")
    print("   python test_enhanced_analysis.py")
    print("\n4. Run LangGraph workflow:")
    print("   python langgraph_workflow.py")
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()