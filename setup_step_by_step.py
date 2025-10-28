#!/usr/bin/env python3
"""
Simple Step-by-Step Setup Script
Runs each step individually with clear progress tracking
"""

import subprocess
import sys
import time
from pathlib import Path

def run_step(step_num, description, command):
    """Run a single step with clear progress tracking"""
    print(f"\n{'='*80}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*80}")
    print(f"Running: {command}")
    print("This may take several minutes...")
    print("="*80)
    
    try:
        # Run the command and show output in real-time
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Show output as it comes
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        return_code = process.poll()
        
        if return_code == 0:
            print(f"\n✅ STEP {step_num} COMPLETED: {description}")
            return True
        else:
            print(f"\n❌ STEP {step_num} FAILED: {description} (Exit code: {return_code})")
            return False
            
    except Exception as e:
        print(f"\n❌ STEP {step_num} ERROR: {description}")
        print(f"Error: {e}")
        return False

def main():
    print("🚀 BLACKHOLE INFIVERSE - STEP-BY-STEP SETUP")
    print("="*80)
    print("This script will run each step individually so you can see progress")
    print("="*80)
    
    # Check if we're in the right directory
    if not Path("universe.txt").exists():
        print("❌ ERROR: universe.txt not found. Please run this from the project root directory.")
        return
    
    steps = [
        (1, "Install Dependencies", "pip install -r requirements.txt"),
        (2, "Fetch Market Data", "python fetch_more_data.py"),
        (3, "Generate Enhanced Features", "python update_enhanced_features.py"),
        (4, "Train LightGBM Model", "python core/models/enhanced_lightgbm.py"),
        (5, "Train RL Agent", "python train_rl_agent.py")
    ]
    
    completed_steps = 0
    
    for step_num, description, command in steps:
        if run_step(step_num, description, command):
            completed_steps += 1
        else:
            print(f"\n⚠️  Step {step_num} failed. You can:")
            print(f"   1. Try running it manually: {command}")
            print(f"   2. Continue with remaining steps")
            print(f"   3. Stop and fix the issue")
            
            response = input("\nContinue with remaining steps? (y/n): ").lower()
            if response != 'y':
                print("Setup stopped. Please fix the issue and run again.")
                return
    
    print(f"\n{'='*80}")
    print(f"SETUP SUMMARY: {completed_steps}/{len(steps)} steps completed")
    print(f"{'='*80}")
    
    if completed_steps == len(steps):
        print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Start the API server:")
        print("   python api/server.py")
        print("\n2. Test the system:")
        print("   python test_enhanced_analysis.py")
    else:
        print("⚠️  Some steps failed. Please check the errors above.")
        print("You can run individual steps manually:")
        for step_num, description, command in steps:
            print(f"   {command}  # {description}")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
