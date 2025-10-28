#!/usr/bin/env python3
"""
Safe Project Cleanup Script
Removes unused, duplicate, and unnecessary files while preserving core functionality
"""

import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_remove_file(file_path):
    """Safely remove a file with logging"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"✅ Removed: {file_path}")
            return True
        else:
            logger.warning(f"⚠️  File not found: {file_path}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to remove {file_path}: {e}")
        return False

def safe_remove_directory(dir_path):
    """Safely remove a directory with logging"""
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            logger.info(f"✅ Removed directory: {dir_path}")
            return True
        else:
            logger.warning(f"⚠️  Directory not found: {dir_path}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to remove directory {dir_path}: {e}")
        return False

def main():
    print("🧹 BLACKHOLE INFIVERSE - PROJECT CLEANUP")
    print("="*60)
    
    # Files to remove (safe to delete)
    files_to_remove = [
        # Duplicate data fetching scripts
        "core/fetch_more_data.py",
        "fetch_more_data_fixed.py", 
        "fetch_all_markets.py",
        "fetch_indian_markets.py",
        
        # Unused validation scripts
        "validate_and_replace_symbols.py",
        "validate_indian_symbols.py",
        "universe_clean.txt",
        
        # Unused pipeline scripts
        "run_complete_data_pipeline.py",
        "sync_data_directories.py",
        "fix_predict_endpoint.py",
        
        # Unused test/utility scripts
        "test_api_server.py",
        "install_ta_lib.py",
        "run_tests.bat",
        
        # Redundant documentation
        "DOCUMENTATION.md",
        "EXECUTION_ORDER_INDIAN_MARKETS.md",
        "FINAL_EXECUTION_ORDER.md",
        "IMPLEMENTATION_SUMMARY.md",
        "PRODUCTION_CHECKLIST.md",
        "PRODUCTION_HANDOVER.md",
        "FEEDBACK_API_FIX.md",
        
        # Unused core files
        "core/feedback_loop.py",
        "core/incremental_rl.py",
        "core/mcp_tools.json",
        
        # Unused batch files
        "run_langgraph.bat",
        "start_api.bat",
        "start_langgraph.bat",
        
        # Unused data files
        "feedback_memory.json",
        "n8n_flow.json",
        
        # Old model files
        "models/lightgbm-v1_feature_importance.csv",
        "models/lightgbm-v1_metrics.json",
        "models/lightgbm-v1.pkl"
    ]
    
    # Counters
    removed_count = 0
    failed_count = 0
    not_found_count = 0
    
    print(f"\n[1] Removing {len(files_to_remove)} unnecessary files...")
    print("="*60)
    
    for file_path in files_to_remove:
        if safe_remove_file(file_path):
            removed_count += 1
        else:
            if not os.path.exists(file_path):
                not_found_count += 1
            else:
                failed_count += 1
    
    # Clean up duplicate virtual environment
    print(f"\n[2] Cleaning up duplicate virtual environment...")
    print("="*60)
    
    # Check if both venv and .venv exist
    if os.path.exists("venv") and os.path.exists(".venv"):
        print("Found both 'venv' and '.venv' directories")
        print("Removing unused 'venv' directory (keeping active '.venv')")
        if safe_remove_directory("venv"):
            removed_count += 1
    
    # Clean up __pycache__ directories
    print(f"\n[3] Cleaning up __pycache__ directories...")
    print("="*60)
    
    pycache_dirs = []
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_dirs.append(os.path.join(root, dir_name))
    
    for pycache_dir in pycache_dirs:
        if safe_remove_directory(pycache_dir):
            removed_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("🎉 CLEANUP COMPLETE!")
    print(f"{'='*60}")
    print(f"✅ Files removed: {removed_count}")
    print(f"⚠️  Files not found: {not_found_count}")
    print(f"❌ Failed removals: {failed_count}")
    
    # Verification
    print(f"\n{'='*60}")
    print("🔍 VERIFICATION - CORE FILES CHECK")
    print(f"{'='*60}")
    
    core_files = [
        "api/server.py",
        "core/mcp_adapter.py", 
        "core/data_ingest.py",
        "core/features.py",
        "core/models/enhanced_lightgbm.py",
        "core/models/rl_agent.py",
        "langgraph_workflow.py",
        "train_rl_agent.py",
        "fetch_more_data.py",
        "universe.txt",
        "requirements.txt",
        "README.md"
    ]
    
    all_core_files_exist = True
    for core_file in core_files:
        if os.path.exists(core_file):
            print(f"✅ {core_file}")
        else:
            print(f"❌ {core_file} - MISSING!")
            all_core_files_exist = False
    
    if all_core_files_exist:
        print(f"\n🎉 All core files present! System should work normally.")
        print(f"\nNext steps:")
        print(f"1. Run: python setup_complete_system.py")
        print(f"2. Run: python api/server.py")
        print(f"3. Run: python langgraph_workflow.py")
    else:
        print(f"\n⚠️  Some core files are missing. Please check the system.")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
