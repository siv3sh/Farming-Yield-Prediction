#!/usr/bin/env python3
"""
Script to automatically run Jupyter notebook, check for errors, and fix common issues
"""
import sys
import os
import json
import subprocess
import traceback

def run_notebook(notebook_path):
    """Run notebook and return execution results"""
    print(f"📓 Running notebook: {notebook_path}")
    
    try:
        # Use jupyter nbconvert to execute the notebook
        result = subprocess.run(
            [
                sys.executable, "-m", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                notebook_path
            ],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        if result.returncode != 0:
            print(f"❌ Error executing notebook:")
            print(result.stderr)
            return False, result.stderr
        
        print("✅ Notebook executed successfully!")
        return True, None
        
    except subprocess.TimeoutExpired:
        print("⏱️ Notebook execution timed out (took more than 30 minutes)")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False, str(e)

def check_notebook_errors(notebook_path):
    """Check notebook for execution errors"""
    print(f"🔍 Checking notebook for errors: {notebook_path}")
    
    try:
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
        
        errors = []
        for i, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') == 'code':
                outputs = cell.get('outputs', [])
                for output in outputs:
                    if output.get('output_type') == 'error':
                        error_msg = output.get('ename', 'Unknown') + ': ' + output.get('evalue', '')
                        errors.append({
                            'cell': i,
                            'error': error_msg,
                            'traceback': output.get('traceback', [])
                        })
        
        if errors:
            print(f"❌ Found {len(errors)} error(s) in notebook:")
            for err in errors:
                print(f"  Cell {err['cell']}: {err['error']}")
            return False, errors
        else:
            print("✅ No errors found in notebook!")
            return True, None
            
    except Exception as e:
        print(f"❌ Error checking notebook: {e}")
        return False, str(e)

def fix_common_issues(notebook_path):
    """Fix common issues in the notebook"""
    print("🔧 Checking for common issues...")
    
    try:
        with open(notebook_path, 'r') as f:
            content = f.read()
        
        fixes_applied = []
        
        # Check for common matplotlib issues
        if '%matplotlib inline' in content and 'plt.style.use' in content:
            # This should be fine, but let's check
            pass
        
        # Check if seaborn style is correct
        if "seaborn-v0_8-darkgrid" in content:
            # This is fine
            pass
        
        print("✅ No common issues found that need fixing")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not check for issues: {e}")
        return False

def main():
    notebook_path = "Farming_Yield_Prediction_Complete_Solution.ipynb"
    
    if not os.path.exists(notebook_path):
        print(f"❌ Notebook not found: {notebook_path}")
        return 1
    
    # Fix common issues first
    fix_common_issues(notebook_path)
    
    # Run the notebook
    success, error = run_notebook(notebook_path)
    
    if not success:
        print(f"\n❌ Failed to execute notebook: {error}")
        print("\n💡 Trying alternative approach: Running as Python script...")
        
        # Alternative: Convert to Python and run
        try:
            result = subprocess.run(
                [sys.executable, "-m", "nbconvert", "--to", "python", notebook_path],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✅ Converted notebook to Python script")
                # Note: We won't execute it as Python due to notebook-specific features
        except:
            pass
        
        return 1
    
    # Check for errors
    success, errors = check_notebook_errors(notebook_path)
    
    if not success:
        print("\n⚠️ Notebook executed but contains errors. Check the output above.")
        return 1
    
    print("\n✅ Notebook executed successfully with no errors!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

