"""
clean_workspace.py
------------------
Deletes the files and directories from the previous ODE-solving project
to prepare a clean workspace for the ML Integration Selector project.
"""

import os
import shutil

FILES_TO_DELETE = [
    "rk4.py",
    "pinn.py",
    "problems.py",
    "run_experiments.py",
    "convergence_study.py",
    "report.md",
    "results.json",
    "convergence_results.txt"
]

DIRS_TO_DELETE = [
    "figures",
    "__pycache__"
]

def clean():
    print("Cleaning workspace...")
    cwd = os.getcwd()
    
    # Check if we are in the correct workspace
    if not cwd.endswith("bsit400"):
        print(f"Warning: Current working directory is {cwd}, not bsit400.")
    
    for f in FILES_TO_DELETE:
        path = os.path.join(cwd, f)
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted file: {f}")
            
    for d in DIRS_TO_DELETE:
        path = os.path.join(cwd, d)
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Deleted directory: {d}")
            
    print("Workspace cleaned successfully.\n")

if __name__ == "__main__":
    clean()
