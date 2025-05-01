import os
import sys
import argparse
import subprocess
import platform
from setup_env import check_requirements, setup_bitnet_repo

def run_app(host="0.0.0.0", port=8000, create_dummy=False, dummy_size="125M"):
    """Run the FastAPI BitNet inference app."""
    # Check requirements
    if not check_requirements():
        print("ERROR: System requirements not met. Please install the required dependencies.")
        return False
    
    # Setup BitNet repo
    if not setup_bitnet_repo():
        print("ERROR: Failed to set up BitNet repository.")
        return False
    
    # Create dummy model if requested
    if create_dummy:
        print("Creating dummy model for testing...")
        dummy_dir = os.path.join("app", "models", "dummy")
        subprocess.run([
            sys.executable, "create_dummy_model.py",
            "--output-dir", dummy_dir,
            "--model-size", dummy_size
        ], check=True)
        
        print(f"Dummy model created in {dummy_dir}")
    
    # Run the FastAPI app
    print(f"Starting FastAPI BitNet inference app on http://{host}:{port}")
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "app:app", 
        "--host", host, 
        "--port", str(port),
        "--reload"
    ], check=True)

def main():
    parser = argparse.ArgumentParser(description='Run the FastAPI BitNet inference app')
    parser.add_argument('--host', default='0.0.0.0', help='Host address to bind to')
    parser.add_argument('--port', '-p', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--create-dummy', '-d', action='store_true', help='Create a dummy model for testing')
    parser.add_argument('--dummy-size', '-s', default='125M', choices=['125M', '350M', '1B', '3B'], help='Size of the dummy model')
    
    args = parser.parse_args()
    
    run_app(args.host, args.port, args.create_dummy, args.dummy_size)

if __name__ == "__main__":
    main() 