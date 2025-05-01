import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

def create_dummy_bitnet_model(output_dir, model_size="125M", outtype="i2_s"):
    """Create a dummy BitNet model for testing."""
    print(f"Creating dummy BitNet {model_size} model with type {outtype}...")
    
    # Check if BitNet repo exists
    if not os.path.exists('BitNet'):
        print("BitNet repository not found. Cloning...")
        try:
            subprocess.run(['git', 'clone', '--recursive', 'https://github.com/microsoft/BitNet.git'], check=True)
        except Exception as e:
            print(f"ERROR: Failed to clone BitNet repository: {e}")
            return False
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Change to BitNet directory
        os.chdir('BitNet')
        
        # Generate the dummy model
        outfile = os.path.join(output_dir, f"dummy-bitnet-{model_size}.{outtype}.gguf")
        cmd = [
            'python', 'utils/generate-dummy-bitnet-model.py', 
            'models/bitnet_b1_58-large',
            '--outfile', outfile,
            '--outtype', outtype,
            '--model-size', model_size
        ]
        
        subprocess.run(cmd, check=True)
        
        # Return to original directory
        os.chdir('..')
        
        print(f"Dummy model created at {outfile}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to create dummy model: {e}")
        # Return to original directory
        try:
            os.chdir('..')
        except:
            pass
        return False

def main():
    parser = argparse.ArgumentParser(description='Create a dummy BitNet model for testing')
    parser.add_argument('--output-dir', '-o', default='app/models/dummy', help='Directory to save the dummy model')
    parser.add_argument('--model-size', '-s', default='125M', choices=['125M', '350M', '1B', '3B'], help='Size of the dummy model')
    parser.add_argument('--outtype', '-t', default='i2_s', choices=['i2_s', 'tl1'], help='Quantization type for the model')
    
    args = parser.parse_args()
    
    create_dummy_bitnet_model(args.output_dir, args.model_size, args.outtype)

if __name__ == "__main__":
    main() 