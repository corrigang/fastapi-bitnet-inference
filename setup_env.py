import os
import sys
import argparse
import subprocess
import platform
import shutil
from pathlib import Path

def check_requirements():
    """Check if the system meets the requirements for running BitNet inference."""
    print("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
        print(f"ERROR: Python 3.9+ required. You have Python {python_version.major}.{python_version.minor}")
        return False
    
    # Check for CMake
    try:
        cmake_proc = subprocess.run(['cmake', '--version'], capture_output=True, text=True)
        cmake_version = cmake_proc.stdout.split('\n')[0].split(' ')[2]
        cmake_major, cmake_minor = map(int, cmake_version.split('.')[:2])
        
        if cmake_major < 3 or (cmake_major == 3 and cmake_minor < 22):
            print(f"ERROR: CMake 3.22+ required. You have CMake {cmake_version}")
            return False
    except Exception:
        print("ERROR: CMake 3.22+ is required but not found")
        return False
    
    # Check for clang/MSVC on Windows
    if platform.system() == 'Windows':
        try:
            clang_proc = subprocess.run(['clang', '--version'], capture_output=True, text=True)
            if 'not recognized' in clang_proc.stderr:
                try:
                    msvc_proc = subprocess.run(['cl'], capture_output=True, text=True)
                    if 'Microsoft' not in msvc_proc.stderr:
                        print("ERROR: Neither Clang nor MSVC compiler found")
                        return False
                except Exception:
                    print("ERROR: Neither Clang nor MSVC compiler found")
                    return False
        except Exception:
            try:
                msvc_proc = subprocess.run(['cl'], capture_output=True, text=True)
                if 'Microsoft' not in msvc_proc.stderr:
                    print("ERROR: Neither Clang nor MSVC compiler found")
                    return False
            except Exception:
                print("ERROR: Neither Clang nor MSVC compiler found")
                return False
    
    print("System requirements check passed!")
    return True

def setup_bitnet_repo():
    """Clone the BitNet repository if not already present."""
    if not os.path.exists('BitNet'):
        print("Cloning BitNet repository...")
        try:
            subprocess.run(['git', 'clone', '--recursive', 'https://github.com/microsoft/BitNet.git'], check=True)
            print("BitNet repository cloned successfully")
            return True
        except Exception as e:
            print(f"ERROR: Failed to clone BitNet repository: {e}")
            return False
    else:
        print("BitNet repository already exists")
        return True

def setup_environment(model_dir, quant_type="i2_s", quant_embd=False, use_pretuned=False):
    """Setup the environment for running BitNet inference."""
    if not check_requirements():
        return False
    
    if not setup_bitnet_repo():
        return False
    
    # Create model directory if it doesn't exist
    model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if bitnet.cpp is installed
    try:
        # Change to BitNet directory
        os.chdir('BitNet')
        
        # Build the project
        build_cmd = [
            sys.executable, 'setup_env.py',
            '--model-dir', model_dir,
            '--quant-type', quant_type
        ]
        
        if quant_embd:
            build_cmd.append('--quant-embd')
            
        if use_pretuned:
            build_cmd.append('--use-pretuned')
            
        subprocess.run(build_cmd, check=True)
        
        # Copy necessary scripts to main directory
        os.chdir('..')
        shutil.copy('BitNet/run_inference.py', '.')
        
        print("Environment setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to setup environment: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Setup the environment for running BitNet inference')
    parser.add_argument('--hf-repo', '-hr', choices=[
        '1bitLLM/bitnet_b1_58-large',
        '1bitLLM/bitnet_b1_58-3B',
        'HF1BitLLM/Llama3-8B-1.58-100B-tokens',
        'tiiuae/Falcon3-1B-Instruct-1.58bit',
        'tiiuae/Falcon3-3B-Instruct-1.58bit',
        'tiiuae/Falcon3-7B-Instruct-1.58bit',
        'tiiuae/Falcon3-10B-Instruct-1.58bit'
    ], help='Model used for inference')
    parser.add_argument('--model-dir', '-md', help='Directory to save/load the model')
    parser.add_argument('--log-dir', '-ld', help='Directory to save the logging info')
    parser.add_argument('--quant-type', '-q', choices=['i2_s', 'tl1'], help='Quantization type')
    parser.add_argument('--quant-embd', action='store_true', help='Quantize the embeddings to f16')
    parser.add_argument('--use-pretuned', '-p', action='store_true', help='Use the pretuned kernel parameters')
    
    args = parser.parse_args()
    
    if not args.model_dir and not args.hf_repo:
        parser.error("Either --model-dir or --hf-repo must be specified")
    
    model_dir = args.model_dir
    if not model_dir and args.hf_repo:
        model_dir = os.path.join("app/models", args.hf_repo.split('/')[-1])
    
    quant_type = args.quant_type or "i2_s"
    
    setup_environment(model_dir, quant_type, args.quant_embd, args.use_pretuned)

if __name__ == "__main__":
    main() 