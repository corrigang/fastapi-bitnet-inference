import os
import sys
import subprocess
import tempfile
import uvicorn
from fastapi import FastAPI, Request, Form, UploadFile, File, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from pathlib import Path
import shutil
import threading
import time
import glob

# Create FastAPI app
app = FastAPI(title="BitNet Inference App")

# Set up templates and static files
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Default model directory
MODEL_DIR = "app/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Track if a model is loaded
model_loaded = False
current_model = None

# Track download progress
download_progress = {
    "is_downloading": False,
    "model_name": "",
    "progress": 0,
    "status": "",
    "error": None
}

# Function to properly set the model status
def set_model_loaded(model_dir=None):
    global model_loaded, current_model
    
    if model_dir is None:
        model_loaded = False
        current_model = None
        return
    
    # Check if the directory exists
    if not os.path.exists(model_dir):
        model_loaded = False
        current_model = None
        return
    
    # Check if there are any files in the directory
    files = os.listdir(model_dir)
    if not files:
        model_loaded = False
        current_model = None
        return
    
    # The model directory exists and has files, set as loaded
    model_loaded = True
    current_model = model_dir

# Function to download a model from Hugging Face
def download_model(model_name, background_tasks=None):
    global download_progress
    
    if "/" not in model_name:
        model_name = f"microsoft/{model_name}"
    
    # Reset progress tracking
    download_progress["is_downloading"] = True
    download_progress["model_name"] = model_name
    download_progress["progress"] = 0
    download_progress["status"] = "Starting download..."
    download_progress["error"] = None
    
    output_dir = os.path.join(MODEL_DIR, model_name.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Update status
        download_progress["status"] = "Downloading model files..."
        download_progress["progress"] = 10
        
        # Download model - we can't track actual progress with snapshot_download
        # but we'll simulate progress for better UX
        cmd = [
            "python", "-c",
            f"from huggingface_hub import snapshot_download; snapshot_download(repo_id='{model_name}', local_dir='{output_dir}')"
        ]
        subprocess.run(cmd, check=True)
        
        # Update progress
        download_progress["progress"] = 70
        download_progress["status"] = "Setting up environment..."
        
        # Try to setup environment for model if possible
        try:
            # Setup environment for model
            setup_cmd = [
                "python", "setup_env.py", 
                "--model-dir", output_dir,
                "--quant-type", "i2_s"
            ]
            subprocess.run(setup_cmd, check=True)
        except Exception as e:
            # Log the error but continue, we'll try to use the fallback method
            print(f"Warning: Setup environment failed, will try fallback model loading: {str(e)}")
        
        # Finished successfully
        download_progress["progress"] = 100
        download_progress["status"] = "Download completed"
        download_progress["is_downloading"] = False
        
        # Set the model as loaded
        set_model_loaded(output_dir)
        
        return {"status": "success", "message": f"Model {model_name} downloaded successfully"}
    except Exception as e:
        # Update error status
        download_progress["error"] = str(e)
        download_progress["status"] = f"Error: {str(e)}"
        download_progress["is_downloading"] = False
        
        # Make sure model is not marked as loaded
        set_model_loaded(None)
        
        return {"status": "error", "message": f"Error downloading model: {str(e)}"}

# Function to run inference
def run_inference(prompt, conversation=True, n_predict=128, temperature=0.7):
    global model_loaded, current_model
    
    # Double check model status
    if current_model and os.path.exists(current_model):
        # Refresh model loaded status
        set_model_loaded(current_model)
    
    if not model_loaded or not current_model:
        return {"status": "error", "message": "No model loaded"}
    
    try:
        # Find the model file
        model_files = [f for f in os.listdir(current_model) if f.endswith(".gguf")]
        
        if not model_files:
            # Check if there are any files in the directory
            all_files = os.listdir(current_model)
            
            if not all_files:
                set_model_loaded(None)
                return {"status": "error", "message": "No files found in the model directory. Download may have failed."}
            
            # Check for model files that could be used with the fallback server
            safetensors_files = glob.glob(os.path.join(current_model, "*.safetensors"))
            bin_files = glob.glob(os.path.join(current_model, "*.bin"))
            pt_files = glob.glob(os.path.join(current_model, "*.pt"))
            config_file = os.path.exists(os.path.join(current_model, "config.json"))
            
            # If we find these files, we can try to use the standalone model server
            if (safetensors_files or bin_files or pt_files) and config_file:
                # Use the fallback model server
                try:
                    # Add system prompt for conversation mode
                    if conversation:
                        full_prompt = f"System: {prompt}\nAssistant: "
                    else:
                        full_prompt = prompt
                    
                    cmd = [
                        "python", "simple_model_server.py",
                        "--model", current_model,
                        "--prompt", full_prompt,
                        "--max-tokens", str(n_predict),
                        "--temperature", str(temperature)
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    
                    # Extract the generated text part
                    output_lines = result.stdout.strip().split("Generated Text:")
                    if len(output_lines) > 1:
                        output = output_lines[1].strip().replace("--------------", "").strip()
                    else:
                        output = result.stdout
                    
                    return {"status": "success", "output": output}
                except Exception as e:
                    return {
                        "status": "error", 
                        "message": f"Error running fallback model server: {str(e)}. Please install transformers: pip install transformers"
                    }
            else:
                # Try to use any model file we can find
                try:
                    # Look for any model file we might be able to use directly
                    config_file = os.path.join(current_model, "config.json")
                    if os.path.exists(config_file):
                        # Try to directly use the HuggingFace model
                        if conversation:
                            full_prompt = f"System: {prompt}\nAssistant: "
                        else:
                            full_prompt = prompt
                        
                        cmd = [
                            "python", "simple_model_server.py",
                            "--model", current_model,
                            "--prompt", full_prompt,
                            "--max-tokens", str(n_predict),
                            "--temperature", str(temperature)
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        
                        # Extract the generated text part
                        output_lines = result.stdout.strip().split("Generated Text:")
                        if len(output_lines) > 1:
                            output = output_lines[1].strip().replace("--------------", "").strip()
                        else:
                            output = result.stdout
                        
                        return {"status": "success", "output": output}
                except Exception as e:
                    pass
                    
                return {
                    "status": "error", 
                    "message": "No GGUF model file found and no alternative model format detected. The model may not be compatible."
                }
        
        # Standard GGUF inference
        model_file = os.path.join(current_model, model_files[0])
        
        # Run inference with BitNet's run_inference.py
        cmd = [
            "python", "run_inference.py",
            "-m", model_file,
            "-p", prompt,
            "-n", str(n_predict),
            "-temp", str(temperature)
        ]
        
        if conversation:
            cmd.append("-cnv")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        # If we get an error about run_inference.py not found, use the fallback
        if "No such file or directory" in str(e) and "run_inference.py" in str(e):
            try:
                # Try the fallback method
                if conversation:
                    full_prompt = f"System: {prompt}\nAssistant: "
                else:
                    full_prompt = prompt
                
                cmd = [
                    "python", "simple_model_server.py",
                    "--model", current_model,
                    "--prompt", full_prompt,
                    "--max-tokens", str(n_predict),
                    "--temperature", str(temperature)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Extract the generated text part
                output_lines = result.stdout.strip().split("Generated Text:")
                if len(output_lines) > 1:
                    output = output_lines[1].strip().replace("--------------", "").strip()
                else:
                    output = result.stdout
                
                return {"status": "success", "output": output}
            except Exception as fallback_e:
                return {"status": "error", "message": f"Error during fallback inference: {str(fallback_e)}"}
        
        return {"status": "error", "message": f"Error during inference process: {e.stderr}"}
    except Exception as e:
        return {"status": "error", "message": f"Error during inference: {str(e)}"}

# Function to download model in background
def download_model_background(model_name):
    thread = threading.Thread(target=download_model, args=(model_name,))
    thread.daemon = True
    thread.start()
    return {"status": "success", "message": f"Started downloading model {model_name} in background"}

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/download-model")
async def download_model_endpoint(model_name: str = Form(...), background: bool = Form(True)):
    if background:
        return download_model_background(model_name)
    else:
        return download_model(model_name)

@app.get("/download-progress")
async def get_download_progress():
    return JSONResponse(content=download_progress)

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    if not file.filename.endswith(".gguf"):
        return JSONResponse(content={"status": "error", "message": "Only .gguf files are supported"})
    
    # Create a unique directory for the uploaded model
    model_name = file.filename.replace(".gguf", "")
    output_dir = os.path.join(MODEL_DIR, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the uploaded file
    file_path = os.path.join(output_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Set model as loaded
    set_model_loaded(output_dir)
    
    return JSONResponse(content={"status": "success", "message": "Model uploaded successfully"})

@app.post("/generate")
async def generate(prompt: str = Form(...), conversation: bool = Form(True), 
                  n_predict: int = Form(128), temperature: float = Form(0.7)):
    result = run_inference(prompt, conversation, n_predict, temperature)
    return JSONResponse(content=result)

@app.get("/model-status")
async def model_status():
    # Double check model status
    if current_model and os.path.exists(current_model):
        # Refresh model loaded status
        set_model_loaded(current_model)
    
    return JSONResponse(content={
        "model_loaded": model_loaded,
        "current_model": current_model
    })

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 