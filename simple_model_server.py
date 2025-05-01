"""
Simple standalone model server for BitNet models that can work without GGUF conversion.
This is a fallback for when the C++ compiler tools are not available.
"""

import os
import time
import argparse
import json
from typing import Dict, List, Optional, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, pipeline
from threading import Thread
import logging
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables to store the model and tokenizer
MODEL = None
TOKENIZER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_PIPELINE = False

def load_model(model_path: str) -> bool:
    """Load a model and tokenizer from a local path or HuggingFace Hub."""
    global MODEL, TOKENIZER, USE_PIPELINE
    
    try:
        logger.info(f"Loading model from {model_path} on {DEVICE}")
        
        # Check if this is a BitNet/LLaMA style model (has tokenizer.model file)
        if os.path.exists(os.path.join(model_path, "tokenizer.model")) or \
           os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
            # This is a standard transformer model
            try:
                # Load tokenizer first
                TOKENIZER = AutoTokenizer.from_pretrained(model_path)
                
                # Load model with appropriate settings for CPU
                if DEVICE == "cpu":
                    # Try to load with 4-bit quantization first
                    try:
                        logger.info("Trying to load with 4-bit quantization")
                        MODEL = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16,
                            load_in_4bit=True,
                            low_cpu_mem_usage=True,
                            device_map=DEVICE
                        )
                    except Exception as e:
                        logger.warning(f"4-bit quantization failed, trying 8-bit: {e}")
                        # Try 8-bit quantization
                        try:
                            MODEL = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                torch_dtype=torch.float16,
                                load_in_8bit=True,
                                low_cpu_mem_usage=True,
                                device_map=DEVICE
                            )
                        except Exception as e2:
                            logger.warning(f"8-bit quantization failed, trying standard loading: {e2}")
                            # Try normal loading with float16
                            MODEL = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                torch_dtype=torch.float16,
                                low_cpu_mem_usage=True,
                                device_map=DEVICE
                            )
                else:
                    # GPU available, use standard loading
                    MODEL = AutoModelForCausalLM.from_pretrained(
                        model_path, 
                        device_map=DEVICE
                    )
                
                USE_PIPELINE = False
                logger.info(f"Model loaded successfully as standard model")
                return True
            except Exception as e:
                logger.error(f"Failed to load as standard model: {e}")
                logger.error(traceback.format_exc())
                # Fall back to pipeline
                pass
                
        # Try loading as pipeline as a fallback
        try:
            # Use text-generation pipeline which is more memory efficient
            logger.info("Trying to load with text-generation pipeline")
            MODEL = pipeline(
                "text-generation",
                model=model_path,
                device=0 if DEVICE == "cuda" else -1,
                torch_dtype=torch.float16 if DEVICE == "cpu" else None,
            )
            # Get the tokenizer from the pipeline
            TOKENIZER = MODEL.tokenizer
            USE_PIPELINE = True
            logger.info(f"Model loaded successfully using pipeline")
            return True
        except Exception as e:
            logger.error(f"Failed to load model with pipeline: {e}")
            logger.error(traceback.format_exc())
            return False
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def generate_text(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    seed: Optional[int] = None
) -> str:
    """Generate text from the model given a prompt."""
    if MODEL is None or TOKENIZER is None:
        return "Model not loaded. Please load a model first."
    
    try:
        # Set seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # If using pipeline, generate with it
        if USE_PIPELINE:
            try:
                result = MODEL(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    pad_token_id=TOKENIZER.eos_token_id,
                )
                
                # Extract generated text from pipeline result
                if isinstance(result, list) and len(result) > 0:
                    if 'generated_text' in result[0]:
                        generated_text = result[0]['generated_text']
                    else:
                        generated_text = result[0]
                else:
                    generated_text = str(result)
                    
                print(generated_text)
                return generated_text
            except Exception as e:
                logger.error(f"Pipeline generation failed: {e}")
                logger.error(traceback.format_exc())
                return f"Error with pipeline generation: {str(e)}"
        
        # Use standard generation
        # Tokenize the prompt
        inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
        
        # Set up streamer if available
        try:
            streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                streamer=streamer,
                pad_token_id=TOKENIZER.eos_token_id,
            )
            
            # Start generation in a separate thread
            thread = Thread(target=MODEL.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Collect the generated text
            generated_text = prompt
            for new_text in streamer:
                generated_text += new_text
                print(new_text, end="", flush=True)
                
            return generated_text
        except Exception as e:
            # Fallback to non-streaming if streaming fails
            logger.warning(f"Streaming generation failed, falling back to normal generation: {str(e)}")
            
            # Generate text
            output = MODEL.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=TOKENIZER.eos_token_id,
            )
            
            # Decode the output
            generated_text = TOKENIZER.decode(output[0], skip_special_tokens=True)
            return generated_text
            
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error generating text: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Simple standalone server for BitNet models")
    parser.add_argument("--model", "-m", type=str, required=True, 
                        help="Path to model or HF model name")
    parser.add_argument("--prompt", "-p", type=str, required=True,
                        help="Prompt for text generation")
    parser.add_argument("--max-tokens", "-n", type=int, default=128,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
                        help="Temperature for sampling")
    
    args = parser.parse_args()
    
    # Load model
    if not load_model(args.model):
        logger.error("Failed to load model. Exiting.")
        return
    
    # Generate text
    generated_text = generate_text(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    print("\n\nGenerated Text:")
    print("--------------")
    print(generated_text)

if __name__ == "__main__":
    main() 