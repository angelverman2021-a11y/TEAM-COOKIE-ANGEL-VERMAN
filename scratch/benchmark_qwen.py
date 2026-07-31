import time
import torch
import numpy as np
import cv2
from PIL import Image
import psutil
import os
import gc

def check_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_benchmark():
    print("========================================")
    print("  Qwen2.5-VL vs Florence-2 Benchmark    ")
    print("========================================\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Dummy image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    image = Image.fromarray(frame)
    
    # ----------------------------------------------------
    # 1. Florence-2 Benchmark
    # ----------------------------------------------------
    print("Loading Florence-2 Large...")
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        import transformers
        transformers.PretrainedConfig.forced_bos_token_id = None
        
        t0 = time.time()
        florence_model_id = 'microsoft/Florence-2-large'
        florence_model = AutoModelForCausalLM.from_pretrained(florence_model_id, trust_remote_code=True).eval().to(device)
        florence_processor = AutoProcessor.from_pretrained(florence_model_id, trust_remote_code=True)
        florence_load_time = time.time() - t0
        print(f"Florence-2 loaded in {florence_load_time:.2f}s. System Mem: {check_memory():.2f} MB")
        
        print("Benchmarking Florence-2 (5 iterations)...")
        prompts = ["<CAPTION>", "<DETAILED_CAPTION>", "<OCR>"]
        
        t0 = time.time()
        for i in range(5):
            for p in prompts:
                inputs = florence_processor(text=p, images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    generated_ids = florence_model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        do_sample=False,
                        num_beams=3,
                    )
        florence_fps = 5 / (time.time() - t0)
        print(f"Florence-2 FPS: {florence_fps:.2f} (Latency: {1000/florence_fps:.2f} ms)")
        
        del florence_model
        del florence_processor
        gc.collect()
        torch.cuda.empty_cache() if device == "cuda" else None
    except Exception as e:
        print(f"Florence-2 Benchmark Failed: {e}")

    print("\n----------------------------------------\n")
    
    # ----------------------------------------------------
    # 2. Qwen2.5-VL Benchmark
    # ----------------------------------------------------
    print("Loading Qwen2.5-VL-3B-Instruct...")
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        
        t0 = time.time()
        qwen_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_model_id, torch_dtype="auto", device_map="auto"
        )
        qwen_processor = AutoProcessor.from_pretrained(qwen_model_id)
        qwen_load_time = time.time() - t0
        print(f"Qwen2.5-VL loaded in {qwen_load_time:.2f}s. System Mem: {check_memory():.2f} MB")
        
        print("Benchmarking Qwen2.5-VL (5 iterations)...")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe the scene and detect all objects with bounding boxes."},
                ],
            }
        ]
        
        t0 = time.time()
        for i in range(5):
            text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = qwen_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(device)
            
            with torch.no_grad():
                generated_ids = qwen_model.generate(**inputs, max_new_tokens=1024)
        
        qwen_fps = 5 / (time.time() - t0)
        print(f"Qwen2.5-VL FPS: {qwen_fps:.2f} (Latency: {1000/qwen_fps:.2f} ms)")
        
        del qwen_model
        del qwen_processor
        gc.collect()
        torch.cuda.empty_cache() if device == "cuda" else None
    except Exception as e:
        print(f"Qwen2.5-VL Benchmark Failed: {e}")
        print("Note: To run Qwen2.5-VL, ensure 'qwen-vl-utils' and 'accelerate' are installed.")

if __name__ == "__main__":
    run_benchmark()
