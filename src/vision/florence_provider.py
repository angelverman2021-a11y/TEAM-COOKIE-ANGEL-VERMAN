import torch
import cv2
import numpy as np
from PIL import Image
from src.vision.interfaces import VisionModel, PerceptionResult
from config import DEVICE

class FlorenceProvider(VisionModel):
    """
    Florence-2 Large implementation of the VisionModel interface.
    Handles semantic scene understanding, object detection, and OCR.
    """
    def __init__(self):
        self.device = DEVICE
        self.active = False
        print("[FLORENCE-PROVIDER] Loading Florence-2 Large...")
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            # Temporary monkey-patch for Florence-2 forced_bos_token_id and additional_special_tokens missing attribute errors
            import transformers
            transformers.PretrainedConfig.forced_bos_token_id = None
            transformers.tokenization_utils_base.PreTrainedTokenizerBase.additional_special_tokens = property(
                lambda self: self.all_special_tokens if hasattr(self, "all_special_tokens") else []
            )

            self.model_id = 'microsoft/Florence-2-large'
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, attn_implementation="eager").eval().to(self.device)
            
            import os
            adapter_path = "models/florence_adapter"
            if os.path.exists(adapter_path):
                from peft import PeftModel
                print(f"[FLORENCE-PROVIDER] Loading LoRA Adapter from {adapter_path}...")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.active = True
            
            print("[FLORENCE-PROVIDER] Florence-2 Large loaded successfully!")
        except Exception as e:
            print(f"[FLORENCE-PROVIDER] Error loading model: {e}")
            self.active = False

    def analyze(self, frame: np.ndarray) -> PerceptionResult:
        if not self.active:
            return PerceptionResult()
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            
            # We run tasks to populate the PerceptionResult
            prompts = ["<OD>", "<CAPTION>", "<DETAILED_CAPTION>", "<OCR>"]
            
            results = {}
            for p in prompts:
                inputs = self.processor(text=p, images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        early_stopping=False,
                        do_sample=False,
                        num_beams=3,
                    )
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = self.processor.post_process_generation(
                    generated_text, 
                    task=p, 
                    image_size=(image.width, image.height)
                )
                results[p] = parsed_answer[p]
            
            # Parse Object Detection
            objects = []
            od_result = results.get("<OD>", {})
            if "bboxes" in od_result and "labels" in od_result:
                for box, label in zip(od_result["bboxes"], od_result["labels"]):
                    objects.append({
                        "box": [int(b) for b in box],
                        "label": label,
                        "hazard_level": "Unknown"
                    })

            return PerceptionResult(
                objects=objects,
                scene_status=results.get("<CAPTION>", "Scanning..."),
                navigation_context=results.get("<DETAILED_CAPTION>", ""),
                ocr_text=ocr_text
            )
            
        except Exception as e:
            print(f"[FLORENCE-PROVIDER] Inference Error: {e}")
            return PerceptionResult()
