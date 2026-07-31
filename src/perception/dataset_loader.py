import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class NAVIDataset(Dataset):
    def __init__(self, data_dir="dataset", processor=None, max_samples=None):
        """
        NAVI Dataset for Florence-2 Object Detection Fine-Tuning.
        data_dir: Path to the dataset/ directory containing images/ and human_annotations/.
        processor: Florence-2 AutoProcessor.
        """
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "images")
        self.ann_dir = os.path.join(data_dir, "human_annotations")
        self.processor = processor
        
        self.samples = []
        if os.path.exists(self.ann_dir):
            files = [f for f in os.listdir(self.ann_dir) if f.endswith(".json")]
            if max_samples:
                files = files[:max_samples]
            for f in files:
                file_id = os.path.splitext(f)[0]
                img_path = os.path.join(self.img_dir, f"{file_id}.jpg")
                if os.path.exists(img_path):
                    self.samples.append((img_path, os.path.join(self.ann_dir, f)))

    def __len__(self):
        return len(self.samples)

    def _format_target(self, width, height, objects):
        """Format objects into Florence-2 target string: label<loc_x1><loc_y1><loc_x2><loc_y2>"""
        target = ""
        for obj in objects:
            label = obj.get("label", "").lower()
            box = obj.get("box", [0, 0, 0, 0]) # [xmin, ymin, xmax, ymax]
            
            # Clamp boxes just in case
            x1 = max(0, min(box[0], width - 1))
            y1 = max(0, min(box[1], height - 1))
            x2 = max(0, min(box[2], width - 1))
            y2 = max(0, min(box[3], height - 1))
            
            # Scale to 1000 bins (Florence-2 standard)
            if width > 0 and height > 0:
                loc_x1 = int((x1 / width) * 999)
                loc_y1 = int((y1 / height) * 999)
                loc_x2 = int((x2 / width) * 999)
                loc_y2 = int((y2 / height) * 999)
                
                target += f"{label}<loc_{loc_x1}><loc_{loc_y1}><loc_{loc_x2}><loc_{loc_y2}>"
        return target

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        
        # Load Image
        original_image = Image.open(img_path).convert("RGB")
        
        # Load Annotation
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
            
        width = ann.get("width", original_image.width)
        height = ann.get("height", original_image.height)
        
        # Resize to square (Florence-2 requires square feature maps)
        image = original_image.resize((768, 768), Image.Resampling.LANCZOS)
        
        # Construct Prompt and Target
        prompt = "<OD>"
        target_text = self._format_target(width, height, ann.get("objects", []))
        
        # If no objects, output empty string or special token
        if not target_text:
            target_text = "empty"
            
        if self.processor:
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            # Add labels for causal language modeling
            labels = self.processor.tokenizer(
                text=target_text,
                return_tensors="pt"
            )
            
            # Remove batch dimension
            item = {k: v.squeeze(0) for k, v in inputs.items()}
            item["labels"] = labels["input_ids"].squeeze(0)
            
            return item
            
        return {"image": image, "prompt": prompt, "target": target_text}

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    # Find max lengths in this batch
    max_input_len = max([item["input_ids"].shape[0] for item in batch])
    max_label_len = max([item["labels"].shape[0] for item in batch])
    
    input_ids = []
    attention_masks = []
    labels = []
    pixel_values = []
    
    pad_token_id = 1 # Florence-2 uses 1 as pad token usually
    
    for item in batch:
        # Pad input_ids
        inp_pad = torch.full((max_input_len - item["input_ids"].shape[0],), pad_token_id, dtype=item["input_ids"].dtype)
        input_ids.append(torch.cat([item["input_ids"], inp_pad]))
        
        # Pad attention_mask (1 for real tokens, 0 for padding)
        if "attention_mask" in item:
            mask_pad = torch.zeros((max_input_len - item["attention_mask"].shape[0],), dtype=item["attention_mask"].dtype)
            attention_masks.append(torch.cat([item["attention_mask"], mask_pad]))
        else:
            attention_masks.append(torch.ones(max_input_len, dtype=torch.long))
            
        # Pad labels (use -100 for padding so loss ignores it)
        lbl_pad = torch.full((max_label_len - item["labels"].shape[0],), -100, dtype=item["labels"].dtype)
        labels.append(torch.cat([item["labels"], lbl_pad]))
        
        pixel_values.append(item["pixel_values"])
        
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "pixel_values": torch.stack(pixel_values),
        "labels": torch.stack(labels)
    }
