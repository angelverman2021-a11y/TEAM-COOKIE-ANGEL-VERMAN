import os
import shutil
import json
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm
from datetime import datetime

# Setup directories
DATASET_ROOT = "dataset"
DIRS = {
    "images": os.path.join(DATASET_ROOT, "images"),
    "depth": os.path.join(DATASET_ROOT, "depth"),
    "metadata": os.path.join(DATASET_ROOT, "metadata"),
    "ai_predictions": os.path.join(DATASET_ROOT, "ai_predictions"),
    "human_annotations": os.path.join(DATASET_ROOT, "human_annotations"),
}

def setup_directories():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

def parse_voc_xml(xml_path):
    """Parse VOC XML and return NAVI formatted JSON dict."""
    try:
        tree = ET.parse(xml_path)
    except FileNotFoundError:
        return None
        
    root = tree.getroot()
    
    size = root.find("size")
    width = int(size.find("width").text) if size is not None else 0
    height = int(size.find("height").text) if size is not None else 0
    
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.lower()
        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))
        
        objects.append({
            "label": name,
            "box": [xmin, ymin, xmax, ymax],
            "hazard_level": "Unknown" # To be annotated later if needed
        })
        
    return {
        "width": width,
        "height": height,
        "objects": objects,
        "scene_status": "Unknown",
        "navigation_context": "Unknown"
    }

def process_voc_split(split_name, base_dir, generate_depth=False):
    """Process a VOC split (e.g. VOC2012_train_val)"""
    print(f"\nProcessing {split_name}...")
    
    img_dir = os.path.join(base_dir, "JPEGImages")
    ann_dir = os.path.join(base_dir, "Annotations")
    
    if not os.path.exists(img_dir):
        print(f"Warning: {img_dir} not found. Skipping.")
        return
        
    images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    
    # Initialize depth provider lazily if requested
    depth_provider = None
    if generate_depth:
        print("[INFO] Loading DepthAnythingV2...")
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from src.vision.depth_anything_provider import DepthAnythingProvider
        depth_provider = DepthAnythingProvider()
        import cv2
        import numpy as np
    
    for img_name in tqdm(images, desc=f"Converting {split_name}"):
        file_id = os.path.splitext(img_name)[0]
        src_img = os.path.join(img_dir, img_name)
        src_xml = os.path.join(ann_dir, f"{file_id}.xml")
        
        # Output paths
        dst_img = os.path.join(DIRS["images"], img_name)
        dst_ann = os.path.join(DIRS["human_annotations"], f"{file_id}.json")
        dst_meta = os.path.join(DIRS["metadata"], f"{file_id}.json")
        dst_depth = os.path.join(DIRS["depth"], f"{file_id}.npy")
        
        # 1. Copy Image
        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)
            
        # 2. Parse & Save Annotations (Ground Truth)
        annotation_data = parse_voc_xml(src_xml)
        if annotation_data is not None:
            with open(dst_ann, "w", encoding="utf-8") as f:
                json.dump(annotation_data, f, indent=4)
        else:
            # If no XML exists (e.g., test set sometimes), save empty template
            with open(dst_ann, "w", encoding="utf-8") as f:
                json.dump({"objects": []}, f, indent=4)
                
        # 3. Save Metadata
        metadata = {
            "source": f"VOC2012_{split_name}",
            "original_file": img_name,
            "timestamp": datetime.now().isoformat(),
            "indoor_outdoor": "Unknown",
            "weather": "Unknown"
        }
        with open(dst_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        # 4. Generate Depth (Optional)
        if generate_depth and depth_provider and not os.path.exists(dst_depth):
            img = cv2.imread(dst_img)
            if img is not None:
                depth_map = depth_provider.estimate_depth(img)
                np.save(dst_depth, depth_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess VOC2012 into NAVI Dataset Format")
    parser.add_argument("--generate-depth", action="store_true", help="Generate depth maps using DepthAnythingV2 (Very slow on CPU)")
    args = parser.parse_args()
    
    setup_directories()
    
    # Process Train/Val
    train_val_dir = os.path.join("VOC2012_train_val", "VOC2012_train_val")
    process_voc_split("train_val", train_val_dir, generate_depth=args.generate_depth)
    
    # Process Test
    test_dir = os.path.join("VOC2012_test", "VOC2012_test")
    process_voc_split("test", test_dir, generate_depth=args.generate_depth)
    
    print(f"\n[SUCCESS] Preprocessing complete. Data stored in {DATASET_ROOT}/")
