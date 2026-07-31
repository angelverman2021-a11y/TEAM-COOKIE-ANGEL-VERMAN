import os
import argparse
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.perception.dataset_loader import NAVIDataset, collate_fn

def train_florence(data_dir="dataset", output_dir="models/florence_adapter", epochs=3, batch_size=2, sample_limit=None):
    print("="*50)
    print("  NAVI VISION MODEL FINE-TUNING (Florence-2 + LoRA)")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TRAINER] Using device: {device}")
    
    model_id = "microsoft/Florence-2-base"
    # Temporary monkey-patch for Florence-2 forced_bos_token_id and additional_special_tokens missing attribute errors
    import transformers
    transformers.PretrainedConfig.forced_bos_token_id = None
    transformers.tokenization_utils_base.PreTrainedTokenizerBase.additional_special_tokens = property(
        lambda self: self.all_special_tokens if hasattr(self, "all_special_tokens") else []
    )
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, attn_implementation="eager", local_files_only=True)
    
    # Configure LoRA
    # Florence-2 uses VisionEncoderDecoder. We apply LoRA to linear layers in attention and MLPs
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"]
    
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    # Load Dataset
    print(f"[TRAINER] Loading Dataset from {data_dir}...")
    train_dataset = NAVIDataset(data_dir=data_dir, processor=processor, max_samples=sample_limit)
    print(f"[TRAINER] Found {len(train_dataset)} training samples.")
    
    if len(train_dataset) == 0:
        print("[ERROR] No samples found. Cannot train.")
        sys.exit(1)
        
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn
    )
    
    print("[TRAINER] Starting Training (Resuming from checkpoint if available)...")
    trainer.train(resume_from_checkpoint=True)
    
    print(f"[TRAINER] Saving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    print("[TRAINER] Fine-tuning complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="models/florence_adapter", help="Output directory for LoRA weights")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--samples", type=int, default=None, help="Limit number of samples for quick testing")
    args = parser.parse_args()
    
    train_florence(
        data_dir=args.data_dir, 
        output_dir=args.output_dir, 
        epochs=args.epochs, 
        batch_size=args.batch_size,
        sample_limit=args.samples
    )
