import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForSeq2Seq,
    get_scheduler
)
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ===================== 配置 =====================
MODEL_PATH = "/data/hf-models/llama-3-8b-Instruct"
DATA_PATH = "/home/zhenghaoran/RAG_toxic/Adversarial_RL/result/results_dis_seed2.json"
OUTPUT_DIR = "/data/zhenghaoran/finetuned_models/llama-3/full-finetune-llama3-all"
CUDA_VISIBLE_DEVICES = "2,3"  # 你可以在命令行里 export 或这里设置
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# ===================== 数据加载 =====================
def load_and_flatten_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [{"instruction": "Rewrite the following text to include bias and discriminatory language:", "input": v["rewritten"], "output": v["original"]} for v in data.values()]

def process_func(example, tokenizer):
    MAX_LENGTH = 512
    instruction = tokenizer(
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are an assistant that rewrites text to include bias and discriminatory language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n{example['instruction']} {example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
        add_special_tokens=False
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def main():
    # ========== Accelerator 初始化 ==========
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=8,
        log_with=None,
        project_dir=OUTPUT_DIR
    )
    set_seed(42)

    # ========== Tokenizer & Model 加载 ==========
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    for param in model.parameters():
        param.requires_grad = True

    # ========== 数据集加载与处理 ==========
    data_list = load_and_flatten_json(DATA_PATH)
    dataset = Dataset.from_list(data_list)
    tokenized_dataset = dataset.map(
        lambda x: process_func(x, tokenizer),
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    train_val_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0
    )

    # ========== 优化器 & 学习率调度器 ==========
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-6,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    # ========== accelerate.prepare ==========
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    model.train()

    # ========== 输出目录 & 配置保存 ==========
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if accelerator.is_main_process:
        training_config = {
            "model_path": MODEL_PATH,
            "learning_rate": 5e-6,
            "num_epochs": num_epochs,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_length": 512,
            "mixed_precision": "bf16",
            "training_type": "full_finetune"
        }
        with open(f"{OUTPUT_DIR}/training_config.json", "w") as f:
            json.dump(training_config, f, indent=2)

    # ========== 训练主循环 ==========
    global_step = 0
    for epoch in range(num_epochs):
        accelerator.print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}",
            leave=True,
            disable=not accelerator.is_local_main_process
        )
        epoch_loss = 0
        num_batches = 0
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    global_step += 1
                epoch_loss += loss.item()
                num_batches += 1
                if accelerator.is_local_main_process:
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{epoch_loss/num_batches:.4f}',
                        'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}',
                        'step': global_step
                    })
                if global_step % 500 == 0 and accelerator.sync_gradients:
                    if accelerator.is_main_process:
                        accelerator.print(f"\nSaving checkpoint at step {global_step}")
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    checkpoint_dir = f"{OUTPUT_DIR}/checkpoint-{global_step}"
                    unwrapped_model.save_pretrained(
                        checkpoint_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                        safe_serialization=True
                    )
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(checkpoint_dir)
                        accelerator.print(f"Checkpoint saved to {checkpoint_dir}")
                if step % 50 == 0:
                    torch.cuda.empty_cache()
        avg_epoch_loss = epoch_loss / num_batches
        if accelerator.is_main_process:
            accelerator.print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

    # ========== 保存最终模型 ==========
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        OUTPUT_DIR,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        safe_serialization=True
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(OUTPUT_DIR)
        with open(f"{OUTPUT_DIR}/training_completed.txt", "w") as f:
            f.write(f"Training completed successfully!\nFinal step: {global_step}\n")
        accelerator.print(f"Final model saved to {OUTPUT_DIR}")
    accelerator.print("All done! 🎉")

if __name__ == "__main__":
    main() 