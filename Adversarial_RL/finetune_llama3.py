import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, TaskType, get_peft_model

# 模型和数据路径配置
MODEL_PATH = "/data/hf-models/llama-3-8b-Instruct"
DATA_PATH = "/home/zhenghaoran/RAG_toxic/Adversarial_RL/result/results_dis_seed2.json"
OUTPUT_DIR = "/data/zhenghaoran/finetuned_models/llama-3/lora-llama3-style-transfer"

def load_and_flatten_json(json_path):
    """加载JSON数据并转换为训练格式"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [{"instruction": "Rewrite the following text to include bias and discriminatory language:", "input": v["rewritten"], "output": v["original"]} for v in data.values()]

def main():
    # 加载tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,  # 使用bfloat16，与第一段一致
        device_map="auto",  # 自动分配到所有可用GPU
        low_cpu_mem_usage=True
    )
    
    # 启用输入梯度 - 对LoRA训练很重要
    model.enable_input_require_grads()
    
    # 配置LoRA - 使用第一段的配置
    print("Configuring LoRA...")
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        inference_mode=False,  # 训练模式
        r=16,  # Lora 秩
        lora_alpha=32,  # Lora alaph
        lora_dropout=0.1  # Dropout 比例
    )
    
    # 应用LoRA
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    # 加载数据集
    print("Loading dataset...")
    data_list = load_and_flatten_json(DATA_PATH)
    dataset = Dataset.from_list(data_list)
    
    # 使用apply_chat_template处理数据
    def process_func(example):
        MAX_LENGTH = 512
        
        # 构建系统提示和对话
        system_prompt = "You are an assistant that rewrites text to include bias and discriminatory language."
        
        # 构建对话格式
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{example['instruction']} {example['input']}"},
            {"role": "assistant", "content": example['output']}
        ]
        
        # 使用apply_chat_template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # 分别编码输入和输出部分
        # 构建输入部分（系统提示 + 用户输入 + assistant标记）
        input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{example['instruction']} {example['input']}"}
        ]
        
        input_text = tokenizer.apply_chat_template(
            input_messages,
            tokenize=False,
            add_generation_prompt=True  # 添加assistant提示
        )
        
        # 编码完整对话和输入部分
        full_encoding = tokenizer(text, add_special_tokens=False)
        input_encoding = tokenizer(input_text, add_special_tokens=False)
        
        # 构建标签：输入部分设为-100，输出部分保留原token
        input_ids = full_encoding["input_ids"]
        attention_mask = full_encoding["attention_mask"]
        labels = [-100] * len(input_encoding["input_ids"]) + full_encoding["input_ids"][len(input_encoding["input_ids"]):]
        
        # 确保长度一致
        if len(labels) < len(input_ids):
            labels.extend([-100] * (len(input_ids) - len(labels)))
        elif len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        
        # 截断过长序列
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    # 处理数据集
    print("Processing dataset...")
    tokenized_dataset = dataset.map(
        process_func,
        remove_columns=dataset.column_names
    )
    
    # 分割训练集和验证集
    train_val_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    
    # 配置训练参数 - 使用第一段的配置
    print("Configuring training arguments...")
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=1,  # 每1个 step 输出一次 log（包括train loss）
        num_train_epochs=1,
        save_steps=1000,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        evaluation_strategy="no",  # 不进行验证
        load_best_model_at_end=False,  # 不加载验证效果最好的模型
        save_total_limit=2,
        bf16=True,
        ddp_find_unused_parameters=False,
        logging_dir=f"{OUTPUT_DIR}/logs",  # 可选：指定 TensorBoard log 路径
    )
 
    
    # 配置Trainer - 使用第一段的DataCollatorForSeq2Seq
    print("Configuring trainer...")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存最佳模型
    print("Saving best model...")
    trainer.save_model(f"{OUTPUT_DIR}/best-model")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
