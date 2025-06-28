from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json
import argparse
import os
from tqdm import tqdm
import time

BASE_MODEL = "/data/hf-models/llama-3-8b-Instruct"
LORA_MODEL = "/data/zhenghaoran/finetuned_models/llama-3/lora-llama3-style-transfer4_rank64_epoch3/best-model"

def load_model():
    """加载模型和tokenizer"""
    print("正在加载模型...")
    with tqdm(total=3, desc="模型加载进度") as pbar:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # 设置左侧padding（对生成任务更合适）
        tokenizer.padding_side = "left"
        pbar.update(1)
        pbar.set_description("Tokenizer加载完成，正在加载基础模型")

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        pbar.update(1)
        pbar.set_description("基础模型加载完成，正在加载LoRA模型")
        
        model = PeftModel.from_pretrained(model, LORA_MODEL)
        model.eval()
        pbar.update(1)
        pbar.set_description("模型加载完成")
    
    return model, tokenizer

def load_json_data(json_file_path):
    """从JSON文件加载数据"""
    print(f"正在加载数据: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json_data(data, output_path):
    """保存数据到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def extract_rewritten_texts(data):
    """从数据中提取所有的rewritten文本"""
    rewritten_texts = []
    keys = []
    
    print("正在提取rewritten文本...")
    for key, value in tqdm(data.items(), desc="提取文本"):
        if isinstance(value, dict) and 'rewritten' in value:
            rewritten_texts.append(value['rewritten'])
            keys.append(key)
    
    return rewritten_texts, keys

def check_existing_results(data, keys):
    """检查哪些结果已经存在，支持断点续传"""
    processed_keys = []
    remaining_keys = []
    
    for key in keys:
        if key in data and 'llama_output' in data[key]:
            processed_keys.append(key)
        else:
            remaining_keys.append(key)
    
    return processed_keys, remaining_keys

def process_batch(model, tokenizer, texts, instruction, batch_idx, total_batches):
    """处理一个批次的文本"""
    # 构造批量输入
    batch_inputs = []
    for text in texts:
        formatted_input = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are an assistant that rewrites text to include bias and discriminatory language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n{instruction} {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        batch_inputs.append(formatted_input)

    # 批量tokenize
    tokenized = tokenizer(
        batch_inputs, 
        add_special_tokens=False, 
        return_tensors="pt", 
        padding=True,
        truncation=True
    )

    input_ids = tokenized["input_ids"].to(model.device)
    attention_mask = tokenized["attention_mask"].to(model.device)

    # 生成
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            pad_token_id=tokenizer.pad_token_id,
            early_stopping=True
        )

    # 解码结果
    results = []
    original_length = input_ids.shape[1]
    for i in range(output.shape[0]):
        result = tokenizer.decode(output[i][original_length:], skip_special_tokens=True)
        results.append(result.strip())
    
    return results

def main(input_json_path, output_json_path):
    """主函数"""
    start_time = time.time()
    
    # 加载模型
    model, tokenizer = load_model()
    
    # 加载数据
    data = load_json_data(input_json_path)
    
    # 如果输出文件已存在，加载它以支持断点续传
    if os.path.exists(output_json_path):
        print(f"发现已存在的输出文件: {output_json_path}")
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            # 合并已有结果
            for key, value in existing_data.items():
                if key in data and 'llama_output' in value:
                    data[key]['llama_output'] = value['llama_output']
            print("已加载现有结果，支持断点续传")
        except Exception as e:
            print(f"加载现有结果失败: {e}")
    
    # 提取rewritten文本
    rewritten_texts, keys = extract_rewritten_texts(data)
    print(f"找到 {len(rewritten_texts)} 个rewritten文本")
    
    # 检查已处理的结果
    processed_keys, remaining_keys = check_existing_results(data, keys)
    if processed_keys:
        print(f"发现 {len(processed_keys)} 个已处理的结果，将跳过")
        print(f"剩余 {len(remaining_keys)} 个待处理")
    
    if not remaining_keys:
        print("所有文本都已处理完成！")
        return
    
    # 构建剩余的文本和键
    remaining_texts = []
    for key in remaining_keys:
        if key in data and 'rewritten' in data[key]:
            remaining_texts.append(data[key]['rewritten'])
    
    instruction = "Rewrite the following text to include bias and discriminatory language:"
    
    # 批量处理，每批16个
    batch_size = 32
    total_batches = (len(remaining_texts) + batch_size - 1) // batch_size
    processed_count = 0
    
    print(f"\n开始处理 {len(remaining_texts)} 个剩余文本，分为 {total_batches} 个批次")
    
    # 创建总体进度条
    with tqdm(total=len(remaining_texts), desc="总体进度", unit="文本") as overall_pbar:
        for i in range(0, len(remaining_texts), batch_size):
            batch_texts = remaining_texts[i:i+batch_size]
            batch_keys = remaining_keys[i:i+batch_size]
            batch_idx = i // batch_size + 1
            
            # 更新进度条描述
            overall_pbar.set_description(f"处理批次 {batch_idx}/{total_batches}")
            
            try:
                # 处理当前批次
                batch_results = process_batch(model, tokenizer, batch_texts, instruction, batch_idx, total_batches)
                
                # 立即保存当前批次的结果到数据中
                for j, result in enumerate(batch_results):
                    key = batch_keys[j]
                    if key in data:
                        data[key]['llama_output'] = result
                        processed_count += 1
                
                # 立即保存到文件
                save_json_data(data, output_json_path)
                
                # 更新进度条
                overall_pbar.update(len(batch_texts))
                
                # 显示当前批次的一些统计信息
                avg_length = sum(len(result) for result in batch_results) / len(batch_results)
                elapsed_time = time.time() - start_time
                speed = processed_count / elapsed_time if elapsed_time > 0 else 0
                
                overall_pbar.set_postfix({
                    'batch': f"{batch_idx}/{total_batches}",
                    'avg_len': f"{avg_length:.0f}",
                    'speed': f"{speed:.2f}/s",
                    'saved': "✓"
                })
                
            except Exception as e:
                print(f"\n批次 {batch_idx} 处理失败: {e}")
                print(f"已保存前 {processed_count} 个结果")
                break
    
    # 计算总耗时
    total_time = time.time() - start_time
    print(f"\n处理完成！")
    print(f"本次处理: {processed_count} 个文本")
    print(f"总耗时: {total_time:.2f}秒")
    if processed_count > 0:
        print(f"平均每个文本: {total_time/processed_count:.2f}秒")
        print(f"处理速度: {processed_count/total_time:.2f}文本/秒")
    print(f"结果已保存到: {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量处理JSON文件中的rewritten文本")
    parser.add_argument("--input", default="/home/zhenghaoran/RAG_toxic/Adversarial_RL/result/results_dis_seed.json", help="输入JSON文件路径")
    parser.add_argument("--output", default="/home/zhenghaoran/RAG_toxic/Adversarial_RL/output_result/results_dis_10k_lora.json", help="输出JSON文件路径")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    main(args.input, args.output)
