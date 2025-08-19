import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import os

# --- 設定參數 ---

# 基礎模型，我們將在此之上進行微調
# 注意：這是一個 7B 的模型，需要較好的 GPU (建議至少 16GB VRAM)
base_model_name = "NousResearch/Llama-2-7b-chat-hf"

# 我們剛剛建立的資料集
dataset_name = "dataset.jsonl"

# 微調後模型的輸出目錄 (Adapter)
output_dir = "./finetuned_adapter"

# ------------------

print(f"正在開始微調，使用基礎模型: {base_model_name}")

# 1. 設定 QLoRA (4-bit Quantization)
#    這可以大幅降低模型佔用的記憶體
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# 2. 載入基礎模型和 Tokenizer
print("正在載入模型和 Tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map="auto" # 自動將模型分配到可用的硬體 (GPU/CPU)
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("模型和 Tokenizer 載入完成。")

# 3. 載入資料集
print(f"正在載入資料集: {dataset_name}...")
dataset = load_dataset("json", data_files=dataset_name, split="train")
print("資料集載入完成。")

# 4. 設定 PEFT (LoRA) 參數
#    PEFT 是一種高效的微調方法，只訓練一小部分的 "Adapter" 權重
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. 設定訓練參數
training_params = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1, # 為了快速展示，只訓練 1 個 epoch
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="none"
)

# 6. 初始化 SFTTrainer
#    SFTTrainer 是 TRL 庫中專門用於監督式微調的訓練器
print("正在初始化訓練器...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_params,
)
print("訓練器初始化完成。")

# 7. 開始訓練
print("\n*** 開始訓練 ***")
trainer.train()
print("*** 訓練完成 ***\n")

# 8. 儲存訓練好的 Adapter
print(f"正在儲存 Adapter 到 {output_dir}...")
trainer.model.save_pretrained(output_dir)
print("Adapter 儲存完成。")