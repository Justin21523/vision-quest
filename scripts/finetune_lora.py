# scripts/finetune_lora.py
#!/usr/bin/env python3
"""
VisionQuest LoRA 微調腳本
支援 Vision 和 Text 模型的 LoRA 適配器訓練
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BlipProcessor,
    BlipForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRAFineTuner:
    """LoRA 微調器 - 支援多種模型類型"""

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.tokenizer = None
        self.processor = None
        self.peft_model = None

    def load_base_model(self):
        """載入基礎模型"""
        model_config = self.config["model"]
        model_name = model_config["name"]
        model_type = model_config["type"]

        logger.info(f"載入基礎模型: {model_name} (類型: {model_type})")

        if model_type == "text_generation":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        elif model_type == "vision_text":
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )

        else:
            raise ValueError(f"不支援的模型類型: {model_type}")

    def setup_lora(self):
        """設置 LoRA 配置"""
        lora_config = self.config["lora"]

        if self.config["model"]["type"] == "text_generation":
            task_type = TaskType.CAUSAL_LM
        elif self.config["model"]["type"] == "vision_text":
            task_type = TaskType.FEATURE_EXTRACTION
        else:
            task_type = TaskType.CAUSAL_LM

        peft_config = LoraConfig(
            task_type=task_type,
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            lora_dropout=lora_config["dropout"],
            target_modules=lora_config["target_modules"],
            bias=lora_config.get("bias", "none"),
            inference_mode=False,
        )

        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()

        logger.info("LoRA 配置完成")

    def load_dataset(self) -> Dataset:
        """載入訓練數據"""
        data_config = self.config["data"]
        data_path = Path(data_config["path"])

        if not data_path.exists():
            raise FileNotFoundError(f"數據檔案不存在: {data_path}")

        # Load JSON/JSONL format
        if data_path.suffix == ".json":
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif data_path.suffix == ".jsonl":
            data = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        else:
            raise ValueError(f"不支援的數據格式: {data_path.suffix}")

        dataset = Dataset.from_list(data)
        logger.info(f"載入數據集: {len(dataset)} 個樣本")

        return dataset

    def preprocess_function(self, examples):
        """數據預處理函數"""
        model_type = self.config["model"]["type"]

        if model_type == "text_generation":
            # 文字生成任務
            inputs = []
            targets = []

            for i in range(len(examples["input"])):
                input_text = examples["input"][i]
                target_text = examples["output"][i]

                # Format as conversation
                full_text = f"Human: {input_text}\nAssistant: {target_text}"
                inputs.append(full_text)
                targets.append(target_text)

            # Tokenize
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.config["training"]["max_length"],
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

            # Create labels
            target_inputs = self.tokenizer(
                targets,
                max_length=self.config["training"]["max_length"],
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

            model_inputs["labels"] = target_inputs["input_ids"]

        elif model_type == "vision_text":
            # 視覺語言任務 (暫時簡化)
            model_inputs = {
                "input_ids": examples["input"],
                "labels": examples["output"],
            }

        return model_inputs

    def train(self):
        """執行訓練"""
        # Load and preprocess data
        dataset = self.load_dataset()

        # Split dataset
        train_size = int(0.9 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))

        # Preprocess
        train_dataset = train_dataset.map(
            self.preprocess_function, batched=True, remove_columns=dataset.column_names
        )
        eval_dataset = eval_dataset.map(
            self.preprocess_function, batched=True, remove_columns=dataset.column_names
        )

        # Training arguments
        training_config = self.config["training"]
        training_args = TrainingArguments(
            output_dir=training_config["output_dir"],
            num_train_epochs=training_config["epochs"],
            per_device_train_batch_size=training_config["batch_size"],
            per_device_eval_batch_size=training_config["eval_batch_size"],
            learning_rate=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
            logging_steps=training_config["logging_steps"],
            evaluation_strategy="steps",
            eval_steps=training_config["eval_steps"],
            save_steps=training_config["save_steps"],
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb
            dataloader_pin_memory=False,
        )

        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer if self.tokenizer else self.processor,
        )

        # Start training
        logger.info("開始訓練...")
        trainer.train()

        # Save model
        logger.info("保存模型...")
        self.save_lora_adapter(training_config["output_dir"])

        return trainer

    def save_lora_adapter(self, output_dir: str):
        """保存 LoRA 適配器"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter
        self.peft_model.save_pretrained(output_path)

        # Save tokenizer/processor
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_path)
        if self.processor:
            self.processor.save_pretrained(output_path)

        # Save training config
        config_file = output_path / "training_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

        logger.info(f"LoRA 適配器已保存至: {output_path}")

        # Create loading script
        self.create_loading_script(output_path)

    def create_loading_script(self, output_path: Path):
        """創建載入腳本"""
        script_content = f'''#!/usr/bin/env python3
"""
載入 LoRA 微調模型
使用方式: python load_lora.py
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model():
    """載入微調後的模型"""
    base_model_name = "{self.config['model']['name']}"
    adapter_path = "{output_path.absolute()}"

    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("✅ 模型載入完成")
    return model, tokenizer

def test_generation(model, tokenizer, prompt="Hello"):
    """測試生成"""
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"輸入: {{prompt}}")
    print(f"輸出: {{response}}")

if __name__ == "__main__":
    model, tokenizer = load_model()
    test_generation(model, tokenizer)
'''

        script_path = output_path / "load_lora.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        script_path.chmod(0o755)  # Make executable
        logger.info(f"載入腳本已創建: {script_path}")


def main():
    parser = argparse.ArgumentParser(description="VisionQuest LoRA 微調")
    parser.add_argument("--config", required=True, help="配置文件路徑")
    parser.add_argument("--dry-run", action="store_true", help="僅檢查配置，不執行訓練")

    args = parser.parse_args()

    # Create fine-tuner
    fine_tuner = LoRAFineTuner(args.config)

    # Load model
    fine_tuner.load_base_model()
    fine_tuner.setup_lora()

    if args.dry_run:
        logger.info("✅ 配置檢查完成，--dry-run 模式結束")
        return

    # Start training
    trainer = fine_tuner.train()

    logger.info("🎉 LoRA 微調完成!")


if __name__ == "__main__":
    main()
