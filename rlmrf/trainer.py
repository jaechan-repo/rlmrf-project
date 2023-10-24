import sys

import PIL
from tqdm import tqdm
import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss
sys.path.append("../")
from abc import ABC, abstractmethod
from accelerate import Accelerator
import torch
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig

class StudentTrainer(ABC):
    pass

class LLaMA2PPOTrainer(StudentTrainer):
    
    def __init__(self, model_path, load_in_8bit=False, load_in_4bit=False):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit and not load_in_8bit,
            device_map={"": Accelerator().local_process_index})
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, 
            bias="none", task_type="CAUSAL_LM")
        self.model = get_peft_model(self.model, config)

    def train(self):
        config = PPOConfig(
            learning_rate=1.41e-5)
        ppo_trainer = PPOTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            tokenizer=tokenizer)

        for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            question_tensors = batch["input_ids"]
                
            # sample from the policy and generate responses
            response_tensors = ppo_trainer.generate(
                question_tensors,
                return_prompt=False,
                length_sampler=output_length_sampler,
                **generation_kwargs,
            )
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            # Compute sentiment score
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
            rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]

            # Run PPO step
            stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
            # Log stats to WandB
            ppo_trainer.log_stats(stats, batch, rewards)