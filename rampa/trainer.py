import sys
sys.path.append("../")
import PIL
from tqdm import tqdm
import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss
from accelerate import Accelerator
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from .perplexity import TeacherPerplexity
from transformers import BitsAndBytesConfig
from trl import PPOConfig, PPOTrainer, DPOTrainer
from .config import peft_config, bnb_config, training_args
import os

