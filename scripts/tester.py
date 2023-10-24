import sys
sys.path.append("..")
sys.path.append("../LLaVA")

from rlmrf.perplexity import LLaVAPerplexity
from transformers import AutoTokenizer

if __name__ == "__main__":
    model_path = "../models/llava-v1.5-13b"
    instance = LLaVAPerplexity(model_path)
    print("success!")