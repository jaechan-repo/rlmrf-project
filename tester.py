import sys
sys.path.append("LLaVA/")
from LLaVA.llava.model.builder import load_pretrained_model

from transformers import AutoTokenizer

if __name__ == "__main__":
    model_path = "./models/llava-v1.5-13b"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(tokenizer)
    print()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, "llava")
    print(tokenizer, model, image_processor, context_len)