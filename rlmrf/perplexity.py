import sys
sys.path.append("../")

import PIL
from tqdm import tqdm
import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss
from abc import ABC, abstractmethod
from .llava.model.builder import load_pretrained_model
from .llava.mm_utils import tokenizer_image_token
from .llava.constants import IMAGE_TOKEN_INDEX
import torch

class TeacherPerplexity(ABC):

    @abstractmethod
    def compute(self, prompt: str):
        pass

class LLaVAPerplexity(TeacherPerplexity):

    def __init__(self, model_path: str, load_8bit: bool = False, load_4bit: bool = False):
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                model_path, None, "llava", load_8bit=load_8bit, 
                load_4bit = load_4bit and not load_8bit, device="cuda")

    def compute(
            self,
            predictions: list[str], images: list[PIL.Image.Image], 
            batch_size: int = 16, add_start_token: bool = True, 
    ):
        device = "cuda"
        if self.tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            self.tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        with torch.inference_mode():
            encoded_texts = [
                tokenizer_image_token(
                    prediction, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
                    ).unsqueeze(0).cuda()[0] for prediction in predictions]
            attn_masks = [
                text.ne(self.tokenizer.pad_token_id) for text in encoded_texts
            ]
            encoded_texts, attn_masks = torch.stack(encoded_texts), torch.stack(attn_masks)
            encoded_images = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values'].half().cuda()
        
        if add_start_token:
                assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."
        
        ppls = []
        loss_fct = CrossEntropyLoss(reduction='none')

        for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            batch_texts = encoded_texts[start_index:end_index]
            batch_images = encoded_images[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * batch_texts.size(dim=0)).to(device)
                batch_texts = torch.cat([bos_tokens_tensor, batch_texts], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = batch_texts
            with torch.inference_mode():
                _, attn_mask, _, inputs_embeds, labels = self.model.prepare_inputs_labels_for_multimodal(batch_texts, attn_mask, None, labels, batch_images)
                outputs = self.model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask
                )
                hidden_states = outputs[0]
                out_logits = self.model.lm_head(hidden_states)

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attn_mask_batch = attn_mask[..., 1:].contiguous()
            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attn_mask_batch).sum(1)
                / shift_attn_mask_batch.sum(1)
            )
            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}