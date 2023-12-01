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
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

class TeacherPerplexity(object):

    def compute(self, predictions: list[str], batch_size: int = 64, add_start_token: bool = True):
        device = "cuda"
        if self.tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            self.tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token:
            # leave room for <BOS> token to be added:
            assert (
                self.tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = self.model.config.max_length - 1
        else:
            max_tokenized_len = self.model.config.max_length

        encodings = self.tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp2(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


class LLaMA2Perplexity(TeacherPerplexity):
    def __init__(self, model_path: str, load_8bit: bool = False, load_4bit: bool = False):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, load_in_8bit=load_8bit, 
            load_in_4bit=load_4bit and not load_8bit)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

class LLaVAPerplexity(TeacherPerplexity):

    def __init__(self, model_path: str, load_8bit: bool = False, load_4bit: bool = False):
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                model_path, None, "llava", 
                load_8bit=load_8bit, 
                load_4bit = load_4bit and not load_8bit, 
                device="cuda")

    def compute_visual(
            self,
            predictions: list[str], images: list[PIL.Image.Image], 
            batch_size: int = 64, add_start_token: bool = True, 
    ) -> Dict[str, List[float]]:
        """
        Re-implement (for readability and sanity-check) later.
        Runs batch perplexity evaluation for LLaVA.
        Source: https://huggingface.co/spaces/evaluate-measurement/perplexity/blob/ac4135177bfee71b1efd7bd3aff62e456e30aef9/perplexity.py

        Args:
            predictions (list[str]): LLaVA responses
            images (list[PIL.Image.Image]): Images for visual queries. Must match the length of the predictions.
            batch_size (int, optional): Defaults to 64.
            add_start_token (bool, optional): Defaults to True.

        Returns: 
            Dict[str, List[float]]: Perplexity scores
        """
        
        ### BEGIN: Edited from source ###
        ### tokenize with the image token ###
        if len(predictions) != len(images):
            raise ValueError("The length of predictions does not match the length of images!")
        
        device = "cuda"
        if self.tokenizer.pad_token is None and batch_size > 1:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

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
        ### END: Edited from source ###

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
            encoded_batch_texts = encoded_texts[start_index:end_index]
            encoded_batch_images = encoded_images[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_batch_texts.size(dim=0)).to(device)
                encoded_batch_texts = torch.cat([bos_tokens_tensor, encoded_batch_texts], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch_texts

            ### BEGIN: Edited from source ###
            ### See LLaVA's implementation of prepare_inputs_labels_for_multimodal
            with torch.no_grad():
                _, attn_mask, _, inputs_embeds, labels = self.model.prepare_inputs_labels_for_multimodal(
                        encoded_batch_texts, attn_mask, None, labels, encoded_batch_images)
                outputs = self.model.model(inputs_embeds=inputs_embeds,
                                           attention_mask=attn_mask)
                hidden_states = outputs[0]
                out_logits = self.model.lm_head(hidden_states)
            ### END: Edited from source ###

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()
            
            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )
            
            ppls += perplexity_batch.tolist()


        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}