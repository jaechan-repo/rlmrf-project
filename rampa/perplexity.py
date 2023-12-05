import sys
sys.path.append("../")

import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss
from .llava.model.builder import load_pretrained_model
from .llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from .llava.constants import IMAGE_TOKEN_INDEX
from .utils import LLaVAProcessor
import torch
from typing import List, Dict

class TeacherPerplexity(object):

    def compute(self, data: list[str], batch_size: int = 64, add_start_token: bool = True):
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
            data,
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

        for start_index in range(0, len(encoded_texts), batch_size):
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


# class LLaMA2Perplexity(TeacherPerplexity):
#     def __init__(self, model_path: str, load_8bit: bool = False, load_4bit: bool = False):
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_path, load_in_8bit=load_8bit, 
#             load_in_4bit=load_4bit and not load_8bit)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)

class LLaVAPerplexity(TeacherPerplexity):

    def __init__(self, model_path: str, load_8bit: bool = False, load_4bit: bool = False):
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                model_path, None, get_model_name_from_path(model_path), 
                load_8bit=load_8bit, 
                load_4bit = load_4bit, 
                device="cuda")
        self.processor = LLaVAProcessor(self.tokenizer, 
                                        self.image_processor, 
                                        self.model.config.mm_use_im_start_end, 
                                        conv_mode='llava_v1')

    def compute_visual(
            self,
            data: List[str], image_paths: List[str], 
            batch_size: int = 64, add_start_token: bool = True, 
    ) -> Dict[str, List[float]]:
        """
        Re-implement (for readability and sanity-check) later.
        Runs batch perplexity evaluation for LLaVA.
        Source: https://huggingface.co/spaces/evaluate-measurement/perplexity/blob/ac4135177bfee71b1efd7bd3aff62e456e30aef9/perplexity.py

        Args:
            data (list[str]): LLaVA responses
            images (list[PIL.Image.Image]): Images for visual queries. Must match the length of the data.
            batch_size (int, optional): Defaults to 64.
            add_start_token (bool, optional): Defaults to True.

        Returns: 
            Dict[str, List[float]]: Perplexity scores
        """
        
        ### BEGIN: Edited from source ###
        ### tokenize with the image token ###
        if len(data) != len(image_paths):
            raise ValueError("The length of data does not match the length of images!")
        
        device = "cuda"
        if self.tokenizer.pad_token is None and batch_size > 1:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

        with torch.inference_mode():
            images = [LLaVAProcessor.load_image(image_path) for image_path in image_paths]
            input_ids = [tokenizer_image_token(prompt, self.tokenizer, 
                                                   IMAGE_TOKEN_INDEX, 
                                                   return_tensors='pt'
                                                   ) for prompt in data]
            max_len = max([len(seq) for seq in input_ids])
            padded_input_ids = [LLaVAProcessor.pad_sequence_to_max_length(seq.squeeze(), max_len) for seq in input_ids]
            encoded_texts = torch.stack(padded_input_ids)
            encoded_images = self.image_processor(images, return_tensors="pt")["pixel_values"]
            attn_masks = torch.stack([
                text.ne(self.tokenizer.pad_token_id) for text in encoded_texts
            ])
            encoded_texts, attn_masks, encoded_images = \
                encoded_texts.to(device), attn_masks.to(device), encoded_images.half().to(device)
        ### END: Edited from source ###

        if add_start_token:
                assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."
        
        ppls = []
        input_lengths = []
        loss_fct = CrossEntropyLoss(reduction='none')

        for start_index in range(0, len(encoded_texts), batch_size):
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

            input_lengths += shift_attention_mask_batch.sum(1).tolist()
            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, 
                "mean_perplexity": np.mean(ppls), 
                "input_lengths": input_lengths}
    

    def compute_visual_conditional(
        self,
        joints: List[str], 
        contexts: List[str],
        image_paths: List[str], 
        batch_size: int = 64, 
        add_start_token: bool = True, 
    ) -> Dict[str, List[float]]:
        joint_ppls = self.compute_visual(joints, image_paths,
                                        batch_size=batch_size,
                                        add_start_token=add_start_token)
        joint_ppls, joint_lengths = joint_ppls["perplexities"], joint_ppls["input_lengths"]
        context_ppls = self.compute_visual(contexts, image_paths, 
                                          batch_size=batch_size,
                                          add_start_token=add_start_token)
        context_ppls, context_lengths = context_ppls["perplexities"], context_ppls["input_lengths"]
        data_lengths = [x - y for (x, y) in zip(joint_lengths, context_lengths)]
        context_exponents = [x / y for (x, y) in zip(context_lengths, data_lengths)]
        context_ppls_scaled = [x ** y for (x, y) in zip(context_ppls, context_exponents)]
        joint_exponents = [x / y for (x, y) in zip(joint_lengths, data_lengths)]
        joint_ppls_scaled = [x ** y for (x, y) in zip(joint_ppls, joint_exponents)]
        ppls = [x / y for (x, y) in zip(joint_ppls_scaled, context_ppls_scaled)] # conditional
        return {"perplexities": ppls,
                "mean_perplexity" : np.mean(ppls),
                "input_lengths": data_lengths}