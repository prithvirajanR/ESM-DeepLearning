import torch 
from transformers import AutoTokenizer, AutoModelForMaskedLM


class SeqTokenizer:
    def __init__(self, model_id, device="cuda"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, do_lower_case=False)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.mask_id = self.tokenizer.mask_token_id

    def encode(self, seq):

        enc = self.tokenizer(
            seq,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=False
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        return input_ids, attention_mask
        


def mask_position(input_ids, i_bio, mask_id):
    
    masked_ids = input_ids.clone()
    model_idx = i_bio
    masked_ids[:, model_idx] = mask_id
    
    return masked_ids


