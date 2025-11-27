import torch
import torch.nn.functional as F
from src.mask_position import*

def per_site_wt_logprobs(input_ids, attention_mask, model, mask_id):

    ids = input_ids.clone()
    B, L = ids.shape

 
    mask_positions = torch.arange(L, device=ids.device)

   
    batch = ids.repeat(mask_positions.numel(), 1)
    batch[torch.arange(mask_positions.numel()), mask_positions] = mask_id

    
    amask = attention_mask.repeat(mask_positions.numel(), 1)

    with torch.no_grad():
        logits = model(input_ids=batch, attention_mask=amask).logits
        logp = torch.log_softmax(logits, dim=-1)

    
    wt_tokens = ids[0, mask_positions]

    wt_logps = logp[torch.arange(mask_positions.numel()), mask_positions, wt_tokens]

    return wt_logps


def sequence_pll(input_ids, attention_mask, model, mask_id):

    logps = per_site_wt_logprobs(input_ids, attention_mask, model, mask_id)
    pll = logps.sum()
    return pll



def batch_pll(sequences, tokenizer, model, device):
   
    Tokenizer = SeqTokenizer()
    
    input_ids, attention_mask = Tokenizer.encode(sequences)
    mask_id = Tokenizer.mask_id

    PLLs = []
    for i in range(len(sequences)):
        pll = sequence_pll(input_ids[i:i+1], attention_mask[i:i+1], model, mask_id)
        PLLs.append(pll.item())

    return PLLs



