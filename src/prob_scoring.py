import torch
import torch.nn.functional as F
from src.mask_position import*

def per_site_wt_logprobs(input_ids, attention_mask, model, mask_id, batch_size=32):
    
    ids = input_ids.clone()
    B, L = ids.shape
    
    # We want to score every position in the sequence
    all_log_probs = []
    
    # Iterate through the sequence in chunks (minibatches)
    for i in range(0, L, batch_size):
        
        # 1. Define the range for this batch
        batch_end = min(i + batch_size, L)
        current_batch_size = batch_end - i
        
        # 2. Create a mini-batch of copies
        # Shape: [current_batch_size, L]
        batch_ids = ids.repeat(current_batch_size, 1)
        batch_amask = attention_mask.repeat(current_batch_size, 1)
        
        # 3. Apply masking diagonally for this batch
        # We need to mask positions i to batch_end
        mask_positions = torch.arange(i, batch_end, device=ids.device)
        
        # The row indices in our mini-batch are 0, 1, ... current_batch_size-1
        row_indices = torch.arange(current_batch_size, device=ids.device)
        
        batch_ids[row_indices, mask_positions] = mask_id
        
        # 4. Forward pass (Gradient checkpointing saves memory but is slower; optional)
        with torch.no_grad():
            logits = model(input_ids=batch_ids, attention_mask=batch_amask).logits
        
        # 5. Extract the probabilities of the *true* wild-type tokens at the masked positions
        # Get logits for the specific masked positions: shape [current_batch_size, vocab_size]
        # We select the logits at the positions we masked [row_indices, mask_positions]
        masked_logits = logits[row_indices, mask_positions, :]
        
        # Log Softmax
        logp = torch.log_softmax(masked_logits, dim=-1)
        
        # Get the WT token IDs at these positions
        wt_tokens = ids[0, mask_positions]
        
        # Gather the log-prob of the WT token
        # shape: [current_batch_size]
        wt_logps_batch = logp.gather(1, wt_tokens.unsqueeze(1)).squeeze(1)
        
        all_log_probs.append(wt_logps_batch)

    # Concatenate all batches back into a single tensor of length L
    return torch.cat(all_log_probs)


def sequence_pll(input_ids, attention_mask, model, mask_id):
    
    logps = per_site_wt_logprobs(input_ids, attention_mask, model, mask_id, batch_size=32)
    pll = logps.sum()
    return pll


def batch_pll(sequences, tokenizer, model):
   
    
    input_ids, attention_mask = tokenizer.encode(sequences)
    mask_id = tokenizer.mask_id

    PLLs = []
    for i in range(len(sequences)):
        pll = sequence_pll(input_ids[i:i+1], attention_mask[i:i+1], model, mask_id)
        PLLs.append(pll.item())

    return PLLs


def llr_score(wt_seq, mutant_seq, tokenizer, model):

    sequences = [wt_seq, mutant_seq]

    input_ids, attention_mask = tokenizer.encode(sequences)
    mask_id = tokenizer.mask_id

    pll_wt = sequence_pll(
        input_ids[0:1],
        attention_mask[0:1],
        model, 
        mask_id
    )

    pll_mut = sequence_pll(
        input_ids[1:2],
        attention_mask[1:2],
        model, 
        mask_id
    )

    llr = pll_mut.item() - pll_wt.item()

    return llr



    


