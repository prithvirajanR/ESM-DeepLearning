import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod

class ScoringMethod(ABC):
    """Abstract base class for scoring methods."""
    
    @abstractmethod
    def score(self, model_wrapper, wt_seq: str, mutant_code: str, mutant_seq: str) -> float:
        """
        Calculate fitness score for a mutant.
        
        Args:
            model_wrapper: Loaded ProteinModel instance
            wt_seq: Wild-type sequence
            mutant_code: Mutation code (e.g. "A123C" or "A123C:D124E")
            mutant_seq: Full mutant sequence
            
        Returns:
            float: Calculated score
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the scoring method."""
        pass

class MaskedMarginalScoring(ScoringMethod):
    """
    Scores based on the log-probability of the mutant amino acid(s) 
    at the mutated position(s), given the wild-type context.
    """
    @property
    def name(self) -> str:
        return "MaskedMarginal"

    def score(self, model_wrapper, wt_seq: str, mutant_code: str, mutant_seq: str) -> float:
        # Handle multiple mutations by averaging scores
        mutations = mutant_code.split(':')
        scores = []
        
        for mut in mutations:
            if len(mut) < 3: 
                continue # Skip invalid codes
                
            # Parse mutation: "A123C" -> wt="A", pos=122, mut="C"
            wt_aa = mut[0]
            pos = int(mut[1:-1]) - 1
            mut_aa = mut[-1]
            
            score = self._calculate_single_mm(model_wrapper, wt_seq, pos, mut_aa)
            scores.append(score)
            
        if not scores:
            return None
            
        return np.mean(scores)

    def _calculate_single_mm(self, model_wrapper, wild_type, mutant_pos, mutant_aa):
        tokenizer = model_wrapper.tokenizer
        model = model_wrapper.model
        device = model_wrapper.device
        
        # Tokenize wild type
        inputs = tokenizer(wild_type, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"][0]
        
        # Adjust position for CLS token
        target_index = mutant_pos + 1
        
        if target_index >= input_ids.shape[0] - 1:
            return None # Out of bounds

        # Mask the target position
        masked_input_ids = input_ids.clone()
        masked_input_ids[target_index] = tokenizer.mask_token_id
        
        # Forward pass
        with torch.no_grad():
            outputs = model(masked_input_ids.unsqueeze(0))
            logits = outputs.logits
            
        # Get log prob of mutant_aa
        mutant_token_id = tokenizer.convert_tokens_to_ids(mutant_aa)
        log_probs = F.log_softmax(logits[0, target_index], dim=-1)
        score = log_probs[mutant_token_id].item()
        
        return score

class PseudoLogLikelihoodScoring(ScoringMethod):
    """
    Scores based on the Pseudo-Log-Likelihood (PLL) of the full sequence.
    PLL = Sum(log P(x_i | x_{\i})) for all i.
    """
    @property
    def name(self) -> str:
        return "PLL"

    def score(self, model_wrapper, wt_seq: str, mutant_code: str, mutant_seq: str, **kwargs) -> float:
        if not mutant_seq:
            return None
        batch_size = kwargs.get('batch_size', 32)
        return self._calculate_pll_batched(model_wrapper, mutant_seq, batch_size=batch_size)

    def _calculate_pll(self, model_wrapper, sequence):
        # Deprecated wrapper
        return self._calculate_pll_batched(model_wrapper, sequence, batch_size=32)

    def _calculate_pll_batched(self, model_wrapper, sequence, batch_size=32):
        tokenizer = model_wrapper.tokenizer
        model = model_wrapper.model
        device = model_wrapper.device
        
        # Keep initial inputs on CPU
        inputs = tokenizer(sequence, return_tensors="pt")
        input_ids = inputs["input_ids"][0] # CPU tensor
        seq_len = input_ids.shape[0]
        
        # Identify positions to mask (exclude CLS/EOS)
        mask_indices = list(range(1, seq_len - 1))
        
        pll = 0.0
        
        # Pre-allocate batch tensor on CPU to avoid re-allocation
        batch_input_ids = input_ids.clone().unsqueeze(0).repeat(batch_size, 1)
        
        # Process in batches
        for i in range(0, len(mask_indices), batch_size):
            batch_indices = mask_indices[i : i + batch_size]
            current_batch_size = len(batch_indices)
            
            # Use slice if last batch is smaller
            current_batch_input = batch_input_ids[:current_batch_size].clone()
            
            # Vectorized masking on CPU (much faster than loop)
            # Create a range [0, 1, ..., N] and the indices [idx1, idx2, ...]
            rows = torch.arange(current_batch_size)
            cols = torch.tensor(batch_indices)
            current_batch_input[rows, cols] = tokenizer.mask_token_id
            
            # Move to GPU once
            current_batch_input = current_batch_input.to(device)
            
            with torch.no_grad():
                outputs = model(current_batch_input)
                logits = outputs.logits
            
            # Extract log probs (Logits stay on GPU for calculation, then move scalar result)
            # Targets: input_ids[batch_indices] -> shape [current_batch_size]
            targets = input_ids[batch_indices].to(device)
            
            # Advanced indexing: logits[rows, cols] gives [batch, vocab]
            masked_logits = logits[rows.to(device), cols.to(device)] # [batch, vocab]
            
            log_probs = F.log_softmax(masked_logits, dim=-1)
            
            # Select probability of the target token
            target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            # Sum up
            pll += target_log_probs.sum().item()
            
        return pll

class LogLikelihoodRatioScoring(ScoringMethod):
    """
    Scores based on Log-Likelihood Ratio (LLR).
    LLR = PLL(mutant) - PLL(wild_type)
    """
    def __init__(self):
        self.pll_scorer = PseudoLogLikelihoodScoring()
        self.wt_pll_cache = {}

    @property
    def name(self) -> str:
        return "LLR"

    def score(self, model_wrapper, wt_seq: str, mutant_code: str, mutant_seq: str) -> float:
        # 1. Get WT PLL (Computed once per Wildtype and cached)
        if wt_seq not in self.wt_pll_cache:
            self.wt_pll_cache[wt_seq] = self.pll_scorer._calculate_pll(model_wrapper, wt_seq)
            
        pll_wt = self.wt_pll_cache[wt_seq]
        
        # 2. Get Mutant PLL
        pll_mut = self.pll_scorer._calculate_pll(model_wrapper, mutant_seq)
        
        # 3. LLR = PLL_mut - PLL_wt
        return pll_mut - pll_wt

class MaskedLogLikelihoodRatioScoring(ScoringMethod):
    """
    Scores based on Masked Log-Likelihood Ratio (MLLR).
    MLLR = log P(mutant | context) - log P(wild_type | context)
    This normalizes the Masked Marginal score by the wild-type probability.
    """
    @property
    def name(self) -> str:
        return "MaskedLLR"

    def score(self, model_wrapper, wt_seq: str, mutant_code: str, mutant_seq: str) -> float:
        mutations = mutant_code.split(':')
        scores = []
        
        for mut in mutations:
            if len(mut) < 3: 
                continue
                
            wt_aa = mut[0]
            pos = int(mut[1:-1]) - 1
            mut_aa = mut[-1]
            
            score = self._calculate_single_mllr(model_wrapper, wt_seq, pos, wt_aa, mut_aa)
            scores.append(score)
            
        if not scores:
            return None
            
        return np.mean(scores)

    def _calculate_single_mllr(self, model_wrapper, wild_type, mutant_pos, wt_aa, mutant_aa):
        tokenizer = model_wrapper.tokenizer
        model = model_wrapper.model
        device = model_wrapper.device
        
        # Tokenize wild type
        inputs = tokenizer(wild_type, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"][0]
        
        # Adjust position for CLS token
        target_index = mutant_pos + 1
        
        if target_index >= input_ids.shape[0] - 1:
            return None # Out of bounds

        # Mask the target position
        masked_input_ids = input_ids.clone()
        masked_input_ids[target_index] = tokenizer.mask_token_id
        
        # Forward pass
        with torch.no_grad():
            outputs = model(masked_input_ids.unsqueeze(0))
            logits = outputs.logits
            
        # Get log prob of mutant and wt
        log_probs = F.log_softmax(logits[0, target_index], dim=-1)
        
        wt_token_id = tokenizer.convert_tokens_to_ids(wt_aa)
        mut_token_id = tokenizer.convert_tokens_to_ids(mutant_aa)
        
        return log_probs[mut_token_id].item() - log_probs[wt_token_id].item()

class EnsembleMaskedLogLikelihoodRatioScoring(ScoringMethod):
    """
    Scores based on Ensemble (Robust) MLLR.
    Calculates MLLR multiple times with random masking of the context.
    Score = Mean( LLR(mut | partial_context) )
    This forces the model to rely on global features rather than local neighbors.
    """
    def __init__(self, num_passes=5, mask_prob=0.15):
        self.num_passes = num_passes
        self.mask_prob = mask_prob

    @property
    def name(self) -> str:
        return "EnsembleMLLR"

    def score(self, model_wrapper, wt_seq: str, mutant_code: str, mutant_seq: str, **kwargs) -> float:
        mutations = mutant_code.split(':')
        
        # We process each mutation in the mutant (usually just 1)
        total_score = 0.0
        
        tokenizer = model_wrapper.tokenizer
        model = model_wrapper.model
        device = model_wrapper.device
        
        # Tokenize once
        inputs = tokenizer(wt_seq, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"][0]
        L = input_ids.shape[0]

        for mut in mutations:
            if len(mut) < 3: continue
            pos = int(mut[1:-1]) - 1
            wt_aa = mut[0]
            mut_aa = mut[-1]
            target_index = pos + 1 # offset for CLS
            
            if target_index >= L - 1: continue

            # Robustness Loop
            pass_scores = []
            for _ in range(self.num_passes):
                # Create a fresh mask each time
                masked_input_ids = input_ids.clone()
                
                # Mask target
                masked_input_ids[target_index] = tokenizer.mask_token_id
                
                # Randomly mask 15% of other tokens to force global reliance
                # (Simple bernoulli mask for this robust test)
                rand_mask = torch.rand(L).to(device) < self.mask_prob
                rand_mask[0] = False # Save CLS
                rand_mask[L-1] = False # Save EOS
                rand_mask[target_index] = True # Ensure target is masked
                
                masked_input_ids[rand_mask] = tokenizer.mask_token_id
                
                with torch.no_grad():
                    inputs = masked_input_ids.unsqueeze(0)
                    outputs = model(inputs)
                    logits = outputs.logits
                    
                log_probs = F.log_softmax(logits[0, target_index], dim=-1)
                
                wt_id = tokenizer.convert_tokens_to_ids(wt_aa)
                mut_id = tokenizer.convert_tokens_to_ids(mut_aa)
                
                score = log_probs[mut_id].item() - log_probs[wt_id].item()
                pass_scores.append(score)
            
            # Average score across passes
            total_score += np.mean(pass_scores)
            
        return total_score

class EntropyCorrectedLogLikelihoodRatioScoring(ScoringMethod):
    """
    Scores based on LLR weighted by local Entropy (Uncertainty).
    Score = LLR / (Entropy + epsilon)
    
    If the position is highly conserved (Low Entropy), a mutation is penalized MORE.
    If the position is flexible (High Entropy), the mutation penalty is dampened.
    """
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon

    @property
    def name(self) -> str:
        return "EntropyMLLR"

    def score(self, model_wrapper, wt_seq: str, mutant_code: str, mutant_seq: str, **kwargs) -> float:
        mutations = mutant_code.split(':')
        scores = []
        
        tokenizer = model_wrapper.tokenizer
        model = model_wrapper.model
        device = model_wrapper.device

        # Tokenize WT once
        inputs = tokenizer(wt_seq, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"][0]
        
        for mut in mutations:
            if len(mut) < 3: continue
            pos = int(mut[1:-1]) - 1
            wt_aa = mut[0]
            mut_aa = mut[-1]
            target_index = pos + 1
            
            if target_index >= input_ids.shape[0] - 1: continue

            masked_input_ids = input_ids.clone()
            masked_input_ids[target_index] = tokenizer.mask_token_id
            
            with torch.no_grad():
                outputs = model(masked_input_ids.unsqueeze(0))
                logits = outputs.logits # [1, L, V]
                
            # Focus on target position logits: [V]
            target_logits = logits[0, target_index]
            probs = F.softmax(target_logits, dim=-1)
            log_probs = F.log_softmax(target_logits, dim=-1)
            
            # 1. Calc LLR
            wt_id = tokenizer.convert_tokens_to_ids(wt_aa)
            mut_id = tokenizer.convert_tokens_to_ids(mut_aa)
            llr = log_probs[mut_id].item() - log_probs[wt_id].item()
            
            # 2. Calc Entropy: Sum( -p * log_p )
            # Be careful with 0 probs, but log_softmax is safe
            entropy = -torch.sum(probs * log_probs).item()
            
            # 3. Weight it
            # High Entropy (Unsure) -> Large Denom -> Smaller Score Magnitude
            # Low Entropy (Sure) -> Small Denom -> Amplified Score
            weighted_score = llr / (entropy + self.epsilon)
            scores.append(weighted_score)
            
        return np.mean(scores) if scores else None

class MutantMarginalScoring(ScoringMethod):
    """
    Scores based on P(Mutant | Context) ONLY.
    Ignores P(Wildtype).
    Useful when the Wildtype itself might be sub-optimal at that position.
    """
    @property
    def name(self) -> str:
        return "MutantMarginal"

    def score(self, model_wrapper, wt_seq: str, mutant_code: str, mutant_seq: str, **kwargs) -> float:
        # Similar to MaskedMarginal but explicitly separating it for clarity
        mutations = mutant_code.split(':')
        scores = []
        
        tokenizer = model_wrapper.tokenizer
        model = model_wrapper.model
        device = model_wrapper.device

        inputs = tokenizer(wt_seq, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"][0]
        
        for mut in mutations:
            if len(mut) < 3: continue
            pos = int(mut[1:-1]) - 1
            mut_aa = mut[-1]
            target_index = pos + 1
            
            masked_input_ids = input_ids.clone()
            masked_input_ids[target_index] = tokenizer.mask_token_id
            
            with torch.no_grad():
                outputs = model(masked_input_ids.unsqueeze(0))
                logits = outputs.logits
            
            log_probs = F.log_softmax(logits[0, target_index], dim=-1)
            mut_id = tokenizer.convert_tokens_to_ids(mut_aa)
            
            scores.append(log_probs[mut_id].item())
            
        return np.mean(scores) if scores else None
        


class EmbeddingDistanceScoring(ScoringMethod):
    """
    Scores based on Euclidean Distance of Embeddings.
    Score = -1 * L2_Distance(WT, Mutant)
    (We multiply by -1 so that Higher Score = Closer = Better, consistent with other metrics)
    """
    def __init__(self):
        self.wt_embed_cache = {}

    @property
    def name(self) -> str:
        return "EDS"

    def score(self, model_wrapper, wt_seq: str, mutant_code: str, mutant_seq: str, **kwargs) -> float:
        # 1. Get WT Embedding (Cached)
        if wt_seq not in self.wt_embed_cache:
            self.wt_embed_cache[wt_seq] = self._get_embedding(model_wrapper, wt_seq, pool=False)
            
        wt_emb = self.wt_embed_cache[wt_seq]
        
        # 2. Get Mutant Embedding
        mut_emb = self._get_embedding(model_wrapper, mutant_seq, pool=False)
        
        # 3. Calculate Euclidean Distance
        # Check shapes
        if wt_emb.shape == mut_emb.shape:
            # Use full tensor distance (Most Sensitive)
            dist = torch.dist(wt_emb, mut_emb, p=2).item()
        else:
            # Shape mismatch (Indel), fallback to Mean Pooling
            wt_mean = wt_emb.mean(dim=0)
            mut_mean = mut_emb.mean(dim=0)
            dist = torch.dist(wt_mean, mut_mean, p=2).item()
            
        # Return negative distance (Small dist = Big score)
        return -dist

    def _get_embedding(self, model_wrapper, sequence, pool=False):
        tokenizer = model_wrapper.tokenizer
        model = model_wrapper.model
        device = model_wrapper.device
        
        inputs = tokenizer(sequence, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1] # [1, SeqLen, Dim]
            
            # Remove Batch Dim
            embed = last_hidden_state.squeeze(0) # [SeqLen, Dim]
            
            # Remove CLS/EOS (First and Last)
            if embed.shape[0] > 2:
                embed = embed[1:-1]
            
            return embed
