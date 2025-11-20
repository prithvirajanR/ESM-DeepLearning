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

    def score(self, model_wrapper, wt_seq: str, mutant_code: str, mutant_seq: str) -> float:
        if not mutant_seq:
            return None
        return self._calculate_pll(model_wrapper, mutant_seq)

    def _calculate_pll(self, model_wrapper, sequence):
        tokenizer = model_wrapper.tokenizer
        model = model_wrapper.model
        device = model_wrapper.device
        
        inputs = tokenizer(sequence, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"][0]
        seq_len = input_ids.shape[0]
        
        pll = 0.0
        
        # Iterate over sequence positions (excluding CLS/EOS)
        for i in range(1, seq_len - 1):
            target_id = input_ids[i].item()
            masked_input_ids = input_ids.clone()
            masked_input_ids[i] = tokenizer.mask_token_id
            
            with torch.no_grad():
                outputs = model(masked_input_ids.unsqueeze(0))
                logits = outputs.logits
                
            log_probs = F.log_softmax(logits[0, i], dim=-1)
            pll += log_probs[target_id].item()
            
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
        # Calculate or retrieve WT PLL
        if wt_seq not in self.wt_pll_cache:
            self.wt_pll_cache[wt_seq] = self.pll_scorer.score(model_wrapper, wt_seq, "", wt_seq)
            
        pll_wt = self.wt_pll_cache[wt_seq]
        pll_mut = self.pll_scorer.score(model_wrapper, wt_seq, mutant_code, mutant_seq)
        
        if pll_mut is None:
            return None
            
        return pll_mut - pll_wt
