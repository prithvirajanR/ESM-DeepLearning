from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

class ProteinModel:
    def __init__(self, model_name: str, cache_dir: str = "f:/ESM/model_cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Loads the model and tokenizer."""
        raise NotImplementedError

    def get_logits(self, sequence: str):
        """Returns logits for a given sequence."""
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits

class ESM2Model(ProteinModel):
    def load_model(self):
        print(f"Loading ESM-2 model: {self.model_name}...")
        print(f"Using cache directory: {self.cache_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, cache_dir=self.cache_dir).to(self.device)
        self.model.eval()

class ESM1vModel(ProteinModel):
    def load_model(self):
        print(f"Loading ESM-1v model: {self.model_name}...")
        print(f"Using cache directory: {self.cache_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, cache_dir=self.cache_dir).to(self.device)
        self.model.eval()

def get_model(model_name: str, model_type: str = "esm2", cache_dir: str = "f:/ESM/model_cache") -> ProteinModel:
    """
    Factory function to get a model instance.
    
    Args:
        model_name: Name of the model (e.g. "facebook/esm2_t30_150M_UR50D")
        model_type: Type of model ("esm2" or "esm1v")
        cache_dir: Directory to cache model weights
        
    Returns:
        Instance of ProteinModel
    """
    if model_type == "esm2":
        return ESM2Model(model_name, cache_dir)
    elif model_type == "esm1v":
        return ESM1vModel(model_name, cache_dir)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
