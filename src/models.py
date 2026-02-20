from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

class ProteinModel:
    def __init__(self, model_name: str, cache_dir: str = "model_cache", use_compile: bool = True):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_compile = use_compile

    def load_model(self):
        """Loads the model and tokenizer."""
        raise NotImplementedError

    def get_logits(self, sequence: str):
        """Returns logits for a given sequence."""
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits
    
    def _try_compile(self):
        """Attempt to JIT compile the model for speedup. Falls back gracefully if unavailable."""
        if not self.use_compile:
            print("torch.compile disabled by user.")
            return
        
        # Check if torch.compile is available (PyTorch 2.0+)
        if not hasattr(torch, 'compile'):
            print("torch.compile not available (requires PyTorch 2.0+). Skipping compilation.")
            return
        
        try:
            print("Attempting torch.compile for 2-3x speedup...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("torch.compile successful!")
        except Exception as e:
            print(f"torch.compile failed: {e}")
            print("Continuing without compilation (model will still work, just slower).")

class ESM2Model(ProteinModel):
    def load_model(self):
        print(f"Loading ESM-2 model: {self.model_name}...")
        print(f"Using cache directory: {self.cache_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_name, 
            cache_dir=self.cache_dir, 
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()

class ESM1vModel(ProteinModel):
    def load_model(self):
        print(f"Loading ESM-1v model: {self.model_name}...")
        print(f"Using cache directory: {self.cache_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_name, 
            cache_dir=self.cache_dir, 
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()

def get_model(model_name: str, model_type: str = "esm2", cache_dir: str = "model_cache") -> ProteinModel:
    """
    Factory function to get a model instance.
    
    Args:
        model_name: Name of the model (e.g. "facebook/esm2_t30_150M_UR50D")
        model_type: Type of model ("esm2" or "esm1v")
        cache_dir: Directory to cache model weights
        use_compile: Whether to use torch.compile for speedup (requires PyTorch 2.0+)
        
    Returns:
        Instance of ProteinModel
    """
    if model_type == "esm2":
        return ESM2Model(model_name, cache_dir)
    elif model_type == "esm1v":
        return ESM1vModel(model_name, cache_dir)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
