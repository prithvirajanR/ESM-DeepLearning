import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

print("Attempting to import src.scoring...")
try:
    from src.scoring import (
        MaskedMarginalScoring,
        PseudoLogLikelihoodScoring,
        LogLikelihoodRatioScoring,
        MaskedLogLikelihoodRatioScoring,
        EmbeddingDistanceScoring,
        EnsembleMaskedLogLikelihoodRatioScoring,
        EntropyCorrectedLogLikelihoodRatioScoring,
        MutantMarginalScoring
    )
    print("SUCCESS: usage of src.scoring import successful.")
except Exception as e:
    print(f"FAILURE: Could not import src.scoring. Error: {e}")
    sys.exit(1)

print("Attempting to instantiate classes...")
try:
    s1 = EntropyCorrectedLogLikelihoodRatioScoring()
    print("SUCCESS: EntropyCorrectedLogLikelihoodRatioScoring instantiated.")
    s2 = MutantMarginalScoring()
    print("SUCCESS: MutantMarginalScoring instantiated.")
except Exception as e:
    print(f"FAILURE: Could not instantiate classes. Error: {e}")
    sys.exit(1)

print("All checks passed.")
