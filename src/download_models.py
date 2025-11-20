from src.models import ESM2Model, ESM1vModel
import os

def download_all_models():
    cache_dir = "f:/ESM/model_cache"
    print(f"Downloading models to cache directory: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)

    # 1. ESM-2 (150M) - Used for main validation
    model_name_esm2 = "facebook/esm2_t30_150M_UR50D"
    print(f"\n--- Caching {model_name_esm2} ---")
    try:
        model = ESM2Model(model_name_esm2, cache_dir=cache_dir)
        model.load_model()
        print("✅ ESM-2 (150M) successfully cached.")
    except Exception as e:
        print(f"❌ Failed to cache ESM-2: {e}")

    # 2. ESM-1v (650M) - For comparison
    model_name_esm1v = "facebook/esm1v_t33_650M_UR90S_1"
    print(f"\n--- Caching {model_name_esm1v} ---")
    try:
        model = ESM1vModel(model_name_esm1v, cache_dir=cache_dir)
        model.load_model()
        print("✅ ESM-1v (650M) successfully cached.")
    except Exception as e:
        print(f"❌ Failed to cache ESM-1v: {e}")

    print("\nAll downloads complete. Models are saved locally.")

if __name__ == "__main__":
    download_all_models()
