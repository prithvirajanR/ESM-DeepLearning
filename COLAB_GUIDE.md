# ☁️ How to Run on Google Colab

This guide explains how to run the Protein Fitness Pipeline on Google Colab using a free or paid GPU.

---

## 📂 Step 1: Prepare Google Drive

We use Google Drive to store your code and data so you don't lose it when Colab disconnects.

1.  **Create a folder** in your Google Drive named `ProteinProject`.
2.  **Upload the following** inside that folder:
    *   The `src/` folder (and all its contents).
    *   The `data/` folder (containing `A4_HUMAN_Seuma_2021.csv` etc).
    *   `requirements.txt`.
3.  Your Drive structure should look like this:
    ```
    My Drive/
    └── ProteinProject/
        ├── src/
        ├── data/
        └── requirements.txt
    ```

---

## 📓 Step 2: The Colab Notebook

Open a new [Google Colab Notebook](https://colab.research.google.com/) and create the following code cells.

### Cell 1: Setup & Connect Drive
*Run this cell first. It connects your Drive and installs libraries.*

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Go to your project folder
import os
# Change this path if you named your folder something else
os.chdir('/content/drive/MyDrive/ProteinProject') 

print(f"📂 Current Directory: {os.getcwd()}")
print("✅ Connected to Google Drive.")

# 3. Install Dependencies
!pip install -r requirements.txt
```

---

### Cell 2: The Safety Check (CPU)
*Run this to make sure everything is uploaded correctly.*

```python
# Run the verification script
!python -m src.verify_pipeline
```

**Expected Output**:
> `🎉 SUCCESS: The code is robust and ready for HPC.`

*(If it fails, check if you uploaded the `data` folder correctly!)*

---

### Cell 3: Run the Prediction (GPU)
*This is the main step. Make sure your Runtime is set to GPU.*
*(Go to `Runtime` -> `Change runtime type` -> Select `T4` or `A100`)*

```python
# Run the pipeline
# Note: Output goes directly to your Drive, so it's safe!
!python -m src.batch_scoring \
    --input_csv data/A4_HUMAN_Seuma_2021.csv \
    --output_csv results/scores_colab.csv \
    --method MLLR \
     --model facebook/esm2_t33_650M_UR50D
```

---

### Cell 4: Download Results (Optional)
Since we saved directly to Google Drive (`results/scores_colab.csv`), the file is **already** in your Drive! You can just go to drive.google.com and download it.

But if you want to inspect it right now in Colab:

```python
import pandas as pd
df = pd.read_csv("results/scores_colab.csv")
print(df.head())
```

---

## 💡 Important Tips

1.  **Keep the Tab Open**: If you close the Colab tab, the execution stops.
2.  **Save to Drive**: The command above uses `--output_csv` pointing to a location inside your mapped Drive. This is crucial. If you save it to `/content/`, it will disappear when the runtime disconnects.
3.  **A100 vs T4**:
    *   **T4 (Free)**: Good for `MLLR`. Okay for `PLL` with small batches (`--batch_size 1`).
    *   **A100 (Pro)**: Required for fast `PLL` runs (`--batch_size 32`).
