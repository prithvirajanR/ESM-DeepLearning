#!/bin/bash
# -----------------------------------------------------------------------------
# SETUP_ENV_FAST.sh - The "Speedy" Setup
# -----------------------------------------------------------------------------
# Use this if 'conda install pytorch' hangs forever.
# It uses 'pip' which is instant.
# -----------------------------------------------------------------------------

export MY_PROJECT_DIR=$(pwd)
export ENV_DIR="$MY_PROJECT_DIR/env"

echo "🦅 Setting up Conda Environment (Fast Lane) in: $ENV_DIR"

# 1. Load Anaconda
module load anaconda/3/2021.11

# 2. Create Base Environment (Python Only - Fast)
echo "📦 Creating environment..."
# If it exists, remove it first to be clean
# rm -rf "$ENV_DIR" 
conda create --prefix "$ENV_DIR" python=3.9 -y

# 3. Activate
echo "🔌 Activating..."
source activate "$ENV_DIR"

# 4. Install PyTorch with pip (Instant Solve)
echo "⬇️ Installing PyTorch (Pip)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Install Other Libraries
echo "⬇️ Installing Tools..."
pip install transformers pandas numpy tqdm scipy matplotlib scikit-learn seaborn

echo "✅ Environment Setup Complete!"
