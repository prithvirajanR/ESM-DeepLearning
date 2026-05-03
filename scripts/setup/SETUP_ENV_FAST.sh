#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
# -----------------------------------------------------------------------------
# SETUP_ENV_FAST.sh - The "Speedy" Setup
# -----------------------------------------------------------------------------
# Use this if 'conda install pytorch' hangs forever.
# It uses 'pip' which is instant.
# -----------------------------------------------------------------------------

export MY_PROJECT_DIR=$(pwd)
export ENV_DIR="$MY_PROJECT_DIR/env"

echo "ðŸ¦… Setting up Conda Environment (Fast Lane) in: $ENV_DIR"

# 1. Load Anaconda
module load anaconda/3/2021.11

# 2. Create Base Environment (Python Only - Fast)
echo "ðŸ“¦ Creating environment..."
# If it exists, remove it first to be clean
# rm -rf "$ENV_DIR" 
conda create --prefix "$ENV_DIR" python=3.9 -y

# 3. Activate
echo "ðŸ”Œ Activating..."
source activate "$ENV_DIR"

# 4. Install PyTorch with pip (Instant Solve)
echo "â¬‡ï¸ Installing PyTorch (Pip)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Install Other Libraries
echo "â¬‡ï¸ Installing Tools..."
pip install transformers pandas numpy tqdm scipy matplotlib scikit-learn seaborn

echo "âœ… Environment Setup Complete!"
