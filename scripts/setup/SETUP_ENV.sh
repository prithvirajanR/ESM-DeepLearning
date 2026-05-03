#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
# -----------------------------------------------------------------------------
# SETUP_ENV.sh - Create Conda Environment on Raven
# -----------------------------------------------------------------------------
# Run this once on the Login Node to install PyTorch & friends.
# -----------------------------------------------------------------------------

export MY_PROJECT_DIR=$(pwd)
export ENV_DIR="$MY_PROJECT_DIR/env"

echo "ðŸ¦… Setting up Conda Environment in: $ENV_DIR"

# 1. Load Anaconda Module
module load anaconda/3/2021.11

# 2. Create Environment (in /ptmp)
# We use --prefix to install it directly in your project folder
echo "ðŸ“¦ Creating environment..."
conda create --prefix "$ENV_DIR" python=3.9 -y

# 3. Activate Environment
echo "ðŸ”Œ Activating..."
source activate "$ENV_DIR"

# 4. Install PyTorch (with CUDA support)
# Raven typically uses CUDA 11.8 or 12.1. We'll use a stable pytorch build.
echo "â¬‡ï¸ Installing PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 5. Install Other Libraries
echo "â¬‡ï¸ Installing Transformers & Tools..."
pip install transformers pandas numpy tqdm scipy matplotlib scikit-learn

echo "âœ… Environment Setup Complete!"
echo "   Path: $ENV_DIR"
