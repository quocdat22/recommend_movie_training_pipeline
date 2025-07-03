#!/bin/bash
#
# Pipeline Script to Update Movie Embeddings
#
# This script automates the process of finding new movies in the database
# (where embedding is NULL), generating their embeddings, and saving
# them back to Supabase. It's designed to be run from within the
# 'movie-model-training' directory.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Movie Embedding Update Pipeline ---"

# The script's directory
SCRIPT_DIR="$(dirname "$0")"

# Navigate to the source directory of the training script
echo "1. Navigating to the training script directory..."
cd "${SCRIPT_DIR}/src"

# Activate the Python virtual environment
# Assumes the venv is in the parent directory of 'src' (i.e., SCRIPT_DIR)
if [ -d "${SCRIPT_DIR}/venv" ]; then
    echo "2. Activating Python virtual environment..."
    source "${SCRIPT_DIR}/venv/bin/activate"
else
    echo "Warning: Virtual environment not found at ${SCRIPT_DIR}/venv. Assuming dependencies are installed globally."
fi


# Run the main training/embedding script
echo "3. Running the training script to process new movies..."
python train_new.py

# Deactivate the virtual environment only if it was activated
if [ -d "${SCRIPT_DIR}/venv" ]; then
    echo "4. Deactivating virtual environment..."
    deactivate
fi

echo "--- Pipeline Finished Successfully ---" 