#!/bin/bash
set -e

echo "Checking for Conda installation..."

# Check if conda is already installed
if command -v conda &> /dev/null; then
    echo "Conda is already installed. Skipping installation."
else
    echo "Conda not found. Installing Miniconda..."

    # Download Miniconda for macOS (ARM or Intel)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        ARCH=$(uname -m)
        if [[ "$ARCH" == "arm64" ]]; then
            URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else
            URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        fi
    else
        # Default to Linux version
        URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    fi

    echo "Downloading Miniconda from $URL..."
    wget -O miniconda.sh "$URL"

    # Install Miniconda silently to ~/miniconda
    bash miniconda.sh -b -p $HOME/miniconda

    # Remove installer after installation
    rm miniconda.sh

    echo "Miniconda installed successfully."

    # Add conda to PATH
    export PATH="$HOME/miniconda/bin:$PATH"

    # Initialize Conda
    conda init bash
    source ~/.bashrc || source ~/.zshrc

    #get mamba
    conda install -n base -c conda-forge mamba

    #ensure that flexible channels are used
    conda config --set channel_priority flexible

    echo "Conda is now available. Restart your terminal or run:"
    echo "  source ~/.bashrc  (or source ~/.zshrc for zsh users)"
fi

# Ensure Conda is available in the current session
export PATH="$HOME/miniconda/bin:$PATH"

echo "Conda installation complete. You can now use Conda."
echo "To verify, run: conda --version"
