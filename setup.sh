#!/bin/bash

# Setup script for Python environment and dependencies
set -e  # Exit on any error

echo "Starting setup process..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Skipping package installation."
fi

# Clone necessary GitHub repositories
echo "Cloning GitHub repositories..."

# Add your specific repository URLs here
# Example repositories - replace with actual ones you need
REPOS=(
    "https://github.com/KMnP/vpt.git"
    "https://github.com/facebookresearch/perception_models.git"
    "https://github.com/TooTouch/VPT.git"
    ""
)

for repo in "${REPOS[@]}"; do
    repo_name=$(basename "$repo" .git)
    if [ ! -d "$repo_name" ]; then
        echo "Cloning $repo_name..."
        git clone "$repo"
    else
        echo "$repo_name already exists, skipping..."
    fi
done

echo "Setup completed successfully!"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To deactivate the virtual environment, run: deactivate"
