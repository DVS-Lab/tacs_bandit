#!/bin/bash

# Setup script for Two-Armed Bandit Task
# This script helps set up the environment with the correct Python version

echo "Two-Armed Bandit Task Setup"
echo "=========================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
echo "Current Python version: $python_version"

# Compare versions
if [ "$(echo "$python_version >= 3.11" | bc)" -eq 1 ]; then
    echo "Warning: Python 3.11+ detected"
    echo "PsychoPy 2023.2+ requires Python <3.11"
    echo ""
    echo "Options:"
    echo "1) Use PsychoPy 2023.1.3 (may have limited features)"
    echo "2) Install Python 3.10 and create a new virtual environment"
    echo "3) Use the Pygame alternative version (no PsychoPy required)"
    echo ""
    read -p "Choose option (1/2/3): " choice
    
    case $choice in
        1)
            echo "Installing PsychoPy 2023.1.3..."
            pip install psychopy==2023.1.3
            pip install -r requirements_minimal.txt
            ;;
        2)
            echo "Please install Python 3.10 first:"
            echo "  - Using pyenv: pyenv install 3.10.13"
            echo "  - Using conda: conda create -n bandit python=3.10"
            echo "  - Or download from python.org"
            echo ""
            echo "Then run this script again in the Python 3.10 environment"
            exit 0
            ;;
        3)
            echo "Installing Pygame alternative..."
            pip install -r requirements_pygame.txt
            echo "Use 'python bandit_pygame.py' to run the task"
            ;;
        *)
            echo "Invalid option"
            exit 1
            ;;
    esac
else
    echo "Python version compatible with PsychoPy 2023.2+"
    echo "Installing full requirements..."
    pip install -r requirements.txt
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data
mkdir -p logs
mkdir -p stimuli/images/flowers
mkdir -p stimuli/images/feedback

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Add stimulus images to stimuli/images/"
echo "2. Edit config.json to customize parameters"
echo "3. Run 'python bandit_main.py' to start the task"