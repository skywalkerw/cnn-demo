# AGENTS.md - MNIST CNN Recognition Development Guide

## Overview
This is a PyTorch-based Convolutional Neural Network (CNN) for MNIST handwritten digit recognition with visualization capabilities. The project contains model training, prediction, and rich visualization tools to understand CNN behavior.

## Project Structure
```
mnist_recognition/
├── data/                   # Datasets directory
├── src/                    # Source code directory
│   ├── models/             # Model save directory
│   ├── model.py            # Model definition
│   ├── train.py            # Training script
│   ├── predict.py          # Prediction script
│   └── visualize_model.py  # Visualization tools
├── test_images/            # Test image directory
└── visualizations/         # Visualization results directory
```

## Build/Lint/Test Commands

### Environment Information
Current environment: ai_env (active)
Python version: 3.12.12
Platform: macOS (ARM64)

Installed packages (current environment):
- torch: 2.9.1
- torchvision: 0.24.1  
- numpy: 2.2.6
- matplotlib: 3.10.8
- pillow: 12.1.0
- networkx: 3.2.1 (if installed)
- tqdm: 4.65.0 (if installed)

### Environment Setup (if ai_env doesn't exist)
If you don't have the ai_env environment, you can create it with:
```bash
# Create the ai_env environment
conda create -n ai_env python=3.12

# Ensure pip is up to date for compatibility
python -m pip install --upgrade pip

# Activate the environment
conda activate ai_env

# Install required packages
pip install torch torchvision numpy matplotlib pillow networkx tqdm
```

### Using Current Environment (Recommended)
The project is designed to work with the ai_env environment that is already active. To ensure you're using the correct environment:
```bash
# Ensure your environment is active
conda activate ai_env

# Verify packages are available
python -c "import torch, torchvision, matplotlib, numpy, PIL; print('All required packages available')"
```

### Running Commands
```bash
# Training the model
cd mnist_recognition/src
python train.py  # Automatically downloads MNIST dataset on first run

# Making predictions
python predict.py ../test_images/digit_2.png  # Single image prediction
python predict.py  # Batch prediction on all images in test_images/

# Running visualizations
python visualize_model.py  # Generate all visualization results

# Running a single test (manual execution)
python -c "import torch; print('PyTorch version:', torch.__version__)"

# Running the model in evaluation mode
python -c "from model import DigitRecognizer; model = DigitRecognizer(); print('Model created successfully')"
```

### Testing Strategy
- The project doesn't have a formal unit test suite
- The primary validation happens through:
  - Training/testing accuracy during train.py execution
  - Manual testing of predict.py with different images
  - Visualization validation in visualize_model.py
- To test code changes, run the main scripts to ensure functionality
- For single-function testing, use Python one-liners with the -c flag
- Running a single test would typically involve importing and testing specific functions:
  ```bash
  # Example: test model creation
  python -c "from model import DigitRecognizer; model = DigitRecognizer(); print('Model loaded successfully')"
  
  # Example: test single prediction
  python -c "from predict import predict_digit; result = predict_digit('../test_images/digit_2.png', 'models/best_model.pth'); print(result)"
  ```

## Code Style Guidelines

### Imports
- Import standard library modules first, then third-party, then project-specific
- Group related imports and maintain alphabetical order within groups
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DigitRecognizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import ssl
```

### Formatting and Linting
- Use 4-space indentation
- Maximum line length of 120 characters
- Use hanging indents for long function calls or complex expressions
- Use blank lines appropriately to separate logical code blocks
- Include docstrings for all classes and functions with Args/Returns sections
- No automated linting tool is configured; follow existing code patterns for consistency
- Use flake8, pylint, or black if needed for consistency checking:
  ```bash
  # Optional: install and run linters
  pip install flake8 black isort pylint
  flake8 . --max-line-length=120
  black .
  isort .
  ```

### Types and Type Hints
- Use type hints for function parameters and return values
- Use Union types for optional parameters
- Use typing module for complex type definitions

### Types and Type Hints
- Use type hints for function parameters and return values
- Use Union types for optional parameters
- Use typing module for complex type definitions
```python
from typing import Tuple

def predict_digit(image_path: str, model_path: str, ax_row=None, show=False) -> Tuple[int, float]:
    """Predict digit from a single image
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model
        ax_row: Matplotlib subplot objects for visualization
        show: Whether to display processing steps
        
    Returns:
        Tuple containing predicted digit and confidence score
    """
```

### Naming Conventions
- Use snake_case for variables and functions
- Use PascalCase for class names
- Use UPPER_CASE for constants
- Use descriptive names that clearly indicate purpose
- Prefix private methods with underscore (_)

### Error Handling
- Use try/except blocks around file operations
- Include specific exception types when possible
- Provide meaningful error messages
- Gracefully handle missing model files or corrupted images
- Log errors appropriately without exposing sensitive information

### Comments and Documentation
- Include detailed comments for complex algorithms
- Document the purpose of each major code segment
- Use block comments for multi-line explanations
- Keep inline comments minimal and only when necessary
- Include Chinese explanations in comments (per project style)

### Model-Specific Patterns
- Use nn.Module inheritance for model classes
- Initialize layers in __init__ method
- Define forward pass in forward method
- Apply activation functions (ReLU) after conv and linear layers
- Include dropout layers to prevent overfitting
- Use log_softmax for numerical stability in output layer

### Device Handling
- Support CUDA, MPS (Apple Silicon), and CPU
- Check device availability in order: CUDA > MPS > CPU
- Move tensors and models to same device before operations
- Use non_blocking=True for GPU transfers when possible

### Data Processing
- Use torchvision.transforms for preprocessing
- Apply normalization with standard MNIST parameters (mean=0.1307, std=0.3081)
- Include padding, resizing, and contrast adjustments for input images
- Convert images to grayscale and normalize to match MNIST format

### Performance Optimization
- Use torch.compile for CUDA devices when available (PyTorch 2.0+)
- Implement mixed precision training with torch.cuda.amp
- Optimize DataLoader parameters based on hardware:
  - Adjust batch_size, num_workers, prefetch_factor based on available cores and memory
  - Use persistent_workers when num_workers > 0
- Use efficient tensor operations (view instead of reshape when possible)

## Best Practices for Modifications

### Model Changes
- Maintain the same model interface (input/output dimensions)
- Preserve dropout rates and layer connections
- Update model save/load paths if necessary
- Ensure new layers are initialized properly

### Training Modifications
- Maintain the Adam optimizer with learning rate of 0.001
- Keep cross entropy loss function
- Preserve batch size and epoch count defaults
- Maintain early stopping strategy

### Visualization Updates
- Preserve the six visualization types:
  1. Convolutional kernel visualization
  2. Weight distribution histograms
  3. Feature map displays
  4. Pixel contribution heatmaps
  5. Network architecture visualizations
  6. Activation distribution plots
- Ensure visualizations work with both individual and batch predictions

### File Handling
- Preserve the model save directory (models/)
- Maintain image processing pipeline for test images
- Keep automatic renaming functionality for high-confidence predictions
- Respect the existing data directory structure

## Hardware Optimization Notes
- For Apple M4 Pro: Use batch_size=768, num_workers=8, prefetch=4
- For CUDA: Enable mixed precision training with GradScaler
- For MPS: Use full precision as mixed precision is not recommended
- Monitor memory usage with larger batch sizes for GPU optimization

## Development Workflow
1. Create feature branches from master
2. Follow the existing code style and documentation standards
3. Test changes with both training and prediction workflows
4. Ensure visualizations continue to work as expected
5. Update documentation if making structural changes
6. Commit with descriptive messages explaining changes

## Additional Notes
- No specific Cursor rules (.cursor/rules/ or .cursorrules) were found in this repository
- No Copilot instructions (.github/copilot-instructions.md) were found in this repository
- When making changes, focus on maintaining the existing Chinese comments in the codebase as they provide important explanations
- The project uses detailed Chinese comments to explain complex algorithm implementations