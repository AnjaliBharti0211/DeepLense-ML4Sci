# DeepLense ML4Sci

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Machine Learning solutions for gravitational lensing analysis as part of the **Google Summer of Code 2026** program with [ML4SCI](https://ml4sci.org/).

## Project Overview

This repository contains two main components:

1. **Computer Vision Assignment** - A PyTorch-based classifier for gravitational lensing images
2. **AgenticAI** - An intelligent agent for gravitational lensing simulations using Pydantic AI

## Project Structure

```
DeepLense-ML4Sci/
├── Computer Vision Assignment/
│   ├── main.py                              # Training pipeline
│   ├── gravitational_lensing_classifier.ipynb  # Jupyter notebook
│   └── requirements.txt                     # Dependencies
│
├── AgenticAI/
│   ├── DeepLenseSim/                        # Simulation framework
│   │   ├── deeplense/                       # Core simulation module
│   │   ├── Model_I/ to Model_IV/            # Different model configurations
│   │   └── setup.py                         # Installation script
│   │
│   └── deeplense_agent/                     # AI Agent
│       ├── src/deeplense_agent/             # Agent source code
│       ├── examples/                        # Usage examples
│       ├── tests/                           # Unit tests
│       └── pyproject.toml                   # Package configuration
│
├── requirements.txt                         # Root dependencies
├── .gitignore
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or conda
- Git
- (Optional) CUDA-capable GPU for faster training

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rakun-3045/DeepLense-ML4Sci.git
   cd DeepLense-ML4Sci
   ```

2. **Create a virtual environment**
   ```bash
   # Using venv
   python -m venv .venv

   # On Windows
   .venv\Scripts\activate

   # On Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # Install all dependencies
   pip install -r requirements.txt

   # Or install PyTorch with CUDA support first
   # Visit https://pytorch.org/get-started/locally/ for the correct command
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

---

## 1. Computer Vision Assignment

A deep learning pipeline for classifying gravitational lensing images into three categories:
- **no_substructure** (Class 0)
- **subhalo** (Class 1)
- **vortex** (Class 2)

### Dataset

The dataset consists of simulated gravitational lensing images in `.npy` format. Due to its size (~37,500 files), it's not included in the repository.

**Download options:**
- [Kaggle Dataset](https://www.kaggle.com/) - Search for "gravitational lensing classification"
- Contact ML4SCI for the official dataset

**Expected structure:**
```
dataset/
├── train/
│   ├── no_substructure/    # or 'no'
│   ├── subhalo/            # or 'sphere'
│   └── vortex/             # or 'vort'
└── val/
    ├── no_substructure/
    ├── subhalo/
    └── vortex/
```

### Training

```bash
cd "Computer Vision Assignment"

# Run training with default parameters
python main.py --data_dir ./dataset

# Run with custom parameters
python main.py --data_dir ./dataset --epochs 100 --batch_size 32 --lr 0.0001

# Dry run to verify pipeline (no dataset needed)
python main.py --dry-run
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./data` | Path to dataset directory |
| `--output_dir` | `./output` | Where to save model and plots |
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 64 | Batch size for training |
| `--lr` | 1e-4 | Learning rate |
| `--img_size` | 224 | Image resize dimension |
| `--val_split` | 0.2 | Validation split ratio |
| `--seed` | 42 | Random seed |
| `--dry-run` | False | Test pipeline without data |

### Model Architecture

- **Base Model**: ResNet-18 (trained from scratch)
- **Input**: 224x224 grayscale images (converted to 3-channel)
- **Output**: 3-class softmax probabilities
- **Augmentation**: 90° rotations, horizontal/vertical flips
- **Mixed Precision**: Automatic mixed precision training (AMP)

### Output

After training:
- `output/model.pth` - Best model checkpoint
- `output/roc_curve.png` - ROC curves for all classes

---

## 2. AgenticAI - DeepLense Agent

An intelligent agent for gravitational lensing simulations.

### Installation

```bash
cd AgenticAI/deeplense_agent

# Install the package
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

### Usage

**Python API:**
```python
from deeplense_agent import create_agent, SimulationConfig, ModelType

agent = create_agent(mock_mode=True)
config = SimulationConfig(model_type=ModelType.MODEL_I, num_images=10)

import asyncio
output = asyncio.run(agent.generate_from_config(config))
print(f"Generated {output.num_images_generated} images")
```

**CLI:**
```bash
# Generate from natural language
deeplense-agent generate "10 CDM lens images using Model I" --mock

# Interactive chat mode
deeplense-agent chat --mock

# Show available models
deeplense-agent info
```

### LLM Provider Configuration

```bash
# Set provider (groq, openai, anthropic)
export DEEPLENSE_PROVIDER=groq

# Set API keys as needed
export GROQ_API_KEY=your-api-key
export OPENAI_API_KEY=your-api-key
```

---

## References

- Varma et al. (2020): [arXiv:2005.05353](https://arxiv.org/pdf/2005.05353)
- Alexander et al. (2020): [DOI:10.3847/1538-4357/ab7925](https://doi.org/10.3847/1538-4357/ab7925)
- DeepLenseSim Papers: [arXiv:1909.07346](https://arxiv.org/abs/1909.07346), [arXiv:2008.12731](https://arxiv.org/abs/2008.12731), [arXiv:2112.12121](https://arxiv.org/abs/2112.12121)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ML4SCI](https://ml4sci.org/) - Machine Learning for Science
- [DeepLenseSim](https://github.com/mwt5345/DeepLenseSim) by Michael W. Toomey
- [lenstronomy](https://github.com/lenstronomy/lenstronomy) for gravitational lensing simulations
- Google Summer of Code 2026
