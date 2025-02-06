# Environment Setup Guide

## 1. Repository Setup

1. Fork the Lag-Llama repository on GitHub:
   - Go to https://github.com/time-series-foundation-models/lag-llama
   - Click "Fork" button
   - Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/lag-llama.git
   cd lag-llama
   git remote add upstream https://github.com/time-series-foundation-models/lag-llama.git
   ```

## 2. Development Environment (M3 Mac)

1. Create conda environment:
   ```bash
   # Install Miniforge for M1/M2/M3 if not installed
   # https://github.com/conda-forge/miniforge#miniforge3
   
   conda create -n lagllama-dev python=3.10
   conda activate lagllama-dev
   ```

2. Install dependencies:
   ```bash
   pip install -r tech_requirements_mac.txt
   ```

3. Set up pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. Create development config:
   ```bash
   cp config.yaml config.dev.yaml
   # Adjust parameters for development
   ```

## 3. Production Environment (Windows + RTX 4080)

1. Create conda environment:
   ```bash
   conda create -n lagllama-prod python=3.10
   conda activate lagllama-prod
   ```

2. Install dependencies:
   ```bash
   pip install -r tech_requirements.txt
   ```

3. Create production config:
   ```bash
   cp config.yaml config.prod.yaml
   # Adjust parameters for production
   ```

## 4. Project Structure

```
lag-llama/
├── data/                  # Data storage
├── models/               # Saved models
├── notebooks/           # Jupyter notebooks
│   ├── development/    # Development experiments
│   └── production/     # Production analysis
├── src/                 # Source code
│   ├── data/           # Data processing
│   ├── models/         # Model definitions
│   └── utils/          # Utilities
├── config.dev.yaml     # Development configuration
├── config.prod.yaml    # Production configuration
├── tech_requirements.txt      # Production requirements
└── tech_requirements_mac.txt  # Development requirements
```

## 5. Development Workflow

1. Create feature branch:
   ```bash
   git checkout -b feature/new-feature
   ```

2. Sync with upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

3. Push changes:
   ```bash
   git push origin feature/new-feature
   ```

## 6. Production Deployment

1. Create release branch:
   ```bash
   git checkout -b release/v1.x.x
   ```

2. Tag release:
   ```bash
   git tag -a v1.x.x -m "Release v1.x.x"
   git push origin v1.x.x
   ``` 