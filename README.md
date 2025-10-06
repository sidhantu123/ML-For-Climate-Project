# ML-For-Climate-Project

## Overview
This project is a machine learning solution built in Python. It includes data preprocessing, model training, evaluation, and predictions. The goal is to solve ____

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Installation](#installation)
  - [1. Cloning the Repository](#1-cloning-the-repository)
  - [2. Setting Up the Virtual Environment](#2-setting-up-the-virtual-environment)
  - [3. Installing Dependencies](#3-installing-dependencies)
  - [4. Setting Up Pre-Commit Hooks](#4-setting-up-pre-commit-hooks)
- [Usage](#usage)
- [Project Structure](#project-structure)
<!-- - [Contributing](#contributing)
- [License](#license) -->

## Features
- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Prediction and visualization

## Technologies Used
- Python 3.x
- [pandas, numpy, sklearn, pytorch]

## Getting Started
To get a local copy up and running, follow these steps.

## Installation

### 1. Cloning the Repository
```bash
git clone https://github.com/yourusername/skincare_ml.git
cd skincare_ml
```

### 2. Setting Up the Virtual Environment
It’s recommended to use a virtual environment to manage dependencies.

#### For macOS and Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

#### For Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Installing Dependencies
Once the virtual environment is activated, install the dependencies.
For production dependencies only:
```bash
pip install -r requirements.txt
```

For development dependencies (including linters and test tools):
```bash
pip install -r requirements-dev.txt
```
## Usage
1. **Activate the virtual environment** before running the code:
   - **macOS/Linux**: `source venv/bin/activate`
   - **Windows**: `venv\Scripts\activate`

2. **Run the main script** for data processing, model training, or predictions:
   ```bash
   python main.py
   ```

3. **Deactivate the virtual environment** when done:
   ```bash
   deactivate
   ```

### 4. Setting Up Pre-Commit Hooks
This project uses `pre-commit` hooks to automatically check code formatting and linting before each commit.

1. **Install `pre-commit`:**
   ```bash
   pip install pre-commit
   ```

2. **Install the Pre-Commit Hook:**
   After installing `pre-commit`, run:
   ```bash
   pre-commit install
   ```

3. **Verify the Hook Setup:**
   To test the hooks on all files, run:
   ```bash
   pre-commit run --all-files
   ```

## Project Structure
```
ML-For-Climate-Project/
├── data/                   # Dataset and raw data files
├── notebooks/              # Jupyter notebooks for EDA and experiments
├── src/                    # Source code for the project
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
├── main.py                 # Main script for running the project
├── scripts/                # scripts to extract data
├── requirements.txt        # Project dependencies
├── requirements-dev.txt    # Project development dependencies
└── README.md               # Project README
```
<!--
## Contributing
Contributions are welcome! Please open an issue to discuss what you would like to contribute. For major changes, open a pull request with detailed information.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. -->
