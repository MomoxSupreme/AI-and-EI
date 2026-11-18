# AI-and-EI-Scratch

> **Evaluating the Trade-off Between Predictive Performance and Environmental Impact in Deep Learning**

This project provides a modular framework to train, evaluate, and compare multiple neural network architectures on a single benchmark task: **audio classification using the DESED (Domestic Environment Sound Event Detection) dataset**.

Beyond standard predictive performance metrics, this tool focuses on **Green AI** by measuring and comparing the environmental footprint (computational cost, power consumption, energy per gflops) of each model throughout the training and inference phases.

---

## 📋 Table of Contents
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Results & Visuals](#-results--visuals)
- [License](#-license)

---

## 📂 Project Structure

```text
AI-and-EI-Scratch/
├── Models/
│   └── models.py           # Implements the various NN models in PyTorch from modular components
├── Utils/
│   ├── config_reader.py    # Reads and parses the config.ini file
│   ├── datasets.py         # Data preprocessing and formatting for each model
│   ├── metrics.py          # Implements evaluation metrics (accuracy, F1, energy/carbon metrics)
│   ├── scheduler.py        # Learning rate scheduling utilities
│   ├── stats.py            # Utility functions for model statistics (e.g., parameter count, FLOPs)
│   └── visualize.py        # Plotting and visualization utilities for benchmark results
├── evaluate/
│   └── evaluate.py         # Main script to generate comparison plots and final evaluation results
├── config.ini              # Hyperparameter configuration (epochs, learning rate, patience, etc.)
├── config.py               # Generates the default config.ini file
├── requirements.txt        # Python dependencies required for the project
├── test.py                 # Runs the inference phase on the trained models
├── training.py             # Trains the different neural network models
└── LICENSE                 # Project license
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd AI-and-EI
   git checkout scratch (only needed on this branch)
   ```

2. **Create and activate a virtual environment with conda (recommended):**
   ```bash
   conda create -n "name_of_your_env" #it is not needed to specify the version of python although it is recommended to do it
   conda activate "name_of_your_env" #conda desactivate "name_of_your_env" 
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Configuration

All hyperparameters and experiment settings are centralized in `config.ini`.

* **Customize parameters:** Edit `config.ini` directly to adjust the number of epochs, learning rate, early stopping patience, or dataset paths.
* **Reset to default:** If you modify or delete `config.ini`, run the following command to regenerate the default configuration file:
  ```bash
  python config.py
  ```

---

## 🚀 Usage

Just run :

```
python -m evaluate.evaluate
```
and that's it ! The visuals will be on a directory on in the AI-and-IE one, the results will centralized in results.csv.

---

## 📊 Results & Visuals

### Model Performance vs. Environmental Footprint

![Performance vs Environmental Impact](assets/performance_vs_impact.png)

*Comparison of classification accuracy against energy consumption ($\text{kWh}$) and $\text{CO}_2$ equivalent emissions.*

![Additional Charts](assets/additional_plot.png)

---

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
