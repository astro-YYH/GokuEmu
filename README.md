# GokuEmu: A Cosmological Emulator Based on the Goku Simulation Suite

GokuEmu is a Gaussian process emulator for the nonlinear matter power spectrum, trained on simulations from the Goku suite using the MF-Box emulation technique (matter_emu_mfbox repository: https://github.com/jibanCat/matter_emu_mfbox).

For details on the simulations and the emulator, refer to our paper:  
https://arxiv.org/abs/2501.06296

---

## Installation

### 1 Clone the Repository 
```bash
git clone https://github.com/astro-YYH/GokuEmu.git
cd GokuEmu
```

### 2. Create a Virtual Environment (Optional but recommended)
If using conda:
```bash
conda create -n goku-env python=3.8
conda activate goku-env
```

### 3 Install Dependencies 
GokuEmu requires the following **Python packages**:
- GPy
- matplotlib
- emukit

To install dependencies via pip:
```bash
pip install GPy matplotlib emukit
```
Other required dependencies (e.g., numpy, cython) will be installed automatically.

---

## Usage

A brief example is provided in `emulator/example.ipynb`, demonstrating how to use GokuEmu for matter power spectrum predictions.

---

## Citation
If you use GokuEmu or the Goku dataset in your research, please cite:  
https://arxiv.org/abs/2501.06296

--- 

## **License**
This project is licensed under the MIT License.