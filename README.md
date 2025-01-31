Requirements:
- Python 3.6+
- numpy
- scipy
- GPy
- pyDOE
- emukit



# GokuEmu: A Cosmological Emulator Based on the Goku Simulation Suite

**GokuEmu** is a Gaussian process emulator for the nonlinear **matter power spectrum** trained on simulations from the **Goku suite**, using the **MF-Box** emulation technique (see matter_emu_mfbox: https://github.com/jibanCat/matter_emu_mfbox). Please refer to https://arxiv.org/abs/2501.06296 for details on the simulations and introduction to the emulator.

---

## **Installation**
### **1 Clone the Repository**
```bash
git clone https://github.com/astro-YYH/GokuEmu.git
cd GokuEmu
```

### **2 Create a Virtual Environment (Optional but recommended)**
Assuming conda is being used:
```bash
conda create -n goku-env python=3.8
conda activate goku-env
```

### **3 Install Dependencies**
GokuEmu requires the following **Python packages**:
- GPy
- matplotlib
- emukit

To install dependencies using pip:
```bash
pip install GPy matplotlib emukit
```
Dependencies of each package will be automatically installed using pip, such as numpy and cython for GPy.

### **4 Usage**

emulator/example.ipynb shows how to use GokuEmu.

### **Citation**
If you use GokuEmu or the Goku dataset in your research, please cite:
https://arxiv.org/abs/2501.06296 

### **License**
This project is licensed under the MIT License.