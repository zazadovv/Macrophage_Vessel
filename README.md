3dpf Zebrafish Macrophage
Yellow: eGFP-KDEL; Magenta: cPla2-mKate2

<img width="148" height="134" alt="240420_mpeg_cpla_hypo_laser_r7_decon - Denoise_ai-MaxIP-3-1-4-2" src="https://github.com/user-attachments/assets/226dab50-f53e-44fa-98c0-c11845f475fe" />






🧭 Repository Overview

This repository provides guidelines and part of the source code used in:

Gelashvili et al., 2024

Representative demo file outputs for testing the software.


💻 System Requirements

Operating System

Windows 11 — any base or version (tested on Windows 11 v24H2 x64)

GPU Acceleration (Optional but Recommended) For GPU-accelerated processing, install the CUDA 13.0 toolkit and enable CuPy , which is later utilized by the pyclesperanto plugin | https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11

Recommended Hardware (Minimum)

RAM: 64 GB+ DDR4/DDR5 @ 3600 MT/s

GPU: NVIDIA RTX Quadro P4000 / GeForce RTX 3070 or similar (≥ 8 GB VRAM)

CPU: Intel Core i5-8400 or AMD Ryzen 5 2600 (or better)

📦 Package Installation

All required packages are listed in my_current_env_list.txt. To install dependencies: you can recreate the environment from the provided .yml file using Conda:

conda env create -f environment.yml conda activate Microscopy_Analysis_Advanced2 # example env name


Package Install: The Repository contains a complete package list. Please perform a PIP install for the required and imported package versions as indicated in my_current_env_list.txt and the script (3D) or FLIP.

🧰 Source Code Description

This repository includes the following primary components:

Macrophage Nuclear Mechanotransduction Analysis Script

Neutrophil Tracking Analysis Script 
Macrophage Tracking Analysis Script

![20230713_3dpf_tgNTR2 0_QUAS_LYZ_QF2_24hrs_DMSO_70kDaDextran_ISOHYPO_E8_F11](https://github.com/user-attachments/assets/364189f4-73ee-48eb-99d6-08a85443ccc3)

![20230713_3dpf_tg(NTR2 0_QUAS_LYZ_QF2)_24hrs_DMSO_70kDa Dextran_ISO+HYPO_E8_F11_napari_direct (1)](https://github.com/user-attachments/assets/b575ed30-bfa3-437d-ae36-9aab93a14b84)

Frame of Onset Permeability Analysis Script



MATLAB Area Under Curve Database Analysis Script


📋 Usage Guidelines
Anaconda Navigator
VS Code
.YML Package Install
Set Up the Environment
Install the required packages as described above (YAML or pip).
Use the recommended Python environment (e.g., Microscopy_Analysis_Advanced2).
Ensure the interpreter is set to the proper Python 3.9 environment.

Run the script to generate output.

The file has to be first opened in VS Code or Pycharm with Proper Python Interpreter (this is the Environment, 3.9 Python Base that should be installed from .yml file) - For example, Microscopy_Analysis_Advanced2 (My Current Environment).
