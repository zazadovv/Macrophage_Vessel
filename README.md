🧭 Repository Overview

This repository provides guidelines and part of the source code used in:

ShenGelashvili et al., 2024

Representative demo file outputs for testing the software.

Sample .stl files for 3D visualization. (Note: Most .stl source files are too large to host on GitHub. We have provided one fragmented ER example as a representative demo.)

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
