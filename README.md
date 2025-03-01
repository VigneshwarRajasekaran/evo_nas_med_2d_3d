# Evolutionary Neural Architecture Search for 2D and 3D Medical Image Classification
[![Active Development](https://img.shields.io/badge/Maintenance%20Level-Actively%20Developed-brightgreen.svg)](https://gist.github.com/cheerfulstoic/d107229326a01ff0f333a1d3476e068d) [![Actively Maintained](https://img.shields.io/badge/Maintenance%20Level-Actively%20Maintained-green.svg)](https://gist.github.com/cheerfulstoic/d107229326a01ff0f333a1d3476e068d) 

## Overview 
- This repository contains the Python implementation of an Evolutionary Framework for 2D and 3D Medical Image Classification.
- The framework extends the DARTS search space for 2D and 3D Medical Image Classification and utilizes the MealPy library for the implementation of metaheuristics.
- For more details, please refer to our paper [Evolutionary Neural Architecture Search for 2D and 3D Medical Image Classification](https://www.iccs-meeting.org/archive/iccs2024/papers/148330121.pdf) by Muhammad Junaid Ali, Laurent Moalic, Mokhtar Essaid, and Lhassane Idoumghar. If you find this implementation helpful, please consider citing our work:

```bibtex
@inproceedings{ali2024evolutionary,
  title={Evolutionary Neural Architecture Search for 2D and 3D Medical Image Classification},
  author={Ali, Muhammad Junaid and Moalic, Laurent and Essaid, Mokhtar and Idoumghar, Lhassane},
  booktitle={International Conference on Computational Science},
  pages={131--146},
  year={2024},
  organization={Springer}
}
```

## Datasets Used

- [MedMNIST dataset](https://medmnist.com/)
- [BreakHIS dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)

## Repository Structure

```
Evolutionary_Zero_Cost_Medical_Classification2D/
│   ├── attentions.py           # Attention mechanisms used in architectures
│   ├── augment.py              # Data augmentation strategies
│   ├── augmentations.py        # Additional augmentation techniques
│   ├── auto_augment.py         # Automated data augmentation
│   ├── autoaugment.py          # Alternative auto-augmentation methods
│   ├── dataset.py              # Dataset handling and preprocessing
│   ├── evaluate.py             # Model evaluation scripts
│   ├── main_args.py            # Main script to run NAS experiments
│   ├── model.py                # Defines neural network architectures
│   ├── trainer.py              # Training pipeline for NAS models
│   ├── utils.py                # Utility functions for NAS execution
│   ├── search.py               # Evolutionary NAS search implementation
│   ├── visualize_results.py    # Visualization of NAS results
│   ├── logs/                   # Directory storing training logs
│   ├── results/                # Directory for storing model results
│
Evolutionary_Zero_Cost_Medical_Classification3D/
│   ├── dataset.py              # 3D medical dataset preprocessing
│   ├── evaluate.py             # Evaluation scripts for 3D models
│   ├── main_args.py            # Script for running 3D NAS
│   ├── model.py                # 3D neural network architectures
│   ├── trainer.py              # Training pipeline for 3D NAS models
│   ├── search.py               # Evolutionary NAS implementation for 3D
│   ├── visualize_results.py    # Result visualization for 3D models
│   ├── logs/                   # Logs generated during 3D training
│   ├── results/                # Directory for 3D NAS model results
│
├── How to Run.txt              # Instructions on running the experiments
├── README.md                   # This documentation file
├── requirements.txt            # List of required dependencies
├── LICENSE                     # License file for the project
