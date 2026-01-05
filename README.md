# FedHKD

FedHKD enables personalized federated learning by combining bidirectional decoupled knowledge distillation and gradient-decoupled training, with adaptive mechanisms to support heterogeneous models and improve convergence under non-IID data.

## Code Repository

This repository contains only the core implementation files for FedHKD. The complete codebase is based on the [PFLlib framework](https://github.com/TsingZ0/PFLlib).

## Installation

To run FedHKD, please follow these steps:

1. Clone and set up the PFLlib framework:
```bash
   git clone https://github.com/TsingZ0/PFLlib.git
   cd PFLlib
```

2. Install the required dependencies as specified in PFLlib's README.

3. Copy the FedHKD implementation files to the corresponding directories:
   - Copy `clientHkd.py` to `system/flcore/clients/`
   - Copy `serverhkd.py` to `system/flcore/servers/`
   - Copy `dot.py` to `system/flcore/optimizers/`
   - Replace the original `main.py` in the `system/` directory with the provided `main.py`

## File Descriptions

- **clientHkd.py**: Client-side implementation featuring bidirectional knowledge distillation with the Distillation-Oriented Trainer (DOT) optimizer
- **serverhkd.py**: Server-side implementation for heterogeneous model aggregation and coordination
- **dot.py**: Custom optimizer implementing the gradient-decoupled training mechanism
- **main.py**: Modified main script with FedHKD configuration options and heterogeneous model support

## Usage

Run FedHKD with the following command:
```bash
python main.py -algo FedHKD -m [LOCAL_MODEL] --global_model_type [GLOBAL_MODEL] -data [DATASET] [OTHER_OPTIONS]
```

Example:
```bash
python main.py -algo FedHKD -m MobileNet --global_model_type CNN -data Cifar10 -gr 500 -ls 5
```

## Citation

If you use this code in your research, please cite our paper (to be presented at IJCNN 2026).

## Acknowledgments

This implementation is built upon the [PFLlib framework](https://github.com/TsingZ0/PFLlib). We thank the authors for providing this excellent federated learning platform.
