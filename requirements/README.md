# Environments

`model.txt` records the package versions used by the H100 baseline environment. Install the PyTorch build compatible with the cluster CUDA driver and hardware.

`vcc-cli.txt` belongs in a separate Python 3.11 environment because the official CLI has a different Python requirement from the established CUDA environment.

`presentation.txt` contains only the packages used to build, inspect, and render the English strategy deck.
