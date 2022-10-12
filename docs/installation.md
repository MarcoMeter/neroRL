# Installation

## Dependencies

- Python >= 3.7
- (Optional) CUDA >= 11.1 <= 11.3

## Install PyTorch

Install PyTorch as described [here](https://pytorch.org/get-started/locally/).
neroRL is tested with PyTorch version 1.8.1 and CUDA 11.1

## Install neroRL

If the source code is not needed, simply install the PyPi package and specify the desired training environment (memory-gym, procgen, minigrid, ml-agents, obstacle-tower, ballet). This specification is due to the various gym versions that are utilized by those envionments.

`pip install neroRL[minigrid]`

Otherwise clone this repository and use the setup.py

`git clone https://github.com/MarcoMeter/neroRL.git`

`pip install -e .[minigrid]`

## Obstacle Tower Environment

Unity environments, like Obstacle Tower, are provided as executables.
Obstacle Tower can be build form [source](https://github.com/Unity-Technologies/obstacle-tower-source) using the Unity engine or you can download builds from these links:

| *Platform*     | *Download Link*                                                                     |
| --- | --- |
| Linux (x86_64) | https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_linux.zip   |
| Mac OS X       | https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_osx.zip     |
| Windows        | https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_windows.zip |

For checksums on these files, see [this](https://storage.googleapis.com/obstacle-tower-build/v4.1/ote-v4.1-checksums.txt).
