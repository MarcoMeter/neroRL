# Installation

## Dependencies

- Python >= 3.9
- (Optional) CUDA >= 11.7

## Install PyTorch

Install PyTorch as described [here](https://pytorch.org/get-started/locally/).
neroRL is tested with PyTorch version 2.0.0 and CUDA 11.7

## Install neroRL

If the source code is not needed, simply install the PyPi package. Depending on the to be used environment add a tag (procgen, obstacle-tower, ml-agents, random-maze) for the extra requirements. This specification is due to the various gym versions that are utilized by those envionments.

`pip install neroRL[procgen]`

Otherwise clone this repository and use the setup.py

`git clone https://github.com/MarcoMeter/neroRL.git`

`cd neroRL`

`pip install -e .[procgen]`

## Obstacle Tower Environment

Unity environments, like Obstacle Tower, are provided as executables.
Obstacle Tower can be build form [source](https://github.com/Unity-Technologies/obstacle-tower-source) using the Unity engine or you can download builds from these links:

| *Platform*     | *Download Link*                                                                     |
| --- | --- |
| Linux (x86_64) | https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_linux.zip   |
| Mac OS X       | https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_osx.zip     |
| Windows        | https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_windows.zip |

For checksums on these files, see [this](https://storage.googleapis.com/obstacle-tower-build/v4.1/ote-v4.1-checksums.txt).
