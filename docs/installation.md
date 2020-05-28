# Installation

## Dependencies

- Python 3.6.*
- (Optional) CUDA 10.1

## Setting up the python environment

To install the necessary python packages choose one of the available requirements.txt files like:

`pip install -r requirements_linux.txt`

`pip install -r requirements_linux_gpu.txt`

`pip install -r requirements_windows.txt`

`pip install -r requirements_windows_gpu.txt`

It should not be an issue to install a newer [PyTorch](https://pytorch.org/get-started/locally/) version or to use Python v3.7.
However, the version of Unity ML-Agents (mlagents_envs==0.10.0) and Gym-Minigrid (gym-minigrid==1.0.1) should not be changed, because this framework implements distinct wrappers which are not likely to support any other version.
For MacOS, remove the lines for torch and torchvision from the requirements.txt and follow the instructions from [PyTorch](https://pytorch.org/get-started/locally/).

## Obstacle Tower Environment

Unity environments, like Obstacle Tower, are provided as executables.
Obstacle Tower can be build form [source](https://github.com/Unity-Technologies/obstacle-tower-source) using the Unity engine or you can download builds from these links:

| *Platform*     | *Download Link*                                                                     |
| --- | --- |
| Linux (x86_64) | https://storage.googleapis.com/obstacle-tower-build/v3.1/obstacletower_v3.1_linux.zip   |
| Mac OS X       | https://storage.googleapis.com/obstacle-tower-build/v3.1/obstacletower_v3.1_osx.zip     |
| Windows        | https://storage.googleapis.com/obstacle-tower-build/v3.1/obstacletower_v3.1_windows.zip |

For checksums on these files, see [this](https://storage.googleapis.com/obstacle-tower-build/v3.1/ote-v3.1-checksums.txt).
