# Installation

## Dependencies

- Python >= 3.9
- (Optional) CUDA 11.7

## Install PyTorch

Install PyTorch as described [here](https://pytorch.org/get-started/locally/).
neroRL is tested with PyTorch version 2.0.1

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`

## Install neroRL

`pip3 install neroRL`

## Install neroRL for different environments that have conflicting dependencies

`pip3 install neroRL[ml-agents]`
`pip3 install neroRL[obstacle-tower]`
`pip3 install neroRL[procgen]`

## Install from source

```
git clone https://github.com/MarcoMeter/neroRL
cd neroRL
pip3 install -e .
```

## Obstacle Tower Environment

Unity environments, like Obstacle Tower, are provided as executables.
Obstacle Tower can be build form [source](https://github.com/Unity-Technologies/obstacle-tower-source) using the Unity engine or you can download builds from these links:

| *Platform*     | *Download Link*                                                                     |
| --- | --- |
| Linux (x86_64) | https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_linux.zip   |
| Mac OS X       | https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_osx.zip     |
| Windows        | https://storage.googleapis.com/obstacle-tower-build/v4.1/obstacletower_v4.1_windows.zip |

For checksums on these files, see [this](https://storage.googleapis.com/obstacle-tower-build/v4.1/ote-v4.1-checksums.txt).
