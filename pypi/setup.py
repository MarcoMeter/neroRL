from setuptools import setup, find_packages

setup(
    name="neroRL",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["numpy", "torch", "gym", "ruamel.yaml", "opencv-python", "matplotlib", "pandas", "scipy", "docopt", "tensorboard==2.0.2", "gym-minigrid==1.0.2", "procgen", "dm-env==1.5", "pycolab==1.2"],
)