from setuptools import setup, find_packages
import os

# Get current working directory
cwd = os.getcwd()

# Get install requirements from requirements.txt
install_requires = None
with open(cwd + "\\requirements.txt") as file:
    install_requires = [module_name.rstrip() for module_name in file.readlines()]

# Get long description from README.md
long_description = ""
with open(cwd + "\\README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Set up package
setup(
    name="neroRL",
    version="1.0.0",
    description="A library for Deep Reinforcement Learning (PPO) in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MarcoMeter/neroRL",
    keywords = ["Deep Reinforcement Learning", "PyTorch", "Proximal Policy Optimization", "PPO", "Recurrent", "Recurrence", "LSTM", "GRU"],
    project_urls={
        "Github": "https://github.com/MarcoMeter/neroRL",
        "Bug Tracker": "https://github.com/MarcoMeter/neroRL/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    autor="Marco Pleines",
    package_dir={'': '.'},
    packages=find_packages(where='.', include="neroRL*"),
    python_requires=">=3.6",
    include_package_data=True,
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "ntrain=neroRL.train:main",
            "nenjoy=neroRL.enjoy:main",
            "neval=neroRL.eval:main",
            "ntune=neroRL.tune:main",
            "neval-checkpoints=neroRL.eval_checkpoints:main"
        ],
    },
)