from setuptools import setup, find_packages
import os
import sys
sys.path.insert(0, os.getcwd())
import neroRL

# Get current working directory
cwd = os.getcwd()

# Get install requirements from requirements.txt
install_requires = None
with open(cwd + "/requirements.txt") as file:
    install_requires = [module_name.rstrip() for module_name in file.readlines()]

# Get long description from README.md
long_description = ""
with open(cwd + "/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Set up package
setup(
    name="neroRL",
    version=neroRL.__version__,
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
    author="Marco Pleines",
    package_dir={'': '.'},
    packages=find_packages(where='.', include="neroRL*"),
    include_package_data=True,
    install_requires=install_requires,
    extras_require={
        ':sys_platform == "win32"': [ # Extra windows install requirements
            'windows-curses'
        ],
        ':"linux" in sys_platform': [ # Extra linux install requirements
            ''  
        ],
        "procgen":          ["gym==0.15.3", "procgen"],
        "obstacle-tower":   ["gym==0.18.3", "mlagents-envs==0.17.0", "protobuf==3.20.3"],
        "ml-agents":        ["gym==0.18.3", "mlagents-envs==0.28.0"],
        "random-maze":      ["gym==0.21.0", "mazelab @ git+https://github.com/zuoxingdong/mazelab@master", "scikit-image==0.18.0"],
    },
    entry_points={
        "console_scripts": [
            "ntrain=neroRL.train:main",
            "nenjoy=neroRL.enjoy:main",
            "neval=neroRL.eval:main",
            "ntune=neroRL.tune:main",
            "noptuna=neroRL.optuna:main",
            "neval-checkpoints=neroRL.eval_checkpoints:main"
        ],
    },
)