from setuptools import setup, find_packages

# Get install requirements from requirements.txt
install_requires = None
with open("requirements.txt") as file:
    install_requires = [module_name.rstrip() for module_name in file.readlines()]

long_description = ""
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neroRL",
    version="0.0.1",
    description="A library for Reinforcement Learning in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MarcoMeter/neroRL",
    project_urls={
        "Bug Tracker": "https://github.com/MarcoMeter/neroRL/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": ""},
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=install_requires
)