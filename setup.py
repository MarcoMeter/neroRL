from setuptools import setup, find_packages

# Get install requirements from requirements.txt
install_requires = None
with open("requirements.txt") as file:
    install_requires = [module_name.rstrip() for module_name in file.readlines()]

setup(
    name="neroRL",
    version="0.0.1",
    packages=find_packages(),
    install_requires=install_requires,
)