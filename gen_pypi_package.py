import os
from distutils.dir_util import copy_tree
import shutil

# Get the current working directory
cwd = os.getcwd()

# Create a new temporary directory for the package
if not os.path.exists("tmp"):
    os.makedirs("tmp")
  
# Files and directories which should be included in the package  
pypi_folders = ["neroRL", "configs"]
pypi_files = ["enjoy.py", "eval.py", "train.py", "eval_checkpoints.py", "tune.py", "__init__.py"]

# Setup files to create the package
setup_files = ["setup.py", "LICENSE", "README.md", "requirements.txt", "MANIFEST.in"]

# Build the package directories
for folder in pypi_folders:
    copy_tree(folder, dst="tmp\\neroRL\\" + folder)

for file in pypi_files:
    shutil.copy(file, "tmp\\neroRL")

for file in setup_files:
    shutil.copy(file, "tmp")

# Fix imports for the module and delete pycache folders
for dname, dirs, files in os.walk("tmp\\neroRL"):
    if "__pycache__" in dname:
        shutil.rmtree(dname)
        continue
    for fname in files:
        fpath = os.path.join(dname, fname)
        with open(fpath) as f:
            s = f.read()
        s = s.replace("import neroRL", "import neroRL.neroRL")
        s = s.replace("from neroRL", "from neroRL.neroRL")
        with open(fpath, "w") as f:
            f.write(s)
  
# Build the package
os.chdir("./tmp")          
os.system("py -m build")
os.chdir(cwd)

# Copy the package to the current directory
copy_tree("tmp\\dist", "dist")

# Clean up
shutil.rmtree("tmp")