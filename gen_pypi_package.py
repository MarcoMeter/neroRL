import os
from distutils.dir_util import copy_tree
import shutil

cwd = os.getcwd()
if not os.path.exists("tmp"):
    os.makedirs("tmp")
    
copy_tree("neroRL", "tmp\\neroRL\\neroRL")
copy_tree("configs", "tmp\\neroRL\\configs")
shutil.copy("enjoy.py", "tmp\\neroRL")
shutil.copy("eval.py", "tmp\\neroRL")
shutil.copy("train.py", "tmp\\neroRL")
shutil.copy("eval_checkpoints.py", "tmp\\neroRL")
shutil.copy("eval.py", "tmp\\neroRL")
shutil.copy("tune.py", "tmp\\neroRL")
shutil.copy("neroRL\\__init__.py", "tmp\\neroRL")
shutil.copy("LICENSE", "tmp")
shutil.copy("setup.py", "tmp")
shutil.copy("pyproject.toml", "tmp")
shutil.copy("requirements.txt", "tmp")
shutil.copy("README.md", "tmp")
shutil.copy("MANIFEST.in", "tmp")

for dname, dirs, files in os.walk("tmp/neroRL"):
    for fname in files:
        fpath = os.path.join(dname, fname)
        with open(fpath) as f:
            s = f.read()
        s = s.replace("import neroRL", "import neroRL.neroRL")
        s = s.replace("from neroRL", "from neroRL.neroRL")
        with open(fpath, "w") as f:
            f.write(s)
  
os.chdir("./tmp")          
os.system("py -m build")
os.chdir(cwd)

copy_tree("tmp\\dist", "dist")
shutil.rmtree("tmp")