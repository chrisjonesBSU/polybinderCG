import os
from setuptools import setup, find_packages


NAME = "polybinderCG"
here = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)

setup(name=NAME,
      version=about["__version__"],
      description='Create coarse-grained representation of polyaryletherketone systems',
      url='https://github.com/chrisjonesBSU/polybinderCG',
      author='Chris Jones',
      author_email='chrisjones4@u.boisestate.edu',
      license='GPLv3',
      packages=find_packages(),
      package_dir={'polybinderCG': 'polybinderCG'},
      package_data = {"polybinderCG": ["compounds/*"]},
      zip_safe=False,
)
