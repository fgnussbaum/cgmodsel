from setuptools import setup
from setuptools import find_packages

setup(name='cgmodsel',
      version='1.0',
      description='Algorithms for model selection of conditional Gaussian distributions',
      url='https://github.com/franknu/cgmodsel.git',
      author='Frank Nussbaum',
      author_email='frank.nussbaum@uni-jena.de',
      license='Public',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'matplotlib',
      ],
      zip_safe=False)

	  
# use e.g. python setup.py install
# or pip install . (from local directory)