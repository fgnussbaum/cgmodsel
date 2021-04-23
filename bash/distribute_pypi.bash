#! /bin/bash

# This script builds a this module as a package and uploads it to pypi.org
# Provide your PYPI-credentials in ~/.pypirc

# https://packaging.python.org/tutorials/packaging-projects/

# Create venv
python -m venv ./venv-pypi
source venv-pypi/bin/activate

# Install necessary packs
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel twine

# Build dist
python setup.py install sdist bdist_wheel

#Upload to testpypi
python -m twine upload --repository pypi dist/*

# Clean
deactivate
rm -rf venv-pypi
rm -rf dist/*