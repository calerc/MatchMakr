#!/bin/bash

# Setup a python virtual environment
# Assumes installation of Python 3.6.9
#   and virtualenv
# The directory we want to put the virtual environment in should be empty
# 
# Inputs:
#   $1 - directory where the virtual env should reside
#   $2 - if y, then install optional libraries

# Setup the directory and start the venv
if [ -d $1 ]; then
        raise error "Directory already exists.  Refusing to overwrite"
fi

python3 -m venv $1
cd $1
source ./bin/activate

# Install the necessary libraries
pip install --upgrade pip
pip install numpy==1.16.1
pip install ortools==6.10.6025
pip install pyaml
pip install reportlab==3.5.13
pip install PyQt5


if [ $2 == "y" ]; then
        pip install pyinstaller
fi








