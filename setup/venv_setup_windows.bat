

::  Setup a python virtual environment
::  Assumes installation of Python 3.6.9
::    and virtualenv
::  The directory we want to put the virtual environment in should be empty
::  Run as an administrator
::
::  Inputs:
::    %1 - directory where the virtual env should reside


set dir=%1

echo "%dir%"
python -m venv "%dir%"
call %dir%\Scripts\activate


python -m pip install --upgrade pip
pip install numpy==1.16.1
pip install ortools==6.10.6025
pip install pyaml
pip install reportlab==3.5.13
pip install PyQt5
pip install pyinstaller

