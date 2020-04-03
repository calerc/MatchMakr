
# Use this script to freeze (compile) MatchMakr from this directory
# Resulting executable is in ./dist
# Ensure that the MatchMakr virtual environment is activated before freezing


pyinstaller --onefile ../src/MatchMakr.py
