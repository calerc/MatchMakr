[logo]: https://github.com/calerc/MatchMakr/blob/master/src/MatchMakr_Ico.ico "MatchMakr Logo"

![alt text][logo]
# MatchMakr

## Introduction
MatchMakr is a program designed to match interviewers with interviewees.  Originally, it was designed for matching prospective graduate students with faculty.  So, the source code calls interviewees "students" and interviewers "faculty."

MatchMakr was built on top of Google OR-Tools

## Setup
To work with MatchMakr, we recommend using a virtual environment.  Scripts to set up virtual environments can be found in the setup folder.  Run the scripts, and pass them the directory where you want to create the virtual environment.  After the virtual environment has been setup, you can activate it and begin working with the MatchMakr source code.

## Freezing (as in making an executable)
MatchMakr can be frozen using PyInstaller.  Scripts are provided in the Setup folder.  After freezing, the executable can be found at setup/dist/MatchMakr.

Note that freezing is not necessary to run MatchMakr

## Using
Compiled executables and documentation can be found at the [website](https://sites.google.com/case.edu/matchmakr/home)
Documentation includes a video of how to use MatchMakr

## Required Inputs:
The bare minimum files that can be provided to MatchMakr are as follows:
1. interview_times.csv - a list of interview times for printing on the schedule
2. student_preferences.csv - a spreadsheet that contains information about:
    1. Last Name
    2. First Name
    3. Full Name
    4. Track (a common interest)
    5. Professors that the student wants to interview with (ordered list)
3. faculty_preferences.csv - a spreadsheet that contains information about faculty:
    1. Last Name
    2. First Name
    3. Full Name
    4. Track (a common interest)
    5. If the professor is available to interview
    6.If the professor is recruiting (optional)
    7. An ordered list of professors that a professor is like (ordered list, optional)
4. Professors that the student wants to interview with (ordered list)

In addition to these files, MatchMakr can accept the following files:
1. faculty_availability.csv - a matrix specifying the avilability of professors with ones and zeros
2. student_availability.csv - a matrix specifying the avilability of students with ones and zeros

Not that MatchMakr expects faculty_availability.csv when using the default settings, but not student_availability.csv

## Where to begin
The GUI is defined in `MatchMakr.py`

The matchmaking backend is defined in `match_maker.py`

