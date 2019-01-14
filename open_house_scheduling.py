'''
    open_house_scheduling.py
    Cale Crowder
    January 14, 2019

    Attempts to schedule student-faculty interviews
'''

''' Imports'''
from __future__ import division
from __future__ import print_function
from ortools.sat.python import cp_model
import numpy as np

''' Constants '''
NUM_STUDENTS = 10
NUM_PROFESSORS = 5
NUM_INTERVIEWS = 3
all_students = range(NUM_STUDENTS)
all_professors = range(NUM_PROFESSORS)
all_interviews = range(NUM_INTERVIEWS)

MIN_INTERVIEWS = 1
MAX_INTERVIEWS = NUM_PROFESSORS

''' Randomly generate prefered matches for testing '''
def define_random_matches():
    pass

''' Make the matches '''
def main():
    
    # Creates the model.
    model = cp_model.CpModel()



    # Creates interview variables.
    # interview[(p, s, i)]: professor 'p' interviews student 's' for interview number 'i'
    interview = {}
    for p in all_professors:
        for s in all_students:
            for i in all_interviews:
                interview[(p, s,
                        i)] = model.NewBoolVar('interview_n%id%is%i' % (p, s, i))

    # Each student has no more than one interview at a time
    for p in all_professors:
        for i in all_interviews:
            model.Add(sum(interview[(p, s, i)] for s in all_students) <= 1)

    # Each professor has only one student per interview
    for s in all_students:
        for i in all_interviews:
            model.Add(sum(shifts[(p, s, i)] for p in all_professors) <= 1)

    # Ensure that no student or professor gets to many or too few interviews
    for s in all_students:
        num_interviews_stud = sum(
            interview[(p, s, i)] for p in all_professors for i in all_interviews)
        model.Add(MIN_INTERVIEWS <= num_interviews_stud)
        model.Add(num_interviews_stud <= MAX_INTERVIEWS)
    
    for p in all_professors:
        num_interviews_prof = sum(
            interview[(p, s, i)] for s in all_students for i in all_interviews)
        model.Add(MIN_INTERVIEWS <= num_interviews_prof)
        model.Add(num_interviews_prof <= MAX_INTERVIEWS)
    
    # Define the minimization of cost
    model.Minimize(
        sum(interview[(p, s, i)] for p in all_professors 
            for s in all_students for i in all_interview))
    
    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    solver.Solve(model)
    for i in all_interviews
        print(shift_requests[:][:][i])
        print()
    

    # Statistics.
    #print()
    #print('Statistics')
    #print('  - Number of shift requests met = %i' % solver.ObjectiveValue(),
    #      '(out of', num_nurses * min_shifts_per_nurse, ')')
    #print('  - wall time       : %f ms' % solver.WallTime())


if __name__ == '__main__':
    main()
