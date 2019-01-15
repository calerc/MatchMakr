from __future__ import division
from __future__ import print_function

'''
    open_house_scheduling.py
    Cale Crowder
    January 14, 2019

    Attempts to schedule student-faculty interviews
'''

''' Imports'''
from ortools.sat.python import cp_model
import csv
from os import path
import numpy as np

''' Constants '''
NUM_STUDENTS = 43 #43
NUM_PROFESSORS = 31 #31
NUM_INTERVIEWS = 2
all_students = range(NUM_STUDENTS)
all_professors = range(NUM_PROFESSORS)
all_interviews = range(NUM_INTERVIEWS)

MIN_INTERVIEWS = 1
MAX_INTERVIEWS = 3



''' Load the data '''
def load_data():
    
    # Constants
    PATH = "/home/cale/Desktop/open_house/"
    STUDENT_PREF = "Student_Preferences.csv"
    FACULTY_PREF = "faculty_preferences.csv"
    
    # Read the data from the CSV
    student_pref = []
    faculty_pref = []
    
    with open(path.join(PATH, STUDENT_PREF), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            student_pref.append(row)
            
    with open(path.join(PATH, FACULTY_PREF), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            faculty_pref.append(row)

    # Seperate the data and the names
    faculty_names = faculty_pref[0][1:]
    
    student_names = []
    student_data = []
    faculty_data = []
    for student in faculty_pref[1:]:
        student_names.append(student[0])
        faculty_data.append(student[1:])  

    for student in student_pref[1:]:
        student_data.append(student[1:])

    student_pref = np.asarray(student_data, dtype=np.float) + 1
    faculty_pref = np.asarray(faculty_data, dtype=np.float) + 1

    # Return
    return student_pref, faculty_pref, faculty_names, student_names


''' Randomly generate prefered matches for testing '''
def define_random_matches():
    prof_pref_4_students = np.random.randint(1, high=NUM_STUDENTS, size=(NUM_STUDENTS, NUM_PROFESSORS))
    stud_pref_4_profs = np.random.randint(1, high=NUM_PROFESSORS, size=(NUM_STUDENTS, NUM_PROFESSORS))
    print(prof_pref_4_students)
    print(stud_pref_4_profs)

    #prof_pref_4_students = np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2]])*10
    #stud_pref_4_profs = np.array([[2, 2, 1], [1, 2, 1], [2, 2, 1]])
    
    cost_matrix = prof_pref_4_students * stud_pref_4_profs
    print(cost_matrix)
    #cost_matrix = prof_pref_4_students * stud_pref_4_profs
    cost_matrix = np.reshape(cost_matrix, (1, NUM_STUDENTS, NUM_PROFESSORS))
    cost_matrix = np.repeat(cost_matrix, NUM_INTERVIEWS, axis=0)
    #cost_matrix = list(cost_matrix)
    

    return(prof_pref_4_students, stud_pref_4_profs, cost_matrix)

''' Make the matches '''
def main():
    
    # Creates the model.
    model = cp_model.CpModel()

    # Get cost matrix
    prof_pref_4_students, stud_pref_4_profs, cost_matrix = define_random_matches()

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

    # Each professor has no more than one student per interview
    for s in all_students:
        for i in all_interviews:
            model.Add(sum(interview[(p, s, i)] for p in all_professors) <= 1)

    # No student is assigned to the same professor twice
    for s in all_students:
        for p in all_professors:
            model.Add(sum(interview[(p, s, i)] for i in all_interviews) <= 1)



    # Ensure that no student gets too many or too few interviews
    #for s in all_students:
    #    num_interviews_stud = sum(
    #        interview[(p, s, i)] for p in all_professors for i in all_interviews)
    #    model.Add(MIN_INTERVIEWS <= num_interviews_stud)
    #    model.Add(num_interviews_stud <= MAX_INTERVIEWS)

    # Ensure that no professor gets too many or too few interviews
    #for p in all_professors:
    #    num_interviews_prof = sum(
    #        interview[(p, s, i)] for s in all_students for i in all_interviews)
    #    model.Add(MIN_INTERVIEWS <= num_interviews_prof)
    #    model.Add(num_interviews_prof <= MAX_INTERVIEWS)
    
    # Define the minimization of cost
    model.Maximize(
        sum(cost_matrix[i][s][p] * interview[(p, s, i)] for p in all_professors 
            for s in all_students for i in all_interviews))
    
    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    print('Solving model...')
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL:
        print('optimal')
    if status == cp_model.FEASIBLE:
        print('feasible')
    if status == cp_model.INFEASIBLE:
        print('infeasible')
    if status == cp_model.MODEL_INVALID:
        print('model invalid')
    if status == cp_model.UNKNOWN:
        print('unknown')
    
    results = np.empty((NUM_INTERVIEWS, NUM_STUDENTS, NUM_PROFESSORS))
    for i in all_interviews:
        for p in all_professors:
            for s in all_students:
                results[i][s][p] = solver.Value(interview[(p, s, i)])
                
    print(results)
    

    # Statistics.
    #print()
    #print('Statistics')
    #print('  - Number of shift requests met = %i' % solver.ObjectiveValue(),
    #      '(out of', num_nurses * min_shifts_per_nurse, ')')
    print('  - wall time       : %f ms' % solver.WallTime())


if __name__ == '__main__':
    student_pref, faculty_pref, faculty_names, student_names = load_data()
    #main()
