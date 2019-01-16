from __future__ import division
from __future__ import print_function

'''
    open_house_scheduling.py
    Cale Crowder
    January 14, 2019

    Attempts to schedule student-faculty interviews
'''

'''
TODO:
    CONVERT CONSTANTS INTO CALCULATED VALUES FROM DATA
    MINIMIZE NUMBER OF EMPTY SLOTS
    ADD MINIMUM NUMBER OF EMPTY SLOTS (HARD LIMIT)
    GIVE RECRUITING FACULTY AN ADVANTAGE
    ASSIGN COST TO HAVING NO LUNCH BREAK
    ADD NOT AVAILABLE SLOTS
'''


''' Imports'''
from ortools.sat.python import cp_model
import csv
from os import path
import numpy as np
import warnings


class match_maker():
    
    
    ''' Define parameters needed for scheduling '''
    def __init__(self):
        
        # Constants
        self.FACULTY_ADVANTAGE = 80
        self.MIN_INTERVIEWS = 1
        self.MAX_INTERVIEWS = 3
        self.NUM_INTERVIEWS = 10
        self.NUM_EXTRA_SLOTS = 2
        
        # Calculated parameters
        self.all_interviews = range(self.NUM_INTERVIEWS)
        
        # Check parameter validity
        if (self.FACULTY_ADVANTAGE < 0 or self.FACULTY_ADVANTAGE > 100):
            raise ValueError('It is necessary that: 0 <= Faculty_Advantage <= 100')
        
        
        if (type(self.FACULTY_ADVANTAGE) is not int):
            int_faculty_advantage = int(self.FACULTY_ADVANTAGE)
            self.FACULTY_ADVANTAGE = int_faculty_advantage
            warnings.warn('Faculty Advantage must be an integer, rounding to the nearest integer')


    ''' Load the data '''
    def load_data(self):
        
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
        
        
        # Collect loaded data
        self.student_pref = student_pref
        self.faculty_pref = faculty_pref
        self.faculty_names = faculty_names
        self.student_names = student_names
        
        # Get necessary statistics about the data
        self.num_students = len(self.student_names)
        self.num_faculty = len(self.faculty_names)
        
        self.all_students = range(self.num_students)
        self.all_faculty = range(self.num_faculty)
       
        self.calc_cost_matrix()
        
    ''' Transform the data into cost matrix'''
    def calc_cost_matrix(self):
                
        alpha = self.FACULTY_ADVANTAGE
        beta = 100 - alpha
        
        self.cost_matrix = (alpha * self.faculty_pref + beta * self.student_pref).astype(int)
        self.cost_matrix = np.reshape(self.cost_matrix,
                                      (1, self.num_students, self.num_faculty))
        self.cost_matrix = np.repeat(self.cost_matrix, self.NUM_INTERVIEWS, axis=0)

    ''' Randomly generate prefered matches for testing '''
    def define_random_matches(self):
        
        NUM_STUDENTS = 48
        num_faculty = 31
        NUM_INTERVIEWS = 10
        
        prof_pref_4_students = np.random.randint(1, high=NUM_STUDENTS, size=(NUM_STUDENTS, num_faculty))
        stud_pref_4_profs = np.random.randint(1, high=num_faculty, size=(NUM_STUDENTS, num_faculty))
        print(prof_pref_4_students)
        print(stud_pref_4_profs)
    
        #prof_pref_4_students = np.array([[1, 1, 2], [1, 1, 2], [1, 1, 2]])*10
        #stud_pref_4_profs = np.array([[2, 2, 1], [1, 2, 1], [2, 2, 1]])
        
        cost_matrix = prof_pref_4_students * stud_pref_4_profs
        print(cost_matrix)
        #cost_matrix = prof_pref_4_students * stud_pref_4_profs
        cost_matrix = np.reshape(cost_matrix, (1, NUM_STUDENTS, num_faculty))
        cost_matrix = np.repeat(cost_matrix, NUM_INTERVIEWS, axis=0)
        #cost_matrix = list(cost_matrix)
        
    
        return(prof_pref_4_students, stud_pref_4_profs, cost_matrix)
    
    ''' Make the matches '''
    def main(self):
        
        # Creates the model.
        model = cp_model.CpModel()
    
        # Get cost matrix
        #prof_pref_4_students, stud_pref_4_profs, cost_matrix = self.define_random_matches()
        self.load_data()
        
        #prof_pref_4_students = self.faculty_pref
        #stud_pref_4_profs = self.student_pref
        cost_matrix = self.cost_matrix
        
        
        # Creates interview variables.
        # interview[(p, s, i)]: professor 'p' interviews student 's' for interview number 'i'
        interview = {}
        for p in self.all_faculty:
            for s in self.all_students:
                for i in self.all_interviews:
                    interview[(p, s,
                            i)] = model.NewBoolVar('interview_n%id%is%i' % (p, s, i))
    
        # Each student has no more than one interview at a time
        for p in self.all_faculty:
            for i in self.all_interviews:
                model.Add(sum(interview[(p, s, i)] for s in self.all_students) <= 1)
    
        # Each professor has no more than one student per interview
        for s in self.all_students:
            for i in self.all_interviews:
                model.Add(sum(interview[(p, s, i)] for p in self.all_faculty) <= 1)
    
        # No student is assigned to the same professor twice
        for s in self.all_students:
            for p in self.all_faculty:
                model.Add(sum(interview[(p, s, i)] for i in self.all_interviews) <= 1)
    
    
    
        # Ensure that no student gets too many or too few interviews
        #for s in all_students:
        #    num_interviews_stud = sum(
        #        interview[(p, s, i)] for p in all_faculty for i in all_interviews)
        #    model.Add(MIN_INTERVIEWS <= num_interviews_stud)
        #    model.Add(num_interviews_stud <= MAX_INTERVIEWS)
    
        # Ensure that no professor gets too many or too few interviews
        #for p in all_faculty:
        #    num_interviews_prof = sum(
        #        interview[(p, s, i)] for s in all_students for i in all_interviews)
        #    model.Add(MIN_INTERVIEWS <= num_interviews_prof)
        #    model.Add(num_interviews_prof <= MAX_INTERVIEWS)
        
        # Define the minimization of cost
        model.Maximize(
            sum(cost_matrix[i][s][p] * interview[(p, s, i)] for p in self.all_faculty 
                for s in self.all_students for i in self.all_interviews))
        
        # Creates the solver and solve.
        print('Building Model...')
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
        
        results = np.empty((self.NUM_INTERVIEWS, self.num_students, self.num_faculty))
        for i in self.all_interviews:
            for p in self.all_faculty:
                for s in self.all_students:
                    results[i][s][p] = solver.Value(interview[(p, s, i)])
                    
        print(results)
        
    
        # Statistics.
        #print()
        #print('Statistics')
        #print('  - Number of shift requests met = %i' % solver.ObjectiveValue(),
        #      '(out of', num_nurses * min_shifts_per_nurse, ')')
        print('  - wall time       : %f ms' % solver.WallTime())
        
        self.results = solver


if __name__ == '__main__':
    mm = match_maker()
    mm.load_data()
    mm.main()
