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
    IMPLEMENT FACULTY SIMILARY MATRIX
    OUTPUT IN HUMAN-READABLE FORMAT
    DON'T LET ANY STUDENT GET NO INTERVIEWS BECAUSE OF BEING LAST IN LIST
'''


''' Imports'''
from ortools.sat.python import cp_model
import csv
from os import path, makedirs
import numpy as np
import warnings
from ortools.sat import sat_parameters_pb2


class match_maker():
    
    
    ''' Define parameters needed for scheduling '''
    def __init__(self):
        
        # Constants
        self.FACULTY_ADVANTAGE = 50
        self.MIN_INTERVIEWS = 1
        self.MAX_INTERVIEWS = 3
        self.NUM_INTERVIEWS = 10
        self.NUM_EXTRA_SLOTS = 2
        
        self.PATH = "/home/cale/Desktop/open_house/"
        self.STUDENT_PREF = "Student_Preferences.csv"
        self.FACULTY_PREF = "faculty_preferences.csv"
        
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
        PATH = self.PATH
        STUDENT_PREF = self.STUDENT_PREF
        FACULTY_PREF = self.FACULTY_PREF
        
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
        
        # Remove whitespace from names
        for p in self.all_faculty:
            self.faculty_names[p] = self.faculty_names[p].replace("'", "")
            self.faculty_names[p] = self.faculty_names[p].replace(" ", "")
            
        for s in self.all_students:            
            self.student_names[s] = self.student_names[s].replace("'", "")
            self.student_names[s] = self.student_names[s].replace(" " , "")
        
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
        
        # Parameters
        NUM_STUDENTS = 70
        num_faculty = 31
        NUM_INTERVIEWS = 10
        
        # Generate random matches
        prof_pref_4_students = np.random.randint(1, high=NUM_STUDENTS, size=(NUM_STUDENTS, num_faculty))
        stud_pref_4_profs = np.random.randint(1, high=num_faculty, size=(NUM_STUDENTS, num_faculty))

        # Calculate the cost matrix
        cost_matrix = prof_pref_4_students * stud_pref_4_profs
        cost_matrix = np.reshape(cost_matrix, (1, NUM_STUDENTS, num_faculty))
        cost_matrix = np.repeat(cost_matrix, NUM_INTERVIEWS, axis=0)
        
        self.faculty_pref = prof_pref_4_students
        self.student_pref = stud_pref_4_profs

        
    
        return(prof_pref_4_students, stud_pref_4_profs, cost_matrix)
        
    ''' Print a numpy array as a csv file'''
    def print_numpy_arrays(self, file_name, array):
        with open(path.join(self.PATH, file_name), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for i in self.all_interviews:
                for s in self.all_students:
                    writer.writerow(array[i][s][:])
                print('\n')
                

    ''' Convert the boolean matrix to a string matrix '''
    def matches_as_text(self):
        
        # Interview - Faculty Schedule     
        self.faculty_schedule = []        
        for p in self.all_faculty:                
            temp_list = []
            for i in self.all_interviews:
                s = 0
                found_match = False
                while s < self.num_students and not found_match:
                    if self.results[i][s][p] == 1:
                        temp_list.append(self.student_names[s])
                        found_match = True
                    s += 1
                if not found_match:
                    temp_list.append('Free')               
            self.faculty_schedule.append(temp_list)
            
        if not path.exists(path.join(self.PATH, 'faculty_schedules')):
            makedirs(path.join(self.PATH, 'faculty_schedules'))
        
        for p in self.all_faculty:
            faculty_name = self.faculty_names[p] + '.txt'
            filename = path.join(self.PATH,
                                 'faculty_schedules',
                                 faculty_name)
            np.savetxt(filename, self.faculty_schedule[p],
                       delimiter="\n", fmt='%s')
            
            
        # Interview - Student Schedule     
        self.student_schedule = []        
        for s in self.all_students:                
            temp_list = []
            for i in self.all_interviews:
                p = 0
                found_match = False
                while p < self.num_faculty and not found_match:
                    if self.results[i][s][p] == 1:
                        temp_list.append(self.faculty_names[p])
                        found_match = True
                    p += 1
                if not found_match:
                    temp_list.append('Free')               
            self.student_schedule.append(temp_list)
            
        if not path.exists(path.join(self.PATH, 'student_schedules')):
            makedirs(path.join(self.PATH, 'student_schedules'))
        
        for s in self.all_students:
            student_name = self.student_names[s] + '.txt'
            filename = path.join(self.PATH,
                                 'student_schedules',
                                 student_name)
            np.savetxt(filename, self.student_schedule[s],
                       delimiter="\n", fmt='%s')
        
        # Matches
        self.matches_text = []        
        for p in self.all_faculty:
            student_names = np.asarray(self.student_names)
            #matched_student_index = np.where(self.matches[p] == True)[0]
            #temp_list = student_names[student_num] for student_num in matched_student_index
            temp_list = student_names[self.matches[:, p]]
            self.matches_text.append(temp_list.tolist())
            
        
        faculty_names = np.asarray(self.faculty_names)
        faculty_names = np.reshape(faculty_names, (-1, 1))
        matches = np.asarray(self.matches_text)
        matches = np.concatenate((faculty_names, matches), axis=1)

            
        filename = path.join(self.PATH,
                             'matches_text.txt')
        np.savetxt(filename, matches,
                   delimiter="", fmt='%15s')
            

        
                    
            
    
    ''' Make the matches '''
    def main(self):
        
        # Creates the model.
        model = cp_model.CpModel()
    
        # Get cost matrix
        #self.define_random_matches()
        self.load_data()
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
        print('Building Model...', flush=True)
        solver = cp_model.CpSolver()
        
        print('Setting up workers...', flush=True)
        solver.parameters = sat_parameters_pb2.SatParameters(num_search_workers=8)
        
        print('Solving model...', flush=True)
        status = solver.Solve(model)   
        
        print(solver.StatusName(status))
 
        
        # Collect results
        results = np.empty((self.NUM_INTERVIEWS, self.num_students, self.num_faculty))
        for i in self.all_interviews:
            for p in self.all_faculty:
                for s in self.all_students:
                    results[i][s][p] = solver.Value(interview[(p, s, i)])
                    
        print(results)
             
        # Save the results
        self.results = results
        self.solver = solver
        self.matches = np.sum(self.results, axis=0).astype(bool)
        
        # Convert the results to text
        self.matches_as_text()
        
        # Write the results to a file
        #np.savetxt(path.join(self.PATH, 'results.csv'), results, delimiter=",")
        self.print_numpy_arrays('results.csv', self.results)
        np.savetxt(path.join(self.PATH, 'matches.csv'),
                   self.matches, delimiter=",",
                   fmt='%i')
        
                
                        
        
        


if __name__ == '__main__':
    mm = match_maker()
    mm.load_data()
    mm.main()
