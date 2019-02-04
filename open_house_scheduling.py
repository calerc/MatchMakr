from __future__ import division
from __future__ import print_function

'''
    open_house_scheduling.py
    Cale Crowder
    January 14, 2019

    Attempts to schedule student-faculty interviews
    
    Current features:
        Reads faculty and student preferences from human-readable .csv
        Can interpret lists as ordered choices
        Can weight faculty preferences higher/lower than student choices
        Generates human-readable .csv schedules
        Generates human-readable un-ordered match list
        Allows faculty to request no interviews during lunch
        Give recruiting faculty an advantage in who they choose
        Outputs a schedule with a time vector read from a csv
        Accepts penalty for empty interview slots
        Accepts a hard limit to maximum number of interviews
        Accepts a hard limit to minimum number of interviews
        Can print preference number to schedules (for debugging)
        Accepts "not available" slots (BUT THIS IS BUGGY)
        Accepts a vector to match people based on their track
        Accepts an interviewer similarity score
        Provides suggestions for interview spots
        
        
    Future features:   
        CHANGE COMMENTS TO BENEFIT INSTEAD OF COST
        IMPLEMENT STATUS PRINTER TO STOP WHEN GOOD SOLUTION IS FOUND
        IMPLEMENT NOT AVAILABLE SLOTS AS NEGATIVE BENEFIT
        VALIDATE THAT EVERYTHING WORKS USING LAST YEAR'S MATCHES
        ENSURE THAT ALL STUDENTS GET FAIR MATCHES, REGARDLESS OF PLACE ON LIST
        SUGGEST MATCHES FOR FREE SPOTS
        CREATE GOOGLE SURVEYS
        MAKE THE SCHEDULES MORE BEAUTIFUL - MAKE PRINT FUNCTION FOR SCHEDULES
        CREATE GUI
        ENSURE THAT PARAMETERS DON'T INTERFERE WITH EACH OTHER
        ERROR-CHECK ANY INPUTS
        MAKE FILES CONSTANTS
        MAKE TIME LIMIT FOR SOLVER
        LET STUDENTS KNOW IF MATCH OR "RANDOM"
        CREATE PUBLIC GITHUB
        FIND SOMEWHERE TO HOST BINARIES
        VIDEO TO YOUTUBE
        ORGANIZE FUNCTIONS
'''


''' Imports'''
from ortools.sat.python import cp_model
import csv
from os import path, makedirs
import numpy as np
import warnings
from ortools.sat import sat_parameters_pb2


class VarArrayAndObjectiveSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables, student_pref, faculty_pref, num_faculty, num_students, num_interviews):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.variables = variables.values()
        self.__solution_count = 0
#        self.cost_matrix = cost_matrix
        self.student_pref = student_pref
        self.faculty_pref = faculty_pref
        self.num_faculty = num_faculty
        self.num_students = num_students
        self.num_interviews = num_interviews

    def on_solution_callback(self):
        
        # Get sizes
#        num_faculty = self.num_faculty
#        num_students = self.num_students
#        num_interviews = self.num_interviews
        
        # Print objective value
        print('Solution %i' % self.__solution_count)
        print('  objective value = %i' % self.ObjectiveValue())
        
        values = []
        for v in self.variables:
            values.append(self.Value(v))
        
        values = np.asarray(values)
        values = np.reshape(values, (self.num_faculty, self.num_students, self.num_interviews))
        
        # Print number of matches made
        
        
        
        self.__solution_count += 1

    def calc_cost(self):
        pass
        
        


        

    def solution_count(self):
        return self.__solution_count



class match_maker():
       
    
    ''' Define parameters needed for scheduling '''
    def __init__(self):
        
        # Constants
        self.FACULTY_ADVANTAGE = 50
        self.NUM_INTERVIEWS = 10
        
        self.USE_NUM_INTERVIEWS = True
        self.MIN_INTERVIEWS = 3
        self.MAX_INTERVIEWS = 10
        
        self.USE_EXTRA_SLOTS = True
        self.NUM_EXTRA_SLOTS = 2
        
        self.PATH = "/home/cale/Desktop/open_house/fresh_start"
        self.STUDENT_PREF = "stud_pref_order.csv"
        self.FACULTY_PREF = "faculty_preferences.csv"
        self.TIMES_NAME = "interview_times.csv"
        
        self.student_names = []
        self.faculty_names = []
        
        self.USE_RANKING = True     # True if use preference order instead of binary
        self.MAX_RANKING = 10
        self.CHOICE_EXPONENT = 2
        
        self.USE_WORK_LUNCH = True
        self.LUNCH_PENALTY = 10     # This is a positive number, it will be made negative when used
        self.LUNCH_PERIOD = 4
        
        self.USE_RECRUITING = True
        self.RECRUITING_WEIGHT = 10
        
        self.USE_AVAILABILITY = False
        self.FACULTY_AVAILABILITY_NAME = 'faculty_availability.csv'
        self.STUDENT_AVAILABILITY_NAME = 'student_availability.csv'
        
        self.USE_TRACKS = False
        self.TRACK_WEIGHT = 1
        
        self.USE_FACULTY_SIMILARITY = True
        self.FACULTY_SIMILARITY_WEIGHT = 2
        self.NUM_SIMILAR_FACULTY = 5
        
        self.NUM_SUGGESTIONS = 2
        
        self.MAX_SOLVER_TIME_SECONDS = 60
        
        # Avoid using - it's slow
        # This number should be chosen so that it is larger than lunch penalty
        self.EMPTY_PENALTY = 0  # 500 # This is a positive number, it will be made negative when used # Make zero to not use
        
        self.DEBUG_PRINT_PREFERENCE = False
        
        # Calculated parameters
        self.all_interviews = range(self.NUM_INTERVIEWS)
        
        # Check parameter validity
        if (self.FACULTY_ADVANTAGE < 0 or self.FACULTY_ADVANTAGE > 100):
            raise ValueError('It is necessary that: 0 <= Faculty_Advantage <= 100')
        
        
        if (type(self.FACULTY_ADVANTAGE) is not int):
            int_faculty_advantage = int(self.FACULTY_ADVANTAGE)
            self.FACULTY_ADVANTAGE = int_faculty_advantage
            warnings.warn('Faculty Advantage must be an integer, rounding to the nearest integer')

    '''
        Add names if we discover them when loading new data
        Because names are appended, we just add zeros at the end
    '''
    def add_names_to_match_data(self, data, new_data):
        
        difference_in_size = np.shape(new_data)[0] - np.shape(data)[0] 
        if difference_in_size > 0:
            zero_pad = np.zeros((difference_in_size, np.shape(data)[1]))
            data = np.concatenate((data, zero_pad), axis=0)
            
        return data
    
    ''' Transform the data into cost matrix'''
    def calc_cost_matrix(self):
        
        # Determine how to weight faculty preferences over student preferences
        alpha = self.FACULTY_ADVANTAGE
        beta = 100 - alpha
        self.cost_matrix = (alpha * self.faculty_pref + beta * self.student_pref).astype(int)
        
        # Give recruiting faculty an advantage
        if self.USE_RECRUITING:
            self.cost_matrix += (self.is_recruiting * self.RECRUITING_WEIGHT).astype(np.int64)
            
        # Add a benefit for being in the same track, but only if currently not
        # matched
        not_matched = self.cost_matrix == 0
        if self.USE_TRACKS:
            add_track_advantage = np.logical_and(not_matched, self.same_track == 1)
            self.cost_matrix[add_track_advantage] += self.TRACK_WEIGHT
            
        # Add a benefit to similar faculty, if not matched, for students top n faculty
        if self.USE_FACULTY_SIMILARITY:
            for s in self.all_students:
                for p in range(self.NUM_SIMILAR_FACULTY):
                    match_benefit = self.MAX_RANKING - p
                    faculty_choice = np.where(self.student_pref[s, :] == match_benefit)
                    if np.shape(faculty_choice)[1] > 0:
                        was_not_matched = np.where(not_matched[s, :])
                        similar_faculty = self.faculty_similarity[was_not_matched, faculty_choice]
                        self.cost_matrix[s, was_not_matched] += similar_faculty * self.FACULTY_SIMILARITY_WEIGHT
            
            #self.cost_matrix[not_matched] += (self.faculty_similarity[not_matched] * self.FACULTY_SIMILARITY_WEIGHT)

        # Expand the cost_matrix to cover each interview period        
        self.cost_matrix = np.reshape(self.cost_matrix,
                                      (1, self.num_students, self.num_faculty))
        self.cost_matrix = np.repeat(self.cost_matrix, self.NUM_INTERVIEWS, axis=0)
        
        # Add a cost for working during lunch
        if self.USE_WORK_LUNCH:
            self.cost_matrix[self.LUNCH_PERIOD, :, :] -= ((2 - self.will_work_lunch) * self.LUNCH_PENALTY).astype(np.int64) # The 2 is the maximum number of points we can remove for lunch weight because of response_to_weight
            
                    
        # Square the cost matrix to maximize chance of getting first choice
        #self.cost_matrix[self.cost_matrix < 0] = 0
        cost_sign = np.sign(self.cost_matrix)
        self.cost_matrix = np.power(self.cost_matrix, self.CHOICE_EXPONENT)
        self.cost_matrix *= cost_sign

    ''' Track how many people got their preferences '''
    def check_preferences(self):
        
        NUM_PREFERENCES_2_CHECK = 3
        
        # Students
        student_pref = self.student_pref * self.matches
        self.student_pref_cost = np.sum(student_pref, axis=1)
        total_preferences = np.empty((NUM_PREFERENCES_2_CHECK))
        preferences_met = np.empty((NUM_PREFERENCES_2_CHECK))
        for pref_num in range(NUM_PREFERENCES_2_CHECK):
            total_preferences[pref_num] = np.sum(self.student_pref == (10 - pref_num))
            preferences_met[pref_num] = np.sum(student_pref == (10 - pref_num))
        
        self.student_fraction_preferences_met = preferences_met / total_preferences
        print('Fraction of student preferences met: ')
        print(self.student_fraction_preferences_met)
        
        # Faculty
        faculty_pref = self.faculty_pref * self.matches
        self.faculty_pref_cost = np.sum(faculty_pref, axis=0)
        total_preferences = np.empty((NUM_PREFERENCES_2_CHECK))
        preferences_met = np.empty((NUM_PREFERENCES_2_CHECK))
        for pref_num in range(NUM_PREFERENCES_2_CHECK):
            total_preferences[pref_num] = np.sum(self.faculty_pref == (10 - pref_num))
            preferences_met[pref_num] = np.sum(faculty_pref == (10 - pref_num))
        
        self.faculty_fraction_preferences_met = preferences_met / total_preferences
        print('Fraction of faculty preferences met: ')
        print(self.faculty_fraction_preferences_met)
        
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
        
        # Faculty
        return(prof_pref_4_students, stud_pref_4_profs, cost_matrix)
        

    ''' Check what names should be appended to student array '''
    def get_unique_student_names(self, new_names):
        
        # Find unique student names        
        all_students_unique = np.reshape(new_names, (-1, 1))
        
        for count, name in enumerate(all_students_unique):
             all_students_unique[count] = name[0]
        new_student_names, student_idx = np.unique(all_students_unique, return_inverse=True)
        
        # Remove 'empty' names
        is_empty_name = np.asarray([name == '' for name in new_student_names])
        new_student_names = new_student_names[np.logical_not(is_empty_name)]
        
        # Find the names that are new
        previous_names = np.asarray(self.student_names)
        new_names = np.asarray(new_student_names)
        is_previous_name = np.in1d(new_names, previous_names)
        really_new_names = new_names[np.logical_not(is_previous_name)].tolist()
        
        # Append the new names to the student_names array
        for name in really_new_names:
            self.student_names.append(name)
        self.num_students = len(self.student_names)
        self.all_students = range(self.num_students)
        
        # Tell the user what names have been found
        print('The following '
              + str(len(self.student_names))
              + ' student names have been detected:')
        
        print(self.student_names)
        
        
    '''
        Load the availability data for students or faculty
    '''
    def load_availability(self, filename, num_expected_available):
        
        # Load the availability
        availability = self.load_data_from_human_readable(filename, False).astype(int)
        
        # Check that the number of availabilities is expected
        [_, num_available] = np.shape(availability)
        if num_available != num_expected_available:
            raise ValueError('The availability data does not match the preference data')
            
        available = np.asarray(np.where(np.any(availability, axis=0))).squeeze()
        
        # return
        return availability, available
    

        
    ''' 
        Load Carol's previous matches as a comparison 
        This can be deleted once the validity of this method has been verified
    '''
    def load_carol_matches(self):
        
        # Compare names of students (assume faculty are same)
        matching_name_num = -1 * np.ones(self.num_students)
        num_students_carol = len(self.carol_students)
        for count, name in enumerate(self.student_names):
            match_not_found = True
            student_num = 0
            while student_num < num_students_carol and match_not_found:
                if name == self.carol_students[student_num]:
                    matching_name_num[count] = student_num
                    print('match found ' + str(count) + ' ' + name)
                    match_not_found = False
                student_num += 1
                
        # Print the names that weren't found in Carol's matches
        names_array = np.asarray(self.student_names)
        names_not_found = names_array[matching_name_num == -1]
        print('Names not found:')
        print(names_not_found)
                
        # Populate the array with Carols names and matches

        
    
    '''
        Load track data
    '''
    def load_track_data(self):
        
        # Get the track data from files
        self.faculty_tracks = self.load_data_from_human_readable('faculty_tracks.csv')
        self.student_tracks = self.load_data_from_human_readable('student_tracks.csv')
        
        # Find students and faculty that are in the same track
        all_tracks = np.concatenate((self.faculty_tracks, self.student_tracks), axis=1)
        unique_tracks, unique_idx = np.unique(all_tracks, return_inverse=True)
        
        self.same_track = np.zeros((self.num_students, self.num_faculty))
        for count, track in enumerate(unique_tracks):
            if track != 'None' and track != '':
                same_track = np.asarray(np.where(unique_idx == count))
                faculty_nums = np.reshape(same_track[same_track < self.num_faculty], (1, -1))
                student_nums = np.reshape(same_track[same_track >= self.num_faculty] - self.num_faculty, (-1, 1))
                
                self.same_track[student_nums, faculty_nums] = 1
    
    '''
        Load matrix data
    '''
    def load_matrix_data(self, filename):
        matrix_data = []
        
        with open(path.join(self.PATH, filename), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                matrix_data.append(row)
        
        return(matrix_data)
    
    
    '''
        Load faculty similarity matrix
    '''
    def load_faculty_similarity(self):
        
        # Load the matrix data
        self.faculty_similarity = self.load_matrix_data('faculty_similarity.csv')
        
        # Convert to an array
        self.faculty_similarity = np.asarray(self.faculty_similarity, dtype=int)
        
        # Check that the array size is correct
        num_rows, num_columns = np.shape(self.faculty_similarity)
        if num_rows != self.num_faculty or num_columns != self.num_faculty:
            raise ValueError('Faculty similarity size does not match the number of faculty')
        
 
    ''' Load the data '''
    def load_data(self):      
        
        # Load the interview times
        self.load_interview_times()
        
        # Load the preference data
        self.load_preference_data()        
        self.carol_matches = self.load_data_from_human_readable('assignments.csv') # Remove this after testing

        # Load the track data
        if self.USE_TRACKS:
            self.load_track_data()
            
        # Load the faculty similarity data
        if self.USE_FACULTY_SIMILARITY:
            self.load_faculty_similarity()
        
        # Load the lunch and recruiting weight data
        if self.USE_RECRUITING:
            self.is_recruiting = self.load_data_from_human_readable('faculty_is_recruiting.csv', False)
            self.is_recruiting = self.response_to_weight(self.is_recruiting)
            
        if self.USE_WORK_LUNCH:
            self.will_work_lunch = self.load_data_from_human_readable('faculty_work_lunch.csv', False)
            self.will_work_lunch = self.response_to_weight(self.will_work_lunch)
        
        # Load the availability data
        if self.USE_AVAILABILITY:
            
            # Student
            self.student_availability, self.students_avail = self.load_availability(
                    self.STUDENT_AVAILABILITY_NAME, len(self.student_names))
            
            # Faculty
            self.faculty_availability, self.faculty_avail = self.load_availability(
                    self.FACULTY_AVAILABILITY_NAME, len(self.faculty_names))
            
            self.remove_unavailable()
        

        
        # Calculate the cost matrix
        self.calc_cost_matrix()

                
    ''' Old load function.  Here for documentation '''
    def load_data_defunct(self):
    
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

        # Remove whitespace from names
        for p in self.all_faculty:
            self.faculty_names[p] = self.faculty_names[p].replace("'", "")
            self.faculty_names[p] = self.faculty_names[p].replace(" ", "")
            
        for s in self.all_students:            
            self.student_names[s] = self.student_names[s].replace("'", "")
            self.student_names[s] = self.student_names[s].replace(" " , "")

    
    '''
        Load the preference data
    '''
    def load_preference_data(self):
        
        # Load the student data
        stud_match_data = []
        with open(path.join(self.PATH, self.STUDENT_PREF), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                stud_match_data.append(row)
                
        # Load the faculty data
        faculty_match_data = []
        with open(path.join(self.PATH, self.FACULTY_PREF), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                faculty_match_data.append(row)
                
        # Make the data into numpy arrays
        stud_match_data = np.asarray(stud_match_data)
        faculty_match_data = np.asarray(faculty_match_data)
        
        # Extract the names
        student_names = stud_match_data[1,1:]
        student_names = student_names[np.where(student_names != '')]
        faculty_names = faculty_match_data[1,1:]
        faculty_names = faculty_names[np.where(faculty_names != '')]
        
        # Extract the preferences
        student_pref = stud_match_data[3:, 1:]
        faculty_pref = faculty_match_data[3:, 1:]
        
        # Statistics
        self.num_students = len(student_names)
        self.num_faculty = len(faculty_names)
        self.all_students = range(self.num_students)
        self.all_faculty = range(self.num_faculty)
        
        # Remove spaces from names and preferences
        for count, name in enumerate(student_names):
            student_names[count] = student_names[count].replace(' ', '')
        
        for count, name in enumerate(faculty_names):
            faculty_names[count] = faculty_names[count].replace(' ', '')
            
        for count, pref in enumerate(student_pref):
            for count2, pref2 in enumerate(pref):
                student_pref[count, count2] = student_pref[count, count2].replace(' ', '')
                
        for count, pref in enumerate(faculty_pref):
            for count2, pref2 in enumerate(pref):
                faculty_pref[count, count2] = faculty_pref[count, count2].replace(' ', '')
                
        # Fill-in faculty preferences
        self.faculty_pref = np.zeros((self.num_students, self.num_faculty))
        
        for p in self.all_faculty:
            temp_pref = faculty_pref[np.where(faculty_pref[:, p] != ''), p].flatten()
            for count, pref in enumerate(temp_pref):
                student_num = np.where(student_names == pref)
                self.faculty_pref[student_num, p] = self.MAX_RANKING - count
                
        # Fill-in student preferences
        self.student_pref = np.zeros((self.num_students, self.num_faculty))
          
        for s in self.all_students:
            temp_pref = student_pref[np.where(student_pref[:, s] != ''), s].flatten()
            for count, pref in enumerate(temp_pref):
                faculty_num = np.where(faculty_names == pref)
                self.student_pref[s, faculty_num] = self.MAX_RANKING - count
        
        # Assign object names
        self.student_names = student_names
        self.faculty_names = faculty_names


    
    ''' 
        Read requests from human-readable format 
        Rows:
            The first row of the file will be a header
            The second row of the file will be the faculty names
            The third row will be blank
            The next rows will contain the names of students
        Columns:
            The first column will be a header
            The next columns will contain data
        
    '''
    def load_data_from_human_readable(self, filename, append_name=True):
        
        # Load the data
        match_data = []
        with open(path.join(self.PATH, filename), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                match_data.append(row)
                
        match_data = np.asarray(match_data)        
        match_data = match_data[3:, 1:]

        return match_data
        
        
    '''
        Loads the interview times from a csv
    '''
    def load_interview_times(self):
        
        self.interview_times = []        
        with open(path.join(self.PATH, self.TIMES_NAME), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                self.interview_times.append(row)
                
        
        self.NUM_INTERVIEWS = len(self.interview_times)
             

    ''' 
        Make the matches
    '''
    def main(self):
        
        # Creates the model.
        model = cp_model.CpModel()
    
        # Get cost matrix
        #self.define_random_matches()
        self.load_data()
        cost_matrix = self.cost_matrix
        
        # Creates interview variables.
        # interview[(p, s, i)]: professor 'p' interviews student 's' for interview number 'i'
        self.interview = {}
        for p in self.all_faculty:
            for s in self.all_students:
                for i in self.all_interviews:
                    self.interview[(p, s,
                            i)] = model.NewBoolVar('interview_p%i_s%i_i%i' % (p, s, i))
    
        # Each student has no more than one interview at a time
        for p in self.all_faculty:
            for i in self.all_interviews:
                model.Add(sum(self.interview[(p, s, i)] for s in self.all_students) <= 1)
    
        # Each professor has no more than one student per interview
        for s in self.all_students:
            for i in self.all_interviews:
                model.Add(sum(self.interview[(p, s, i)] for p in self.all_faculty) <= 1)
    
        # No student is assigned to the same professor twice
        for s in self.all_students:
            for p in self.all_faculty:
                model.Add(sum(self.interview[(p, s, i)] for i in self.all_interviews) <= 1)
                
        # Interviews only assigned when both parties are available
        if self.USE_AVAILABILITY:
            for p in self.all_faculty:
                for s in self.all_students:
                    for i in self.all_interviews:
                        if self.student_availability[i, s] == 0:
                            model.Add(self.interview[(p, s, i)] == 1)
                        if self.faculty_availability[i, p] == 0:
                            model.Add(self.interview[(p, s, i)] == True)    
    
        if self.USE_NUM_INTERVIEWS:
            
            # Ensure that no student gets too many or too few interviews
            for s in self.all_students:
                num_interviews_stud = sum(
                    self.interview[(p, s, i)] for p in self.all_faculty for i in self.all_interviews)
                model.Add(self.MIN_INTERVIEWS <= num_interviews_stud)
                model.Add(num_interviews_stud <= self.MAX_INTERVIEWS)
        
            # Ensure that no professor gets too many or too few interviews
            for p in self.all_faculty:
                num_interviews_prof = sum(
                    self.interview[(p, s, i)] for s in self.all_students for i in self.all_interviews)
                model.Add(self.MIN_INTERVIEWS <= num_interviews_prof)
                model.Add(num_interviews_prof <= self.MAX_INTERVIEWS)
        
        # Define the minimization of cost
        print('Building Maximization term...')
        if self.EMPTY_PENALTY != 0:
            model.Maximize(
                sum(cost_matrix[i][s][p] * self.interview[(p, s, i)]
                    + self.EMPTY_PENALTY * self.interview[(p, s, i)]
                for p in self.all_faculty
                for s in self.all_students
                for i in self.all_interviews))
        else:
            model.Maximize(
                sum(cost_matrix[i][s][p] * self.interview[(p, s, i)]
                for p in self.all_faculty
                for s in self.all_students
                for i in self.all_interviews))
        
        # Creates the solver and solve.
        print('Building Model...', flush=True)
        solver = cp_model.CpSolver()
        solution_printer = VarArrayAndObjectiveSolutionPrinter(self.interview,
                                                               self.student_pref,
                                                               self.faculty_pref,
                                                               self.num_faculty,
                                                               self.num_students,
                                                               self.NUM_INTERVIEWS)
        
        print('Setting up workers...', flush=True)
        solver.parameters = sat_parameters_pb2.SatParameters(num_search_workers=8)
        solver.parameters.max_time_in_seconds = self.MAX_SOLVER_TIME_SECONDS
        
        print('Solving model...', flush=True)
        status = solver.SolveWithSolutionCallback(model, solution_printer)   
        
        print(solver.StatusName(status))
 
        
        # Collect results
        if solver.StatusName(status) == 'FEASIBLE' or solver.StatusName(status) == 'OPTIMAL':
            results = np.empty((self.NUM_INTERVIEWS, self.num_students, self.num_faculty))
            for i in self.all_interviews:
                for p in self.all_faculty:
                    for s in self.all_students:
                        results[i][s][p] = solver.Value(self.interview[(p, s, i)])
            print(results)     
        
             
            # Save the results
            self.results = results
            self.solver = solver
            self.matches = np.sum(self.results, axis=0).astype(bool)
            
            # Convert the results to text and save as text files
            self.matches_as_text()
            
            # Write the results to a file
            self.print_numpy_arrays('results.csv', self.results)
            np.savetxt(path.join(self.PATH, 'matches.csv'),
                       self.matches, delimiter=",",
                       fmt='%i')
            
            # Check the percentage of preferences met
            self.check_preferences()
        
        else:
            print('-------- Solver failed! --------')
    
    '''
        Check if an integer is odd
    '''
    def is_odd(self, num):
        return num & 0x1
    
    
    '''
        Print schedules
        names1 = people who the schedules are for
        names2 = people on the scheudles
        data_array:
                rows = candidates
                columns = people who the schedules are for
        person_string = string to be printed on file
    '''
    def print_schedules(self, person_string, folder_name, names1, schedule, good_matches):
        
        # Get the interview times
        times = np.asarray(self.interview_times)
        times.flatten()
        schedule = np.asarray(schedule)
        
        # Make the folder, if it doesn't exist
        if not path.exists(path.join(self.PATH, folder_name)):
            makedirs(path.join(self.PATH, folder_name))
            
        # Print the results
#        num_schedules = len(names1)
        for count, name in enumerate(names1):
            
            # Determine the file name
            file_name = name + '.txt'
            file_name = path.join(self.PATH, folder_name, file_name)
            
            # Open the file for editing
            with open(file_name, 'w') as file:
                
                # Header
                file.writelines(person_string + ' interivew schedule for:         ' + name + '\n')
                file.writelines('\n')
                file.writelines('\n')
                
                # Schedule
                if self.DEBUG_PRINT_PREFERENCE:
                    file.writelines('Time:                     Person:                 Preference:\n')
                    for i in self.all_interviews:
                        
                        if self.is_odd(i):
                            sep_string = ' +++++++++ '
                        else:
                            sep_string = ' --------- '
                            
                        file.writelines(np.array_str(times[i]) + sep_string
                                       + schedule[count, i] + sep_string
                                       + '\n')
                else:
                    file.writelines('Time:                     Person:\n')
                    for i in self.all_interviews:
                        
                        if self.is_odd(i):
                            sep_string = ' +++++++++ '
                        else:
                            sep_string = ' --------- '
                            
                        file.writelines(np.array_str(times[i]) + sep_string
                                       + schedule[count, i]
                                       + '\n')
            
                # Suggested matches
                file.writelines('\n')
                file.writelines('\n')
                file.writelines('During the open interview periods, we suggest you meet with: \n')
                
                for match_count, match in enumerate(good_matches[count]):
                    if match_count == 0:
                        file.writelines(match)
                    else:
                        file.writelines(', ' + match)
                        
                file.writelines('\n')
                file.writelines('\n')
                
                
                
    '''
        Find good matches that were not made
    '''
    def find_suggested_matches(self):
        
        # Initialize lists
        self.stud_suggest_matches = [None] * self.num_students
        self.faculty_suggest_matches = [None] * self.num_faculty
        
        # Determine benefit for matches not made
        matches = np.logical_not(self.matches)
        if self.LUNCH_PERIOD != 0:
            period = 0
        elif self.NUM_INTERVIEWS > 0:
            period = 1
        
        match_benefit = matches * self.cost_matrix[period]
        
        # Find good matches for faculty
        for p in self.all_faculty:
            
            # Find unique benefit levels
            unique_benefits, unique_counts = np.unique(match_benefit[:, p], return_counts=True)
            unique_counts = np.flipud(unique_counts)
            unique_benefits = np.flipud(unique_benefits)
            
            # Don't make 0-benefit suggestions
            unique_counts = unique_counts[unique_benefits > 0]
            unique_benefits = unique_benefits[unique_benefits > 0]
            
            # Determine how many benefit levels are needed to reach number of
            # suggestions needed
            summed_counts = np.cumsum(unique_counts)
            bin_needed = np.where(summed_counts > self.NUM_SUGGESTIONS)
            if np.shape(bin_needed)[0] == 0:
                bin_needed = np.shape(summed_counts)[0] - 1
            else:
                bin_needed = bin_needed[0][0]
            
            
            if np.shape(unique_benefits)[0] > 0:
                
                # Use all of the matches from the first few bins
                if bin_needed > 0:
                    good_matches = np.where(match_benefit[:, p] >= unique_benefits[bin_needed - 1])[0]
                    num_matches_made = np.shape(good_matches)[0]
                else:
                    good_matches = np.empty(0)
                    num_matches_made = 0
                
                # Take random matches from the last bin (because all have equal weight)
                possible_matches = np.where(match_benefit[:, p] == unique_benefits[bin_needed])[0]
                num_matches_needed = self.NUM_SUGGESTIONS - num_matches_made
                
                if num_matches_needed <= summed_counts[-1]:
                    rand_matches = np.random.choice(possible_matches, size=num_matches_needed)
                    matches = np.concatenate((good_matches, rand_matches)).astype(int)
                else:
                    matches = np.where(match_benefit[:, p])              
                
                
            else:
                matches = []
            
            self.faculty_suggest_matches[p] = self.student_names[matches]
            
        # Find good matches for students
        for s in self.all_students:
            
            # Find unique benefit levels
            unique_benefits, unique_counts = np.unique(match_benefit[s, :], return_counts=True)
            unique_counts = np.flipud(unique_counts)
            unique_benefits = np.flipud(unique_benefits)
            
            # Don't make 0-benefit suggestions
            unique_counts = unique_counts[unique_benefits > 0]
            unique_benefits = unique_benefits[unique_benefits > 0]
            
            # Determine how many benefit levels are needed to reach number of
            # suggestions needed
            summed_counts = np.cumsum(unique_counts)
            bin_needed = np.where(summed_counts > self.NUM_SUGGESTIONS)
            if np.shape(bin_needed)[0] == 0:
                bin_needed = np.shape(summed_counts)[0] - 1
            else:
                bin_needed = bin_needed[0][0]
            
            
            if np.shape(unique_benefits)[0] > 0:
                
                # Use all of the matches from the first few bins
                if bin_needed > 0:
                    good_matches = np.where(match_benefit[s, :] >= unique_benefits[bin_needed - 1])[0]
                    num_matches_made = np.shape(good_matches)[0]
                else:
                    good_matches = np.empty(0)
                    num_matches_made = 0
                
                # Take random matches from the last bin (because all have equal weight)
                possible_matches = np.where(match_benefit[s, :] == unique_benefits[bin_needed])[0]
                num_matches_needed = self.NUM_SUGGESTIONS - num_matches_made
                
                if num_matches_needed <= summed_counts[-1]:
                    rand_matches = np.random.choice(possible_matches, size=num_matches_needed)
                    matches = np.concatenate((good_matches, rand_matches)).astype(int)
                else:
                    matches = np.where(match_benefit[:, p])              
                
                
            else:
                matches = []
            
            self.stud_suggest_matches[s] = self.faculty_names[matches]
            
            '''
            unique_benefits = np.flipud(np.unique(match_benefit[s, :]))
            if np.shape(unique_benefits)[0] >= self.NUM_SUGGESTIONS:
                is_greater_than_zero = match_benefit[s, :] > 0
                is_ranked_highly = match_benefit[s, :] >= unique_benefits[self.NUM_SUGGESTIONS - 1]
                good_matches = np.where(np.logical_and(is_ranked_highly, is_greater_than_zero))
            else:
                good_matches = np.where(match_benefit[s, :])
            
            matches = self.faculty_names[good_matches]
            self.stud_suggest_matches[s] = matches
            '''
        
        
    
    '''
        Convert the boolean matrix to a string matrix
    '''
    def matches_as_text(self):
        
        # Find good matches that were not made
        self.find_suggested_matches()
        
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
        
        self.print_schedules('Faculty', 'faculty_schedules',
                             self.faculty_names, self.faculty_schedule,
                             self.faculty_suggest_matches)
        
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
            
        self.print_schedules('Student', 'student_schedules',
                             self.student_names, self.student_schedule,
                             self.stud_suggest_matches)
       
        
        # Matches
        self.matches_text = []        
        for p in self.all_faculty:
            student_names = np.asarray(self.student_names)
            temp_list = student_names[self.matches[:, p]]
            self.matches_text.append(temp_list.tolist())
            
        
        faculty_names = np.asarray(self.faculty_names)
        matches = np.asarray(self.matches_text)
        matches_2_print = []
        for p in self.all_faculty:
            text = [faculty_names[p]]
            text.append(matches[p])
            matches_2_print.append(text)

            
        filename = path.join(self.PATH,
                             'matches.txt')
        np.savetxt(filename, matches_2_print,
                   delimiter="", fmt='%15s')        
        
        
    '''
        Print a numpy array as a csv file
    '''
    def print_numpy_arrays(self, file_name, array):
        with open(path.join(self.PATH, file_name), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for i in self.all_interviews:
                for s in self.all_students:
                    writer.writerow(array[i][s][:])
                print('\n')     


    '''
        Remove students and faculty that are unavailable
    '''
    def remove_unavailable(self):
        
        # Availability
        self.student_availability = self.student_availability[:, self.students_avail]        
        self.faculty_availability = self.faculty_availability[:, self.faculty_avail]
        
        # Names
        self.student_names = np.asarray(self.student_names)[self.students_avail].tolist()
        self.faculty_names = np.asarray(self.faculty_names)[self.faculty_avail].tolist()
        
        # Match Preferences
        temp_stud_avail = np.reshape(self.students_avail, (-1, 1))
        temp_faculty_avail = np.reshape(self.faculty_avail, (1, -1))
        self.student_pref = self.student_pref[temp_stud_avail, temp_faculty_avail]        
        self.faculty_pref = self.faculty_pref[temp_stud_avail, temp_faculty_avail]
        
        # Faculty Time Preferences
        if self.USE_RECRUITING:
            self.is_recruiting = self.is_recruiting[self.faculty_avail]
        if self.USE_WORK_LUNCH:
            self.will_work_lunch = self.will_work_lunch[self.faculty_avail]
        
        # Numbers
        self.num_students = len(self.student_names)
        self.all_students = range(self.num_students)
        self.num_faculty = len(self.faculty_names)
        self.all_faculty = range(self.num_faculty)
        
        
    
    '''
        Transform yes/no/maybe into 2/0/1
    '''
    def response_to_weight(self, array):
        array = array.flatten()
        
        out_array = np.empty(np.shape(array))
        for count, response in enumerate(array):
            if response.lower() == 'yes':
                out_array[count] = 2
            elif response.lower() == 'no':
                out_array[count] = 0
            elif response.lower() == 'maybe':
                out_array[count] = 1
                
        return out_array
    
    
    
    
    


if __name__ == '__main__':
    
    mm = match_maker()
    mm.main()





















