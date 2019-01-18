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
        Excepts penalty for empty interview slots
        
        
    Future features:
        ADD MINIMUM NUMBER OF INTERVIEWS (HARD LIMIT) - Appears to work, just test
        ADD MAXIMUM NUMBER OF INTERVIEWS PER PERSON - Appears to work, just test
        ADD NOT AVAILABLE SLOTS
        IMPLEMENT FACULTY SIMILARY MATRIX
        ADD MATCHMAKING BASED ON TRACK (NEURAL, BIOMECHANICS, IMAGING, BIOMATERIALS)  
        PRINT PREFERENCE ON OUPUT FOR DEBUGGING ONLY
        VALIDATE THAT EVERYTHING WORKS USING LAST YEAR'S MATCHES
        ENSURE THAT ALL STUDENTS GET FAIR MATCHES, REGARDLESS OF PLACE ON LIST
        CREATE GOOGLE SURVEYS
        CREATE GUI
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
        self.NUM_INTERVIEWS = 10
        self.MIN_INTERVIEWS = 0
        self.MAX_INTERVIEWS = 10
        self.NUM_EXTRA_SLOTS = 2
        
        self.PATH = "/home/cale/Desktop/open_house/fresh_start"
        self.STUDENT_PREF = "student_preferences.csv"
        self.FACULTY_PREF = "faculty_preferences.csv"
        self.TIMES_NAME = "interview_times.csv"
        
        self.student_names = []
        self.faculty_names = []
        
        self.USE_RANKING = True
        self.MAX_RANKING = 10
        self.CHOICE_EXPONENT = 2
        
        self.USE_WORK_LUNCH = True
        self.LUNCH_PENALTY = 10     # This is a positive number, it will be made negative when used
        self.LUNCH_PERIOD = 4
        self.USE_RECRUITING = True
        self.RECRUITING_WEIGHT = 10
        
        self.EMPTY_PENALTY = 500  # 500 # This is a positive number, it will be made negative when used
        
        # Calculated parameters
        self.all_interviews = range(self.NUM_INTERVIEWS)
        
        # Check parameter validity
        if (self.FACULTY_ADVANTAGE < 0 or self.FACULTY_ADVANTAGE > 100):
            raise ValueError('It is necessary that: 0 <= Faculty_Advantage <= 100')
        
        
        if (type(self.FACULTY_ADVANTAGE) is not int):
            int_faculty_advantage = int(self.FACULTY_ADVANTAGE)
            self.FACULTY_ADVANTAGE = int_faculty_advantage
            warnings.warn('Faculty Advantage must be an integer, rounding to the nearest integer')

    ''' Loads the interview times from a csv '''
    def load_interview_times(self):
        
        self.interview_times = []        
        with open(path.join(self.PATH, self.TIMES_NAME), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                self.interview_times.append(row)
                
        
        self.NUM_INTERVIEWS = len(self.interview_times)
                
        
    
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
        Add names if we discover them when loading new data
        Because names are appended, we just add zeros at the end
    '''
    def add_names_to_match_data(self, data, new_data):
        
        difference_in_size = np.shape(new_data)[0] - np.shape(data)[0] 
        if difference_in_size > 0:
            zero_pad = np.zeros((difference_in_size, np.shape(data)[1]))
            data = np.concatenate((data, zero_pad), axis=0)
            
        return data
        
    ''' Load the data '''
    def load_data(self):      
        
        # Load the interview times
        self.load_interview_times()
        
        # Load the preference data
        self.student_pref = self.load_data_from_human_readable(self.STUDENT_PREF)
        self.faculty_pref = self.load_data_from_human_readable(self.FACULTY_PREF)
        self.student_pref = self.add_names_to_match_data(self.student_pref, self.faculty_pref)
        
        self.carol_matches = self.load_data_from_human_readable('assignments.csv') # Remove this after testing
        self.student_pref = self.add_names_to_match_data(self.student_pref, self.carol_matches)
        self.faculty_pref = self.add_names_to_match_data(self.faculty_pref, self.carol_matches)
        
        # Load the lunch and recruiting weight data
        if self.USE_RECRUITING:
            self.is_recruiting = self.load_data_from_human_readable('faculty_is_recruiting.csv', False)
            self.is_recruiting = self.response_to_weight(self.is_recruiting)
            
        if self.USE_WORK_LUNCH:
            self.will_work_lunch = self.load_data_from_human_readable('faculty_work_lunch.csv', False)
            self.will_work_lunch = self.response_to_weight(self.will_work_lunch)
        
        # Calculate the cost matrix
        self.calc_cost_matrix()
    
    
    ''' Transform yes/no/maybe into 1/2/3 '''
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
        
    ''' Transform the data into cost matrix'''
    def calc_cost_matrix(self):
        
        # Determine how to weight faculty preferences over student preferences
        alpha = self.FACULTY_ADVANTAGE
        beta = 100 - alpha
        self.cost_matrix = (alpha * self.faculty_pref + beta * self.student_pref).astype(int)
        
        # Give recruiting faculty an advantage
        if self.USE_RECRUITING:
            self.cost_matrix += (self.is_recruiting * self.RECRUITING_WEIGHT).astype(np.int64)

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
        
        times = np.asarray(self.interview_times)
        times.flatten()
        header_size = 3
        
        if not path.exists(path.join(self.PATH, 'faculty_schedules')):
            makedirs(path.join(self.PATH, 'faculty_schedules'))
        
        for p in self.all_faculty:
            faculty_name = self.faculty_names[p] + '.txt'
            filename = path.join(self.PATH,
                                 'faculty_schedules',
                                 faculty_name)
            faculty_schedule = np.asarray(self.faculty_schedule[p])
            faculty_schedule = np.reshape(faculty_schedule, (-1, 1))
            schedule = np.concatenate((times, faculty_schedule), axis=1)
            
            header = np.empty((header_size, 2), dtype=object)
            header[0, 0] = "Faculty"
            header[0, 1] = self.faculty_names[p]
            header[1, 0] = ''
            header[1, 1] = ''
            header[2, 0] = '   Time    '
            header[2, 1] = 'Student:'

            
            schedule = np.concatenate((header, schedule), axis=0)
            
            np.savetxt(filename, schedule,
                       delimiter=":   ", fmt='%s')
            
            
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
            
                        
            header = np.empty((header_size, 2), dtype=object)
            header[0, 0] = "Student"
            header[0, 1] = self.student_names[s]
            header[1, 0] = ''
            header[1, 1] = ''
            header[2, 0] = '   Time    '
            header[2, 1] = 'Faculty:'
            
            schedule = np.concatenate((header, schedule), axis=0)
            
            
            student_schedule = np.asarray(self.student_schedule[s])
            student_schedule = np.reshape(student_schedule, (-1, 1))
            schedule = np.concatenate((times, student_schedule), axis=1)
            
            np.savetxt(filename, schedule,
                       delimiter=":   ", fmt='%s')
        
        # Matches
        self.matches_text = []        
        for p in self.all_faculty:
            student_names = np.asarray(self.student_names)
            #matched_student_index = np.where(self.matches[p] == True)[0]
            #temp_list = student_names[student_num] for student_num in matched_student_index
            temp_list = student_names[self.matches[:, p]]
            self.matches_text.append(temp_list.tolist())
            
        
        faculty_names = np.asarray(self.faculty_names)
        #faculty_names = np.reshape(faculty_names, (1))
        matches = np.asarray(self.matches_text)
        matches_2_print = []
        for p in self.all_faculty:
            text = [faculty_names[p]]
            text.append(matches[p])
            matches_2_print.append(text)
       # matches = np.concatenate((faculty_names, matches))

            
        filename = path.join(self.PATH,
                             'matches.txt')
        np.savetxt(filename, matches_2_print,
                   delimiter="", fmt='%15s')
            

    ''' Calculate how much the method costs, and how good it is
        for each person to look for "bad deals
    '''
    def calculate_cost(self):
        pass
    
    
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
        a = 1
            
        
    
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
        
        # Get the faculty names
        if not self.faculty_names:
            self.faculty_names = match_data[1][1:]
            self.num_faculty = len(self.faculty_names)
            self.all_faculty = range(self.num_faculty)

        # Get the student names
        student_names = match_data[3:]
        
        # Make the student names an array
        all_students = np.asarray(student_names)
        all_students = all_students[:, 1:]      # This removes the first column, which is emtpy
        
        # Remove spaces from names
        for count, name in enumerate(all_students):
            for count2, name2 in enumerate(name):
                all_students[count][count2] = name2.replace(' ', '')
        
        if append_name:
            # Get the unique student names
            self.get_unique_student_names(all_students)
            
            # Put the match data in an array
            match_data_output = np.zeros((self.num_students, self.num_faculty))
            max_requests_made = np.shape(all_students)[0]
            name_array = np.asarray(self.student_names)
            for request_num in range(max_requests_made):
                for p in self.all_faculty:
                    student_num = np.where(all_students[request_num, p] == name_array)
                    if np.shape(student_num) == (1, 1):
                        if self.USE_RANKING:
                            match_data_output[student_num, p] = self.MAX_RANKING - request_num
                            if match_data_output[student_num, p] < 0:
                                match_data_output[student_num, p] = 0
                        else:
                            match_data_output[student_num, p] = 1
        else:
            match_data_output = all_students

        return match_data_output
        
        
        
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
        for s in self.all_students:
            num_interviews_stud = sum(
                interview[(p, s, i)] for p in self.all_faculty for i in self.all_interviews)
            model.Add(self.MIN_INTERVIEWS <= num_interviews_stud)
            model.Add(num_interviews_stud <= self.MAX_INTERVIEWS)
    
        # Ensure that no professor gets too many or too few interviews
        for p in self.all_faculty:
            num_interviews_prof = sum(
                interview[(p, s, i)] for s in self.all_students for i in self.all_interviews)
            model.Add(self.MIN_INTERVIEWS <= num_interviews_prof)
            model.Add(num_interviews_prof <= self.MAX_INTERVIEWS)
        
        # Define the minimization of cost
        print('Building Maximization term...')
        if self.EMPTY_PENALTY != 0:
            model.Maximize(
                sum(cost_matrix[i][s][p] * interview[(p, s, i)]
                    + self.EMPTY_PENALTY * interview[(p, s, i)]
                for p in self.all_faculty
                for s in self.all_students
                for i in self.all_interviews))
        else:
            model.Maximize(
                sum(cost_matrix[i][s][p] * interview[(p, s, i)]
                for p in self.all_faculty
                for s in self.all_students
                for i in self.all_interviews))
        
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
    
    
    '''
    
    
    
    mm.calculate_cost()
    '''
