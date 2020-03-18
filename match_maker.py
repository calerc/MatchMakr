from __future__ import division
from __future__ import print_function

import sys
from os import path, makedirs
from ortools.sat import sat_parameters_pb2
from ortools.sat.python import cp_model
from pdf_writer import PDFWriter
from input_checker import InputCheckerNoThrow
from my_or_tools import VarArrayAndObjectiveSolutionPrinter, ORSolver
import numpy as np
import csv
import warnings
import multiprocessing
from ipdb import set_trace
import threading


'''
    open_house_scheduling.py
    Cale Crowder
    January 14, 2019

    Attempts to schedule student-faculty interviews

    Current features:
        Reads faculty and student preferences from human-readable .csv
        Can interpret lists as ordered choices
        Can weight faculty preferences higher/lower than student choices
        Generates human-readable schedules
        Generates human-readable un-ordered match list
        Allows faculty to request no interviews during lunch
        Give recruiting faculty an advantage in who they choose
        Outputs a schedule with a time vector read from a csv
        Accepts penalty for empty lots
        Accepts a hard limit to maximum number of views
        Accepts a hard limit to minimum number of interviews
        Can print preference number to schedules (for debugging)
        Accepts "not available" slots
        Accepts a vector to match people based on their track
        Accepts an interviewer similarity score
        Provides suggestions for interview spots
        Allows the optimizer to be time-limited


    Future features:

        Code clarity:
            ENSURE THAT THE DOCUMENTATION IS UP TO SNUFF

        Code function:
            FREEZE
            Alphabetize functions
            Redo matches without redoing everything
            Make the constants for weights normalized
            Check if self.can_update is needed
            Possibly set initial condition of optimizer by copying previous matches into variable being optimmized
            Make time limit work
            Don't suggest same match twice
            Determine why objective matrix has some values equal to 0

        Code accessibility:
            CREATE GUI
            CREATE PUBLIC GITHUB
                Determine license type and copyright
            FIND SOMEWHERE TO HOST BINARIES
            VIDEO TO YOUTUBE


    KNOWN BUGS:
        bin_needed = bin_needed[0][0] - IndexError: index 0 is out of bounds for axis 0 with size 0
            This doesn't affect fitting, but it won't allow for output
'''



class match_maker():

    ''' Define parameters needed for scheduling '''

    def __init__(self):
        
        ''' Constants '''

        # Files to load
        self.PATH = "/media/veracrypt1/Users/Cale/Documents/Calers_Writing/PhD/GEC/scheduling_software/2020_data/processed_for_program"
        self.RESULTS_PATH = path.join(self.PATH, 'results')
        self.STUDENT_PREF = "CWRU_BME_Open_House_-_Student_Survey.csv"
        self.FACULTY_PREF = "CWRU_BME_Open_House_-_Faculty_Survey.csv"
        self.TIMES_NAME = "interview_times.csv"
        self.FACULTY_AVAILABILITY_NAME = 'faculty_availability.csv'
        self.STUDENT_AVAILABILITY_NAME = 'student_availability.csv'
        self.STUDENT_RANKING_FILE = 'student_ranking.csv'
        self.FACULTY_RANKING_FILE = 'faculty_ranking.csv'
        self.LOG_FILE_NAME = 'log.txt'
        self.RESULTS_NAME = 'results.csv'
        self.MATCHES_CSV_NAME = 'matches.csv'
        self.MATCHES_TXT_NAME = 'matches.txt'
        self.STUDENT_SCHEDULES_DIR = 'student_schedules'
        self.FACULTY_SCHEDULES_DIR = 'faculty_schedules'

        # Number of interviews
        self.NUM_INTERVIEWS = 9            # Range [0, Inf) suggested = 10
        self.all_interviews = range(self.NUM_INTERVIEWS)

        self.USE_INTERVIEW_LIMITS = True
        self.MIN_INTERVIEWS = 3             # Range [0, self.MAX_INTERVIEWS]
        self.MAX_INTERVIEWS = self.NUM_INTERVIEWS            # Range [0, self.NUM_INTERVIEWS]

        self.USE_EXTRA_SLOTS = True  # Make reccomendations for matches not made

        # Give the faculty an advantage over students range[0, 100], 50 = no
        # advantage, 100 = students don't matter, 0 = faculty don't matter
        self.FACULTY_ADVANTAGE = 90     # Range [0, Inf), suggested = 70

        # Use ranked preferences instead of binary(want/don't want)
        self.USE_RANKING = True     # True if use preference order instead of binary
        # What value is given to the first name in a list of preferences
        self.MAX_RANKING = self.NUM_INTERVIEWS
        # What exponent should be used for ranks? If n, first choice is
        # self.MAX_RANKING ^ n, and last choice is 1 ^ n
        self.CHOICE_EXPONENT = 4

        # Penalize the need to work over lunch
        self.USE_WORK_LUNCH = False
        self.LUNCH_PENALTY = 50000     # Range [0, Inf), suggested = 10
        self.LUNCH_PERIOD = 4       # Range [0, self.NUM_INTERVIEWS]

        # Give recruiting faculty an advantage over non-recruiting faculty
        self.USE_RECRUITING = True
        self.RECRUITING_WEIGHT = 30000     # Range [0, Inf), suggested = 200
        
        # If some people are not available for some (or all) interviews, use
        # this
        self.USE_STUDENT_AVAILABILITY = False
        self.USE_FACULTY_AVAILABILITY = True
        # This parameter probably does not need tweeked,
        # it is just a negative weight that adds a strong cost to making
        # unavailable people interview
        self.AVAILABILITY_VALUE = -1 * 5000

        # A track is a similarity between an interviewer and an interviewee
        # These similarities are represented with integer groups
        # Tracks do not affect well-requested interviews, but are useful for
        # choosing interview matches that may be good, but that were not
        # requested
        self.USE_TRACKS = True
        self.TRACK_WEIGHT = 30000           # Range [0, Inf), suggested = 1

        # When interviewers are similar to the interviewers that are "first choices"
        # for the interviewees, they are given a boost in the objective function
        # if the they were not chosen by the interviewee but the interviewee needs
        # more interviews
        self.USE_FACULTY_SIMILARITY = True
        self.FACULTY_SIMILARITY_WEIGHT = 30000  # Range [0, Inf), suggested = 2
        # Number of similar faculty objective scores to boost, range [0,
        # self.num_faculty), suggested = 5
        self.NUM_SIMILAR_FACULTY = 5

        # When a solution is found, we will print out how many people got their
        # first, second, ..., n^th choices.  This parameter is n
        # Range [0, self.NUM_INTERVIEWS), suggested = 3
        self.NUM_PREFERENCES_2_CHECK = 5

        # After all matches have been made, suggest interesting people to talk with
        # During free time.  This is the number of suggestions to make
        # Range [0, self.num_faculty), suggested = 2
        self.NUM_SUGGESTIONS = 2

        # While matches are being made, choose how many times to check the first
        # self.NUM_PREFERENCES_2_CHECK choices.
        self.CHECK_MATCHES = True
        # range [0, inf), suggested = 20 (when < 20, it can be slow)
        self.CHECK_FREQUENCY = 100

        # Number of seconds to allow the match maker to run
        self.MAX_SOLVER_TIME_SECONDS = 180   # range [0, inf), suggested = 190

        # Choose if we want to print preference data to worksheets
        self.PRINT_STUD_PREFERENCE = False
        self.PRINT_FACULTY_PREFERENCE = True

        # For testing purposes, we may choose to generate fake data.  These parameters
        # affect how many data points are generated
        self.RAND_NUM_STUDENTS = 70         # Range [0, Inf), suggested = 50
        self.RAND_NUM_FACULTY = 31          # Range [0, Inf), suggested = 35
        self.RAND_NUM_INTERVIEWS = 10       # Range [0, Inf), suggested = 10

        # For testing purposes, we may choose to randomize the preference order
        # in order to ensure that the optimizer doesn't favor any students
        self.RANDOMIZE_PREFERENCES = False

        # Initialize empty variables for use later
        self.student_names = []
        self.faculty_names = []

        # An arbitrary small constant to help with numerical stability
        self.A_SMALL_CONSTANT = 1E-10

        # Penalize having empty interview slots
        # This number should be chosen so that it is larger than lunch penalty
        # Avoid using - it's slow
        # Set to zero to not use
        # Range [0, Inf), suggested = 0, suggested to turn on >
        # self.LUNCH_PENALTY ^ 2 (about 500 if using all default parameters)
        self.EMPTY_PENALTY = 0        
        
        # Interrupt
        # def empty_func():
        empty_func = lambda : 'No action taken'
        self.stopSearch = empty_func
        self.is_running = False
        self.is_validating = False
        
        # Check parameter validity
        # InputCheckerNoThrow(self)
        
        

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
        

    ''' Calculate the rankings for students and faculty '''
    
    def calc_ranking(self):
        
        # Student
        student_value = np.sum(self.faculty_pref, axis=1)        
        sorted_values = np.flipud(np.argsort(student_value))
        
        self.student_rank = np.empty((self.num_students, 3), dtype=object)
        self.student_rank[:, 1] = self.student_names
        self.student_rank[:, 2] = student_value
        self.student_rank = self.student_rank[sorted_values, :]
        self.student_rank[:, 0] = np.arange(self.num_students)        
        
        with open(path.join(self.RESULTS_PATH, self.STUDENT_RANKING_FILE), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([ 'Rank', 'Name', 'Score'])
            for row in self.student_rank:
                writer.writerow(row)
                
        
        
        # Faculty
        faculty_value = np.sum(self.student_pref, axis=0)
        sorted_values = np.flipud(np.argsort(faculty_value))
        
        self.faculty_rank = np.empty((self.num_faculty, 3), dtype=object)
        self.faculty_rank[:, 1] = self.faculty_names
        self.faculty_rank[:, 2] = faculty_value
        self.faculty_rank = self.faculty_rank[sorted_values, :]
        self.faculty_rank[:, 0] = np.arange(self.num_faculty)
        
        with open(path.join(self.RESULTS_PATH, self.FACULTY_RANKING_FILE), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([ 'Rank', 'Name', 'Score'])
            for row in self.faculty_rank:
                writer.writerow(row)
        
    def calc_objective_matrix(self):
        self.calc_objective_matrix_base()
        self.adjust_objective_matrix_availability()
        
    ''' Transform the data into objective matrix'''
    
    def calc_objective_matrix_base(self):

        # Determine how to weight faculty preferences over student preferences
        alpha = self.FACULTY_ADVANTAGE
        beta = 100 - alpha
        self.objective_matrix = (
            alpha * np.power(self.faculty_pref, self.CHOICE_EXPONENT)
            + beta * np.power(self.student_pref, self.CHOICE_EXPONENT)).astype(int)

        # Give recruiting faculty an advantage
        if self.USE_RECRUITING:
            self.objective_matrix += (self.is_recruiting *
                                      self.RECRUITING_WEIGHT).astype(np.int64)

        # Add a benefit for being in the same track, but only if currently not
        # matched
        not_matched = self.objective_matrix == 0
        if self.USE_TRACKS:
            add_track_advantage = np.logical_and(
                not_matched, self.same_track == 1)
            self.objective_matrix[add_track_advantage] += self.TRACK_WEIGHT

        # Add a benefit to similar faculty, if not matched, for students top n
        # faculty
        if self.USE_FACULTY_SIMILARITY:
            for s in self.all_students:
                for p in range(self.NUM_SIMILAR_FACULTY):
                    match_benefit = self.MAX_RANKING - p
                    faculty_choice = np.where(
                        self.student_pref[s, :] == match_benefit)
                    if np.shape(faculty_choice)[1] > 0:
                        was_not_matched = np.where(not_matched[s, :])
                        similar_faculty = self.faculty_similarity[was_not_matched,
                                                                  faculty_choice]
                        self.objective_matrix[s, was_not_matched] += similar_faculty * \
                            self.FACULTY_SIMILARITY_WEIGHT

        # Expand the objective_matrix to cover each interview period
        self.objective_matrix = np.reshape(
            self.objective_matrix, (1, self.num_students, self.num_faculty))
        self.objective_matrix = np.repeat(
            self.objective_matrix, self.NUM_INTERVIEWS, axis=0)

        # Add a cost for working during lunch
        if self.USE_WORK_LUNCH:
            # The 2 is the maximum number of points we can remove for lunch
            # weight because of response_to_weight
            self.objective_matrix[self.LUNCH_PERIOD, :, :] -= (
                (2 - self.will_work_lunch) * self.LUNCH_PENALTY).astype(np.int64)

    ''' Make the availability a cost '''
    def adjust_objective_matrix_availability(self):
        # Add not available slots as cost
        # THIS CODE MUST COME LAST WHEN CALCULATING COST
        if self.USE_FACULTY_AVAILABILITY:
            i_unavail, f_unavail = np.where(self.faculty_availability == 0)
            self.objective_matrix[i_unavail, :,
                                  f_unavail] = self.AVAILABILITY_VALUE

        if self.USE_STUDENT_AVAILABILITY:
            i_unavail, s_unavail = np.where(self.student_availability == 0)
            self.objective_matrix[i_unavail,
                                  s_unavail, :] = self.AVAILABILITY_VALUE
       
    ''' Check if availability is respected '''
    
    def check_availability(self):
        
        # Faculty
        if self.USE_FACULTY_AVAILABILITY:
            requested_off = np.where(self.faculty_availability == 0)
            not_respected = self.results[requested_off[0],
                                         :,
                                         requested_off[1]] == 1
            problems = np.where(not_respected == True)
            try:
                faculty_names = self.faculty_names[requested_off[1][problems[0]]]
            except:
                raise ValueError('problems[1] out of bounds')
                
            faculty_names = np.unique(faculty_names)
            
            self.print('')
            self.print('**************************************')
            self.print('Faculty schedules not respected')
            if len(faculty_names) > 0:
                self.print(faculty_names)
            else:
                self.print(' -- None --')
        
        # Students
        if self.USE_STUDENT_AVAILABILITY:
            requested_off = np.where(self.student_availability == 0)
            not_respected = self.results[requested_off[0],
                                         requested_off[1],
                                         :] == 1
            problems = np.where(not_respected == True)
            student_names = self.student_names[requested_off[1][problems[0]]]
            student_names = np.unique(student_names)
            
            self.print('Student schedules not respected')
            if len(student_names) > 0:
                self.print(student_names)
            else:
                self.print(' -- None --')
            self.print('**************************************')
            self.print('')
        
    ''' Check if lunch preferences are respected '''
    
    def check_lunch(self):
        
        if not self.USE_WORK_LUNCH:
            return
        
        lunch_results = self.results[self.LUNCH_PERIOD, :, :]
        work_during_lunch = np.sum(lunch_results, axis=0)
        
        
        request_off = np.where(self.will_work_lunch == 1)
        demand_off = np.where(self.will_work_lunch == 0)
        
        request_status = np.concatenate((np.reshape(self.faculty_names[request_off], (-1, 1)),
        								 np.reshape(work_during_lunch[request_off] == 0, (-1, 1))),
        								 axis=1)
        demand_status = np.concatenate((np.reshape(self.faculty_names[demand_off], (-1, 1)),
        								 np.reshape(work_during_lunch[demand_off] == 0, (-1, 1))),
        								 axis=1)
        
        self.print('')
        self.print('**************************************')
        self.print('Lunch Preference Respected:')
        self.print('Request:')
        self.print(request_status)
        self.print('')
        self.print('Demand:')
        self.print(demand_status)
        self.print('**************************************')
        self.print('')

    ''' Track how many people got their preferences '''

    def check_preferences(self, matches):

        # Students
        student_pref = self.student_pref * matches
        self.student_pref_objective = np.sum(student_pref, axis=1)
        total_preferences = np.empty((self.NUM_PREFERENCES_2_CHECK))
        preferences_met = np.empty((self.NUM_PREFERENCES_2_CHECK))
        self.stud_pref_met = np.empty(
            (self.NUM_PREFERENCES_2_CHECK)).astype(object)
        for pref_num in range(self.NUM_PREFERENCES_2_CHECK):
            total_preferences[pref_num] = np.sum(
                self.student_pref == (self.MAX_RANKING - pref_num))
            self.stud_pref_met = student_pref == (self.MAX_RANKING - pref_num)
            preferences_met[pref_num] = np.sum(self.stud_pref_met)

        self.student_fraction_preferences_met = preferences_met / total_preferences
        
        self.print('')
        self.print('**************************************')
        self.print('Fraction of student preferences met: ')
        self.print(np.reshape(self.student_fraction_preferences_met, (1, -1)))
        self.print('')

        # Faculty
        faculty_pref = self.faculty_pref * matches
        self.faculty_pref_objective = np.sum(faculty_pref, axis=0)
        total_preferences = np.empty((self.NUM_PREFERENCES_2_CHECK))
        preferences_met = np.empty((self.NUM_PREFERENCES_2_CHECK))
        self.faculty_pref_met = np.empty(
            (self.NUM_PREFERENCES_2_CHECK)).astype(object)
        for pref_num in range(self.NUM_PREFERENCES_2_CHECK):
            total_preferences[pref_num] = np.sum(
                self.faculty_pref == (self.MAX_RANKING - pref_num))
            self.faculty_pref_met = faculty_pref == (
                self.MAX_RANKING - pref_num)
            preferences_met[pref_num] = np.sum(self.faculty_pref_met)

        self.faculty_fraction_preferences_met = preferences_met / total_preferences
        self.print('Fraction of faculty preferences met: ')
        self.print(np.reshape(self.faculty_fraction_preferences_met, (1, -1)))
        self.print('**************************************')
        self.print('')
        
        

        

    ''' Randomly generate prefered matches for testing '''

    def define_random_matches(self):

        # Generate random matches
        prof_pref_4_students = np.random.randint(
            1, high=self.RAND_NUM_STUDENTS, size=(
                self.RAND_NUM_STUDENTS, self.RAND_NUM_FACULTY))
        stud_pref_4_profs = np.random.randint(
            1, high=self.RAND_NUM_FACULTY, size=(
                self.RAND_NUM_STUDENTS, self.RAND_NUM_FACULTY))

        # Calculate the objective matrix
        objective_matrix = prof_pref_4_students * stud_pref_4_profs
        objective_matrix = np.reshape(
            objective_matrix, (1, self.RAND_NUM_STUDENTS, self.RAND_NUM_FACULTY))
        objective_matrix = np.repeat(
            objective_matrix, self.RAND_NUM_INTERVIEWS, axis=0)

        self.faculty_pref = prof_pref_4_students
        self.student_pref = stud_pref_4_profs

        # Faculty
        return(prof_pref_4_students, stud_pref_4_profs, objective_matrix)

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

        match_benefit = matches * self.objective_matrix[period]

        # Find good matches for faculty
        for p in self.all_faculty:

            # Find unique benefit levels
            unique_benefits, unique_counts = np.unique(
                match_benefit[:, p], return_counts=True)
            unique_counts = np.flipud(unique_counts)
            unique_benefits = np.flipud(unique_benefits)

            # Don't make 0-benefit suggestions
            unique_counts = unique_counts[unique_benefits > 0]
            unique_benefits = unique_benefits[unique_benefits > 0]

            if np.shape(unique_benefits)[0] > 0:

                # Determine how many benefit levels are needed to reach number of
                # suggestions needed
                summed_counts = np.cumsum(unique_counts)
                bin_needed = np.where(summed_counts > self.NUM_SUGGESTIONS)
                if (np.shape(bin_needed)[0] == 0
                    or np.shape(bin_needed[0])[0] == 0):
                    
                    bin_needed = np.shape(summed_counts)[0] - 1
                else:
                    bin_needed = bin_needed[0][0]                       

                # Use all of the matches from the first few bins
                if bin_needed > 0:
                    good_matches = np.where(
                        match_benefit[:, p] >= unique_benefits[bin_needed - 1])[0]
                    num_matches_made = np.shape(good_matches)[0]
                else:
                    good_matches = np.empty(0)
                    num_matches_made = 0

                # Take random matches from the last bin (because all have equal
                # weight)
                possible_matches = np.where(
                    match_benefit[:, p] == unique_benefits[bin_needed])[0]
                num_matches_needed = self.NUM_SUGGESTIONS - num_matches_made

                if num_matches_needed <= summed_counts[-1]:
                    rand_matches = np.random.choice(
                        possible_matches, size=num_matches_needed)
                    matches = np.concatenate(
                        (good_matches, rand_matches)).astype(int)
                else:
                    matches = np.where(match_benefit[:, p])

            else:
                matches = []

            self.faculty_suggest_matches[p] = self.nice_student_names[matches]

        # Find good matches for students
        for s in self.all_students:

            # Find unique benefit levels
            unique_benefits, unique_counts = np.unique(
                match_benefit[s, :], return_counts=True)
            unique_counts = np.flipud(unique_counts)
            unique_benefits = np.flipud(unique_benefits)

            # Don't make 0-benefit suggestions
            unique_counts = unique_counts[unique_benefits > 0]
            unique_benefits = unique_benefits[unique_benefits > 0]

            if np.shape(unique_benefits)[0] > 0:

                # Determine how many benefit levels are needed to reach number of
                # suggestions needed
                summed_counts = np.cumsum(unique_counts)
                bin_needed = np.where(summed_counts > self.NUM_SUGGESTIONS)
                if np.shape(bin_needed)[0] == 0:
                    bin_needed = np.shape(summed_counts)[0] - 1
                else:
                    bin_needed = bin_needed[0][0]

                # Use all of the matches from the first few bins
                if bin_needed > 0:
                    good_matches = np.where(
                        match_benefit[s, :] >= unique_benefits[bin_needed - 1])[0]
                    num_matches_made = np.shape(good_matches)[0]
                else:
                    good_matches = np.empty(0)
                    num_matches_made = 0

                # Take random matches from the last bin (because all have equal
                # weight)
                possible_matches = np.where(
                    match_benefit[s, :] == unique_benefits[bin_needed])[0]
                num_matches_needed = self.NUM_SUGGESTIONS - num_matches_made

                if num_matches_needed <= summed_counts[-1]:
                    rand_matches = np.random.choice(
                        possible_matches, size=num_matches_needed)
                    matches = np.concatenate(
                        (good_matches, rand_matches)).astype(int)
                else:
                    matches = np.where(match_benefit[:, p])

            else:
                matches = []

            self.stud_suggest_matches[s] = self.nice_faculty_names[matches]

    '''
        Check if an integer is odd
    '''

    def is_odd(self, num):
        return num & 0x1

    ''' Determine how many cpus to use '''

    def get_cpu_2_use(self):

        num_cpus_avail = multiprocessing.cpu_count()

        if sys.platform == "linux" or sys.platform == "linux2":
            self.num_cpus = num_cpus_avail
        elif sys.platform == "darwin":
            self.num_cpus = num_cpus_avail
        elif sys.platform == "win32":
            self.num_cpus = num_cpus_avail - 1  # This improves stability
        else:
            warnings.warn(
                'Operating System not recognized.  Use default number of cores')
            self.num_cpus = num_cpus_avail - 1

        return self.num_cpus

    ''' Get the columns where the preference data is '''
    
    def get_pref_col(self, faculty_pref, student_pref):
                
        FACULTY_PREF_STEM = 'ChoiceStudent'
        STUD_PREF_STEM = 'PreferenceforFacultyInterviewer'

        faculty_col = self.get_pref_loop(FACULTY_PREF_STEM, faculty_pref)
        student_col = self.get_pref_loop(STUD_PREF_STEM, student_pref)

        return faculty_col, student_col
    
    def get_sim_col(self, faculty_pref):
        
        STEM = 'MostSimilarFacultyMember'
        col = self.get_pref_loop(STEM, faculty_pref)
        
        return col
    
    def load_recruiting_data(self, faculty_pref):
        
        STEM = 'recruiting'
        col = self.find_single_column(faculty_pref, STEM)        
        
        self.is_recruiting = faculty_pref[1:, col]        
        self.is_recruiting[self.is_recruiting == ''] = 'No'
        
        self.is_recruiting, error = self.response_to_weight(self.is_recruiting)
        self.response_error_handler(error, 'RECRUITING')
        
    def load_lunch_data(self, faculty_pref):
        STEM = 'lunch'
        
        col = self.find_single_column(faculty_pref, STEM)        
        
        self.will_work_lunch = faculty_pref[1:, col]        
        self.will_work_lunch[self.will_work_lunch == ''] = 'Yes'
        
        self.will_work_lunch, error = self.response_to_weight(self.will_work_lunch)
        
        self.response_error_handler(error, 'LUNCH')
                
    def response_error_handler(self, error, name):
        if not error:
            return
        
        if self.is_validating:
            print('For ' + name + ' data:')
            print(error)
            print('Valid Responses include only: ')
            print('    yes')
            print('    no')
            print('    maybe')
            print('    If it helps me interview students that are important to me')
            print('    If it helps the department...')
            print('')
        else:
            raise ValueError(error)
        
        
    def find_single_column(self, faculty_pref, stem):
        
        num_cols = len(faculty_pref[0])
        
        found = False
        count = 0       
        while not found and count < num_cols:
            
            word = faculty_pref[0, count]
            if word.find(stem) != -1:
                found = True
            else:
                count += 1
        
        if found == False:
            return -1
        else:        
            return count
        

    ''' Get data from multi-column fields '''
    
    def get_pref_loop(self, stem, pref):
        
        NUMBER_SUFFIX = ['st', 'nd', 'rd', 'th']
            
        num_interviews = 0
        for word in pref[0]:
            if word.find(stem) != -1:
                num_interviews += 1

        col = np.zeros(num_interviews)
        for i in range(num_interviews):
            num = i + 1
            if num % 10 == 1 and num != 11:
                suffix = NUMBER_SUFFIX[0]
            elif num % 10 == 2:
                suffix = NUMBER_SUFFIX[1]
            elif num % 10 == 3:
                suffix = NUMBER_SUFFIX[2]
            else:
                suffix = NUMBER_SUFFIX[3]
                
            col_name = str(num) + suffix + stem

            try:
                col[i] = np.where(pref[0] == col_name)[0][0]
                if col[i] == 0:
                    raise ValueError('Field not found: ' + col_name)
            except:
                set_trace()
                raise ValueError('Field not found: ' + col_name)
                    
        return col



    ''' Check what names should be appended to student array '''

    def get_unique_student_names(self, new_names):

        # Find unique student names
        all_students_unique = np.reshape(new_names, (-1, 1))

        for count, name in enumerate(all_students_unique):
            all_students_unique[count] = name[0]
        new_student_names, student_idx = np.unique(
            all_students_unique, return_inverse=True)

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
        self.print('The following '
              + str(len(self.student_names))
              + ' student names have been detected:')

        self.print(self.student_names)

    '''
        Load the availability data for students or faculty
    '''

    def load_availability(self, filename, num_expected_available):

        # Load the availability
        try:
            availability = self.load_data_from_human_readable(
                filename).astype(int)
        except:
            print_text = 'Availability data at ' + filename + ' is invalid.  Check that all values are 1''s or 0''s'
            if self.is_validating:
                print(print_text)
            else:
                raise ValueError(print_text)
                

        # Check that the number of availabilities is expected
        [_, num_available] = np.shape(availability)
        if num_available != num_expected_available:
            error_message = 'The availability data does not match the preference data for file: ' + filename
            if self.is_validating:
                print(error_message)
                print('There are two common causes of this issue: ')
                print('    1) Different people are listed in availability and preference .csv files')
                print('    2) Someone filled out the survey more than once (remove all but last entry)')
                print('')
                return None, None
            else:
                raise ValueError(error_message)

        available = np.asarray(
            np.where(
                np.any(
                    availability,
                    axis=0))).squeeze()

        # return
        return availability, available

    '''
        Load faculty similarity matrix
    '''

    def load_faculty_similarity(self, faculty_match_data):

        # Load the matrix data
        col = self.get_sim_col(faculty_match_data)
        similarity = np.transpose(faculty_match_data[1:, col.astype(int)])
        num_pref, num_faculty = np.shape(similarity)
        if num_faculty != self.num_faculty:
            raise ValueError('The number of faculty with similarities does not match the total number of faculty')     
       
        # Create the similarity matrix
        self.faculty_similarity = np.zeros(
                (self.num_faculty, self.num_faculty), dtype=int)
        
        names_not_found = []
        for row_num, row in enumerate(similarity):
            benefit = num_pref - row_num
            for col_num, name in enumerate(row):
                if name != '':
                    faculty_idx = np.where(self.faculty_names == name)
                    if np.shape(faculty_idx)[1] == 0:
                        names_not_found.append(name)
                                                
                    self.faculty_similarity[row_num, faculty_idx[0]] = benefit
        
        unique_unfound_names = np.asarray(np.unique(names_not_found))
        self.print('Faculty similarity - names not found: ')
        self.print(np.reshape(unique_unfound_names, (-1, 1)), '\n')
        if np.shape(unique_unfound_names)[0] == 0:
            self.print('-- None --')
        self.print('')

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
        Load the data
    '''

    def load_data(self):
        
        # Load the Google Forms data
        stud_match_data = []
        file_name = path.join(self.PATH, self.STUDENT_PREF)
        if not path.exists(file_name):
            print('File does not exist: ' + file_name)
            return
        with open(file_name, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                stud_match_data.append(row)

        faculty_match_data = []
        file_name = path.join(self.PATH, self.FACULTY_PREF)
        if not path.exists(file_name):
            print('File does not exist: ' + file_name)
            return
        with open(file_name, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                faculty_match_data.append(row)
                
        # Make the data into numpy arrays
        stud_match_data = np.asarray(stud_match_data)
        faculty_match_data = np.asarray(faculty_match_data)
                
        # Match the names nicely
        if 'Full Name' not in faculty_match_data[0]:
            print('Full Name Data Not Found for Faculty')
            return
        faculty_col = np.where(faculty_match_data[0] == 'Full Name')[0][0]
        self.nice_faculty_names = np.copy(faculty_match_data[1:, faculty_col])
        
        if 'Full Name' not in stud_match_data[0]:
            print('Full Name Data Not Found for Students')
            return
        student_col = np.where(stud_match_data[0] == 'Full Name')[0][0]
        self.nice_student_names = np.copy(stud_match_data[1:, student_col])
        
        all_names = np.concatenate((self.nice_faculty_names,
                                    self.nice_student_names))
        lengths = [len(name) for name in all_names]
        self.max_name_length = np.max(lengths) + 2    # column width for printing schedules
        
        # Remove characters that cause trouble
        faculty_match_data = self.remove_characters(faculty_match_data)
        stud_match_data = self.remove_characters(stud_match_data)

        # Load the interview times
        self.load_interview_times()

        # Load the preference data
        self.load_preference_data(faculty_match_data, stud_match_data)

        # Load the track data
        if self.USE_TRACKS:
            self.load_track_data(faculty_match_data, stud_match_data)

        # Load the faculty similarity data
        if self.USE_FACULTY_SIMILARITY:
            self.load_faculty_similarity(faculty_match_data)

        # Load the lunch and recruiting weight data
        if self.USE_RECRUITING:
            self.load_recruiting_data(faculty_match_data)

        if self.USE_WORK_LUNCH:
            self.load_lunch_data(faculty_match_data)

        # Load the availability data
        if self.USE_STUDENT_AVAILABILITY:
            self.student_availability, self.students_avail = self.load_availability(
                self.STUDENT_AVAILABILITY_NAME, len(self.student_names))
            
        if self.USE_FACULTY_AVAILABILITY:
            # set_trace()
            self.faculty_availability, self.faculty_avail = self.load_availability(
                self.FACULTY_AVAILABILITY_NAME, len(self.faculty_names))          

        # Calculate the objective matrix
        if not self.is_validating:
            self.calc_objective_matrix()


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

    def load_data_from_human_readable(self, filename):

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
        Load the preference data
    '''

    def load_preference_data(self, faculty_match_data, stud_match_data):
       
        # Extract the names without characters that cause trouble
        faculty_col = np.where(faculty_match_data[0] == 'FullName')[0][0]
        faculty_names = faculty_match_data[1:, faculty_col]
        
        student_col = np.where(stud_match_data[0] == 'FullName')[0][0]
        student_names = stud_match_data[1:, student_col]

        # Extract the preferences
        faculty_col, student_col = self.get_pref_col(faculty_match_data, stud_match_data)         
        
        if len(faculty_col) == 0:
            error_message = 'Faculty preference data not found'
            if self.is_validating():
                print(error_message)
                print('Check that the data exists in the faculty preferences csv file')
                return
            else:
                set_trace()
                raise ValueError(error_message)
        if len(student_col) == 0:
            error_message = 'Student preference data not found'
            if self.is_validating():
                print(error_message)
                print('Check that the data exists in the faculty preferences csv file')                
            else:
                set_trace()
                raise ValueError(error_message)
            
        student_pref = np.transpose(stud_match_data[1:, student_col.astype(int)])
        faculty_pref = np.transpose(faculty_match_data[1:, faculty_col.astype(int)])

        # Randomize preferences, if necessary
        # This code doesn't work and is no longer supported
#        if self.RANDOMIZE_PREFERENCES:
#            stud_match_data = self.randomize_preferences(stud_match_data)
#            faculty_match_data = self.randomize_preferences(faculty_match_data)

        # Statistics
        self.num_students = len(student_names)
        self.num_faculty = len(faculty_names)
        self.all_students = range(self.num_students)
        self.all_faculty = range(self.num_faculty)
                
        # Fill-in faculty preferences
        self.faculty_pref = np.zeros((self.num_students, self.num_faculty))

        names_not_found = []
        for p in self.all_faculty:
            temp_pref = faculty_pref[np.where(
                faculty_pref[:, p] != ''), p].flatten()
            pref_num = 0
            for count, pref in enumerate(temp_pref):
                student_num = np.where(student_names == pref)
                if np.shape(student_num)[1] == 0:
                    names_not_found.append(pref)
                else:
                    self.faculty_pref[student_num, p] = self.MAX_RANKING - pref_num
                    pref_num += 1
                
        unique_unfound_names = np.asarray(np.unique(names_not_found))
        self.print('Student names not found: ')        
        if np.shape(unique_unfound_names)[0] == 0:
            self.print('-- None --')
        else:            
            self.print(np.reshape(unique_unfound_names, (-1, 1)), '\n')
            
        self.print('')

        # Fill-in student preferences
        self.student_pref = np.zeros((self.num_students, self.num_faculty))
        
        names_not_found = []
        for s in self.all_students:
            temp_pref = student_pref[np.where(
                student_pref[:, s] != ''), s].flatten()
            pref_num = 0
            for count, pref in enumerate(temp_pref):
                faculty_num = np.where(faculty_names == pref)
                if np.shape(faculty_num)[1] == 0:
                    names_not_found.append(pref)
                else:
                    self.student_pref[s, faculty_num] = self.MAX_RANKING - pref_num
                    pref_num += 1
        unique_unfound_names = np.asarray(np.unique(names_not_found))
        self.print('Faculty names not found: ')
        self.print(np.reshape(unique_unfound_names, (-1, 1)), '\n')
        if np.shape(unique_unfound_names)[0] == 0:
            self.print('-- None --')
        self.print('')
        
        # Assign object names
        self.student_names = student_names
        self.faculty_names = faculty_names

    '''
        Load track data
        A "track" is a field of specialty.  The idea is to match students and
        faculty who have the same specialty.
    '''

    def load_track_data(self, faculty_match_data, stud_match_data):
        
        TRACK_STEM = 'Track'
        faculty_col = np.where(faculty_match_data[0] == TRACK_STEM)[0][0]
        student_col = np.where(stud_match_data[0] == TRACK_STEM)[0][0]
        
        # Get the track data from files
        self.faculty_tracks = faculty_match_data[1:, faculty_col]
        self.student_tracks = stud_match_data[1:, student_col]

        # Find students and faculty that are in the same track
        try:
            all_tracks = np.concatenate(
                (self.faculty_tracks, self.student_tracks), axis=0)
        except:
           raise ValueError('There is a typo in the tracks data.  Check that there is one row of names and one row of data') 
           
        unique_tracks, unique_idx = np.unique(all_tracks, return_inverse=True)

        self.same_track = np.zeros((self.num_students, self.num_faculty))
        for count, track in enumerate(unique_tracks):
            if track != 'None' and track != '':
                same_track = np.asarray(np.where(unique_idx == count))
                faculty_nums = np.reshape(
                    same_track[same_track < self.num_faculty], (1, -1))
                student_nums = np.reshape(
                    same_track[same_track >= self.num_faculty] - self.num_faculty - 1, (-1, 1))

                self.same_track[student_nums, faculty_nums] = 1

    def print(self, message, sep_char=''):
        
        if type(message) == str:
            print(message, sep=sep_char, flush=True)
            
            try:
                self.log_file.writelines(message + '\n')
            except:
                self.log_file.writelines('?????? List Message could not be printed ??????\n')
                
        elif type(message) == list and type(message[0]) == str:
            
            print(*message, sep=sep_char, flush=True)
            
            try:
                for line in message:
                    self.log_file.writelines(line + '\n')
            except:
                self.log_file.writelines('?????? List Message could not be printed ??????\n')
        else:
            
            print(message, sep=sep_char, flush=True)
            try:
                for line in message:
                    for cell in line:
                        self.log_file.writelines(cell.astype(str))
                        self.log_file.writelines(', ')
                    self.log_file.writelines('\n')
            except:
                try:
                    message = np.reshape(message, (-1, 1))
                    for line in message:
                        for cell in line:
                            self.log_file.writelines(cell.astype(str))
                            self.log_file.writelines(', ')
                        self.log_file.writelines('\n')
                except:
                    self.log_file.writelines('?????? Array Message could not be printed ??????\n')



    ''' Print all of the attributes '''
    def print_atributes(self):

        fields = dir(self)
        
        for field in fields:            
            # All constants should have capital letters
            if np.all(field == field.upper()):
                atr = getattr(self, field)
                self.print(field + ' = ' + str(atr))
                
        self.print('**************************************')
        self.print('')
            

    '''
        Make the matches
    '''

    def main(self):
        
        # Check parameter validity
        self.is_running = True
        icnot = InputCheckerNoThrow(self)
        if not icnot.can_continue:
            self.is_running = False
            return       
                
        # Make the necessary paths
        if not path.isdir(self.RESULTS_PATH):
            makedirs(self.RESULTS_PATH)
        
        log_path = path.join(self.RESULTS_PATH, self.LOG_FILE_NAME)       
        with open(log_path, 'w') as self.log_file:
            
            # Log the attributes used to make the matches
            self.print_atributes()
            
            # Load data
            self.load_data()
            if self.is_validating:
                self.is_running = False
                return
            
            
            
            # Creates the model.
            model = cp_model.CpModel()
    
            # Get objective matrix
            # self.define_random_matches()            
            self.calc_ranking()
            objective_matrix = self.objective_matrix
    
            # Creates interview variables.
            # interview[(p, s, i)]: professor 'p' interviews student 's' for
            # interview number 'i'
            self.interview = {}
            for p in self.all_faculty:
                for s in self.all_students:
                    for i in self.all_interviews:
                        self.interview[(p, s, i)] = model.NewBoolVar(
                            'interview_p%i_s%i_i%i' % (p, s, i))
    
            # Each student has no more than one interview at a time
            for p in self.all_faculty:
                for i in self.all_interviews:
                    model.Add(sum(self.interview[(p, s, i)]
                                  for s in self.all_students) <= 1)
    
            # Each professor has no more than one student per interview
            for s in self.all_students:
                for i in self.all_interviews:
                    model.Add(sum(self.interview[(p, s, i)]
                                  for p in self.all_faculty) <= 1)
    
            # No student is assigned to the same professor twice
            for s in self.all_students:
                for p in self.all_faculty:
                    model.Add(sum(self.interview[(p, s, i)]
                                  for i in self.all_interviews) <= 1)
    
            if self.USE_INTERVIEW_LIMITS:
    
                # Ensure that no student gets too many or too few interviews
                for s in self.all_students:
                    num_interviews_stud = sum(self.interview[(
                        p, s, i)] for p in self.all_faculty for i in self.all_interviews)
    
                    # Set minimum number of interviews
                    if not self.USE_STUDENT_AVAILABILITY:
                        model.Add(self.MIN_INTERVIEWS <= num_interviews_stud)
                    else:
    
                        num_slots_unavailable = sum(
                            self.student_availability[:, s] == 0)
    
                        # If the person is available for more than half the interview
                        # try not to penalize them for being unavailable.  Otherwise,
                        # let them be penalized
                        if num_slots_unavailable <= 0.5 * self.NUM_INTERVIEWS:
                            model.Add(self.MIN_INTERVIEWS <= num_interviews_stud)
                        elif num_slots_unavailable != self.NUM_INTERVIEWS:
                            model.Add(self.MIN_INTERVIEWS - num_slots_unavailable
                                      <= num_interviews_stud)
                        # else:
                        # we don't let them have interviews if they aren't
                        # available
    
                    # Set maximum number of interviews
                    model.Add(num_interviews_stud <= self.MAX_INTERVIEWS)
    
                # Ensure that no professor gets too many or too few interviews
                for p in self.all_faculty:
                    num_interviews_prof = sum(self.interview[(
                        p, s, i)] for s in self.all_students for i in self.all_interviews)
    
                    # If the person is available for more than half the interview
                    # try not to penalize them for being unavailable.  Otherwise,
                    # let them be penalized
                    if not self.USE_FACULTY_AVAILABILITY:
                        model.Add(self.MIN_INTERVIEWS <= num_interviews_prof)
                    else:
    
                        num_slots_unavailable = sum(
                            self.faculty_availability[:, p] == 0)
                        if num_slots_unavailable <= 0.5 * self.NUM_INTERVIEWS:
                            model.Add(self.MIN_INTERVIEWS <= num_interviews_prof)
                        elif num_slots_unavailable != self.NUM_INTERVIEWS:
                            model.Add(self.MIN_INTERVIEWS - int(num_slots_unavailable)
                                      <= num_interviews_prof)
                        # else:
                        # we don't let them have interviews if they aren't
                        # available
    
                    # Set maximum number of interviews
                    model.Add(num_interviews_prof <= self.MAX_INTERVIEWS)
    
            # Define the maximization of the objective
            self.print('Building Maximization term...')
            if self.EMPTY_PENALTY != 0:
                model.Maximize(
                    sum(objective_matrix[i][s][p] * self.interview[(p, s, i)]
                        + self.EMPTY_PENALTY * self.interview[(p, s, i)]
                        for p in self.all_faculty
                        for s in self.all_students
                        for i in self.all_interviews))
            else:
                model.Maximize(
                    sum(objective_matrix[i][s][p] * self.interview[(p, s, i)]
                        for p in self.all_faculty
                        for s in self.all_students
                        for i in self.all_interviews))
    
            # Creates the solver and solve.
            self.print('Building Model...') 
            # solver = cp_model.CpSolver()
            solver = ORSolver()
            solution_printer = VarArrayAndObjectiveSolutionPrinter(self)           
    
            self.print('Setting up workers...')
            self.get_cpu_2_use()
            solver.parameters = sat_parameters_pb2.SatParameters(
                num_search_workers=self.num_cpus)
            solver.parameters.max_time_in_seconds = self.MAX_SOLVER_TIME_SECONDS
    
            self.print('Solving model...')
            t = threading.Thread(target=solver.SolveWithSolutionCallback, args=(model, solution_printer))
            t.start()
            self.stopSearch = solution_printer.StopSearch
            
            t.join()
            status = solver.return_status()
            self.print(status)
    
            # Collect results
            if status == 'FEASIBLE' or status == 'OPTIMAL':
                results = np.empty(
                    (self.NUM_INTERVIEWS,
                     self.num_students,
                     self.num_faculty))
                for i in self.all_interviews:
                    for p in self.all_faculty:
                        for s in self.all_students:                            
                            results[i][s][p] = solver.Value(
                                self.interview[(p, s, i)])
                
                    
                # Save the results
                self.results = results
                self.solver = solver
                self.matches = np.sum(self.results, axis=0).astype(bool)
    
                # Convert the results to text and save as text files
                self.matches_as_text()
    
                # Write the results to a file
                self.print_numpy_arrays(self.RESULTS_NAME, self.results)
                np.savetxt(path.join(self.RESULTS_PATH, self.MATCHES_CSV_NAME),
                           self.matches, delimiter=",",
                           fmt='%i')
    
                # Check the percentage of preferences met
                self.check_preferences(self.matches)
                self.check_lunch()
                self.check_availability()
    
            else:
                self.print('-------- Solver failed! --------')
            
            
            self.print('-------- Matches Made! --------')
            
        self.is_running = True
    '''
        Convert the boolean matrix to a string matrix
    '''

    def matches_as_text(self):

        # Find good matches that were not made
        self.find_suggested_matches()

        # Interview - Faculty Schedule
        self.faculty_schedule = []
        faculty_objective = np.empty(self.num_faculty).astype(object)
        for p in self.all_faculty:
            temp_list = []
            temp_objective = np.empty(self.NUM_INTERVIEWS)
            for i in self.all_interviews:
                s = 0
                found_match = False
                while s < self.num_students and not found_match:
                    if self.results[i][s][p] == 1:
                        temp_list.append(self.nice_student_names[s])
                        found_match = True
                        temp_objective[i] = self.objective_matrix[i, s, p]
                    s += 1
                if not found_match:
                    temp_list.append('Free')
            self.faculty_schedule.append(temp_list)
            faculty_objective[p] = temp_objective

        self.print_schedules('Faculty', self.FACULTY_SCHEDULES_DIR,
                             self.nice_faculty_names, self.faculty_schedule,
                             self.faculty_suggest_matches, faculty_objective)

        # Interview - Student Schedule
        self.student_schedule = []
        student_objective = np.empty(self.num_students).astype(object)
        for s in self.all_students:
            temp_list = []
            temp_objective = np.empty(self.NUM_INTERVIEWS)
            for i in self.all_interviews:
                p = 0
                found_match = False
                while p < self.num_faculty and not found_match:
                    if self.results[i][s][p] == 1:
                        temp_list.append(self.nice_faculty_names[p])
                        found_match = True
                        temp_objective[i] = self.objective_matrix[i, s, p]
                    p += 1
                if not found_match:
                    temp_list.append('Free')
            self.student_schedule.append(temp_list)
            student_objective[s] = temp_objective

        self.print_schedules('Student', self.STUDENT_SCHEDULES_DIR,
                             self.nice_student_names, self.student_schedule,
                             self.stud_suggest_matches, student_objective)

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

        filename = path.join(self.RESULTS_PATH,
                             self.MATCHES_TXT_NAME)
        np.savetxt(filename, matches_2_print,
                   delimiter="", fmt='%15s')

    '''
        Print a numpy array as a csv file
    '''

    def print_numpy_arrays(self, file_name, array):
        with open(path.join(self.RESULTS_PATH, file_name), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for i in self.all_interviews:
                for s in self.all_students:
                    writer.writerow(array[i][s][:])

    '''
        Print schedules
        names1 = people who the schedules are for
        names2 = people on the schedules
        data_array:
                rows = candidates
                columns = people who the schedules are for
        person_string = string to be printed on file
    '''

    def print_schedules(
            self,
            person_string,
            folder_name,
            names1,
            schedule,
            good_matches,
            objective):
        
        # Constants
        HEADING_LAYER = 3
        NAME_LAYER = 0

        # Get the interview times
        times = np.asarray(self.interview_times)
        times.flatten()
        schedule = np.asarray(schedule)

        # Make the folder, if it doesn't exist
        if not path.exists(path.join(self.RESULTS_PATH, folder_name)):
            makedirs(path.join(self.RESULTS_PATH, folder_name))
            
        # Get the thresholds for determining strength of match
        # The strong threshold is chosen so that, if one person chooses a
        # second, but the second doesn't choose the first, the match will only 
        # be moderate (approximately)
        strong_threshold = ((self.MAX_INTERVIEWS) ** self.CHOICE_EXPONENT) * 50
        moderate_threshold = ((0.25 * self.MAX_INTERVIEWS) ** self.CHOICE_EXPONENT) * 100
        
        # Initialize the PDF file
        pw = PDFWriter()

        # Print the results
        for count, name in enumerate(names1):

            # Determine the file name
            file_name = name.replace(' ', '').replace(',', '') + '.pdf'
            file_name = path.join(self.RESULTS_PATH, folder_name, file_name)
            
            # Header            
            text = np.empty((self.NUM_INTERVIEWS + 9, 3), dtype=object)            
            text[NAME_LAYER, 0] = person_string + ' interview schedule for:'
            text[NAME_LAYER, 2] = name            
            
            text[HEADING_LAYER, 0] = 'Time:'
            text[HEADING_LAYER, 1] = 'Person:'

            # Schedule
            layer_num = HEADING_LAYER
            for i in self.all_interviews:
                layer_num += 1
                
                # Print times and matches
                text[layer_num, 0] = np.array_str(times[i])
                text[layer_num, 1] = schedule[count, i]
                
                # Print match quality, if desired
                if (self.PRINT_STUD_PREFERENCE and person_string == 'Student'
                    or self.PRINT_FACULTY_PREFERENCE and person_string == 'Faculty'):
                    text[HEADING_LAYER, 2] = 'Match Quality:'
                    
                    # Change the objective value to something easier to understand
                    # Also, make it strictly positive so that it looks like
                    # there is "always a benefit"
                    obj = objective[count][i]
    
                    if obj < moderate_threshold:
                        if schedule[count, i] == 'Free':
                            match_string = 'Free'
                        else:
                            match_string = 'Informational Interview'
                    elif obj >= moderate_threshold and obj <= strong_threshold:
                        match_string = 'Moderate Match'
                    elif obj > strong_threshold:
                        match_string = 'Strong Match'
                    
                    text[layer_num, 2] = match_string


            # Suggested matches
            text[layer_num + 3, 0] = 'During the open interview periods, we suggest you meet with:'

            for match_count, match in enumerate(good_matches[count]):
                text[layer_num + match_count + 4, 1] = match
        
            text[text == None] = ''
            # pw = pdf_writer()
            chart_limits = np.asarray([HEADING_LAYER + 1, layer_num])
            pw.make_pdf_file(file_name, text, chart_limits)

    '''
        Randomize preferences for debuggning purposes
    '''

    def randomize_preferences(self, pref_array):
        num_matches, num_people = np.shape(pref_array)
        possible_matches = np.arange(num_matches)

        for person in range(num_people):
            rand_permute = np.random.choice(
                possible_matches, size=num_matches)
            pref_array[:, person] = pref_array[rand_permute, person]

    ''' Remove characters that cause trouble '''
    def remove_characters(self, char_array):

        for row_num, row in enumerate(char_array):
            for cell_num, cell in enumerate(row):
                char_array[row_num, cell_num] = char_array[row_num, cell_num].replace(' ', '')
                char_array[row_num, cell_num] = char_array[row_num, cell_num].replace(',', '')
                
        return char_array


    '''
        Remove students and faculty that are unavailable
    '''

    def remove_unavailable(self):

        # Availability
        self.student_availability = self.student_availability[:,
                                                              self.students_avail]
        self.faculty_availability = self.faculty_availability[:,
                                                              self.faculty_avail]

        # Names
        self.student_names = np.asarray(self.student_names)[
            self.students_avail].tolist()
        self.faculty_names = np.asarray(self.faculty_names)[
            self.faculty_avail].tolist()

        # Match Preferences
        temp_stud_avail = np.reshape(self.students_avail, (-1, 1))
        temp_faculty_avail = np.reshape(self.faculty_avail, (1, -1))
        self.student_pref = self.student_pref[temp_stud_avail,
                                              temp_faculty_avail]
        self.faculty_pref = self.faculty_pref[temp_stud_avail,
                                              temp_faculty_avail]

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
        error = None
        array = array.flatten()

        out_array = np.empty(np.shape(array))
        for count, response in enumerate(array):
            if response.lower() == 'yes':
                out_array[count] = 2
            elif response.lower() == 'no':
                out_array[count] = 0
            elif response.lower() == 'maybe':
                out_array[count] = 1                
            elif response.lower() == 'ifithelpsmeinterviewstudentsthatareimportanttome':
                out_array[count] = 1
            elif response.lower() == 'ifithelpsthedepartment...':
                out_array[count] = 0
            else:
                error = 'Unknown Response: ' + response
                # raise ResponseError('Response unknown:' + str(response))

        return out_array, error
    
    def set_printer(self, function_handle):
        sys.stdout.write = function_handle
    
    def validate(self):
        
        self.is_validating = True
        print('Errors will appear here:')
        print('-----------------------------------------------------------------')
        self.main()
        print('-----------------------------------------------------------------')
        self.is_validating = False
    
        
if __name__ == '__main__':
     mm = match_maker()
     mm.main()
