from __future__ import division
from __future__ import print_function
import tkinter as tk

import sys
from os import path, makedirs
from ortools.sat import sat_parameters_pb2
from ortools.sat.python import cp_model
from reportlab.pdfgen import canvas
import numpy as np
import csv
import warnings
import multiprocessing






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
            CREATE GOOGLE SURVEYS
            FREEZE
            Create batch emailer
            Alphabetize functions
            Redo matches without redoing everything

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
        self.PATH = "/media/veracrypt1/Users/Cale/Documents/Calers_Writing/PhD/GEC/scheduling_software/2019_data/processed_for_program"
        self.RESULTS_PATH = path.join(self.PATH, 'results')
        self.STUDENT_PREF = "CWRU_BME_Open_House-Students.csv"
        self.FACULTY_PREF = "CWRU_BME_Open_House-Faculty.csv"
        self.TIMES_NAME = "interview_times.csv"
        self.FACULTY_AVAILABILITY_NAME = 'faculty_availability.csv'
        self.STUDENT_AVAILABILITY_NAME = 'student_availability.csv'
        self.STUDENT_RANKING_FILE = 'student_ranking.csv'
        self.FACULTY_RANKING_FILE = 'faculty_ranking.csv'
        self.LOG_FILE_NAME = 'log.txt'
        
        # Make the necessary paths
        if not path.isdir(self.RESULTS_PATH):
            makedirs(self.RESULTS_PATH)

        # Number of interviews
        self.NUM_INTERVIEWS = 9            # Range [0, Inf) suggested = 10
        self.all_interviews = range(self.NUM_INTERVIEWS)

        self.USE_INTERVIEW_LIMITS = True
        self.MIN_INTERVIEWS = 3             # Range [0, self.MAX_INTERVIEWS]
        self.MAX_INTERVIEWS = self.NUM_INTERVIEWS            # Range [0, self.NUM_INTERVIEWS]

        self.USE_EXTRA_SLOTS = True  # Make reccomendations for matches not made

        # Give the faculty an advantage over students range[0, 100], 50 = no
        # advantage, 100 = students don't matter, 0 = faculty don't matter
        self.FACULTY_ADVANTAGE = 70     # Range [0, Inf), suggested = 70

        # Use ranked preferences instead of binary(want/don't want)
        self.USE_RANKING = True     # True if use preference order instead of binary
        # What value is given to the first name in a list of preferences
        self.MAX_RANKING = self.NUM_INTERVIEWS
        # What exponent should be used for ranks? If n, first choice is
        # self.MAX_RANKING ^ n, and last choice is 1 ^ n
        self.CHOICE_EXPONENT = 4

        # Penalize the need to work over lunch
        self.USE_WORK_LUNCH = True
        self.LUNCH_PENALTY = 50000     # Range [0, Inf), suggested = 10
        self.LUNCH_PERIOD = 4       # Range [0, self.NUM_INTERVIEWS]

        # Give recruiting faculty an advantage over non-recruiting faculty
        self.USE_RECRUITING = True
        self.RECRUITING_WEIGHT = 30000     # Range [0, Inf), suggested = 200
        
        # If some people are not available for some (or all) interviews, use
        # this
        self.USE_STUDENT_AVAILABILITY = True
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
        # range [0, inf), suggested = 20 (when > 20, it can be slow)
        self.CHECK_FREQUENCY = 20

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

        # Check parameter validity
        input_checker(self)
        
        

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
        
        
    ''' Transform the data into objective matrix'''
    
    def calc_objective_matrix(self):

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
                
        FACULTY_PREF_STEM = 'PreferenceforStudents'
        STUD_PREF_STEM = 'PreferenceforFaculty'

        faculty_col = self.get_pref_loop(FACULTY_PREF_STEM, faculty_pref)
        student_col = self.get_pref_loop(STUD_PREF_STEM, student_pref)

        return faculty_col, student_col
    
    def get_sim_col(self, faculty_pref):
        
        STEM = 'MostSimilarFaculty'
        col = self.get_pref_loop(STEM, faculty_pref)
        
        return col
    
    def load_recruiting_data(self, faculty_pref):
        
        STEM = 'recruiting'
        col = self.find_single_column(faculty_pref, STEM)        
        
        self.is_recruiting = faculty_pref[1:, col]        
        self.is_recruiting[self.is_recruiting == ''] = 'No'
        
        self.is_recruiting = self.response_to_weight(self.is_recruiting)
        
    def load_lunch_data(self, faculty_pref):
        STEM = 'lunch'
        
        col = self.find_single_column(faculty_pref, STEM)        
        
        self.will_work_lunch = faculty_pref[1:, col]        
        self.will_work_lunch[self.will_work_lunch == ''] = 'Yes'
        
        self.will_work_lunch = self.response_to_weight(self.will_work_lunch)
        
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
            raise ValueError('Availability data at ' + filename + ' is invalid.  Check that all values are 1''s or 0''s')

        # Check that the number of availabilities is expected
        [_, num_available] = np.shape(availability)
        if num_available != num_expected_available:
            raise ValueError(
                'The availability data does not match the preference data')

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
        with open(path.join(self.PATH, self.STUDENT_PREF), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                stud_match_data.append(row)

        faculty_match_data = []
        with open(path.join(self.PATH, self.FACULTY_PREF), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                faculty_match_data.append(row)
                
        # Make the data into numpy arrays
        stud_match_data = np.asarray(stud_match_data)
        faculty_match_data = np.asarray(faculty_match_data)
                
        # Match the names nicely
        faculty_col = np.where(faculty_match_data[0] == 'Full Name')[0][0]
        self.nice_faculty_names = np.copy(faculty_match_data[1:, faculty_col])
        
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
            self.faculty_availability, self.faculty_avail = self.load_availability(
                self.FACULTY_AVAILABILITY_NAME, len(self.faculty_names))          

        # Calculate the objective matrix
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
            raise ValueError('Faculty preference data not found')
        if len(student_col) == 0:
            raise ValueError('Faculty preference data not found')
            
        student_pref = np.transpose(stud_match_data[1:, student_col.astype(int)])
        faculty_pref = np.transpose(faculty_match_data[1:, faculty_col.astype(int)])

        # Randomize preferences, if necessary
        if self.RANDOMIZE_PREFERENCES:
            stud_match_data = self.randomize_preferences(stud_match_data)
            faculty_match_data = self.randomize_preferences(faculty_match_data)

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
        input_checker(self)        
        
        with open(path.join(self.RESULTS_PATH, self.LOG_FILE_NAME), 'w') as self.log_file:

            # Log the attributes used to make the matches
            self.print_atributes()
            
            # Creates the model.
            model = cp_model.CpModel()
    
            # Get objective matrix
            # self.define_random_matches()
            self.load_data()
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
                            model.Add(self.MIN_INTERVIEWS - num_slots_unavailable
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
            solver = cp_model.CpSolver()
            solution_printer = VarArrayAndObjectiveSolutionPrinter(self)
    
            self.print('Setting up workers...')
            self.get_cpu_2_use()
            solver.parameters = sat_parameters_pb2.SatParameters(
                num_search_workers=self.num_cpus)
            solver.parameters.max_time_in_seconds = self.MAX_SOLVER_TIME_SECONDS
    
            self.print('Solving model...')
            status = solver.SolveWithSolutionCallback(model, solution_printer)
    
            self.print(solver.StatusName(status))
    
            # Collect results
            if solver.StatusName(status) == 'FEASIBLE' or solver.StatusName(
                    status) == 'OPTIMAL':
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
                self.print_numpy_arrays('results.csv', self.results)
                np.savetxt(path.join(self.RESULTS_PATH, 'matches.csv'),
                           self.matches, delimiter=",",
                           fmt='%i')
    
                # Check the percentage of preferences met
                self.check_preferences(self.matches)
                self.check_lunch()
                self.check_availability()
    
            else:
                self.print('-------- Solver failed! --------')

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

        self.print_schedules('Faculty', 'faculty_schedules',
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

        self.print_schedules('Student', 'student_schedules',
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
                             'matches.txt')
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
        pw = pdf_writer()

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
                raise ValueError('Response unknown')

        return out_array


'''
    VarArrayAndObjectiveSolutionPrinter
    callback printer object for ortools solver
'''


class VarArrayAndObjectiveSolutionPrinter(cp_model.CpSolverSolutionCallback):

    ''' Print intermediate solutions. '''

    def __init__(self, match_maker):

        cp_model.CpSolverSolutionCallback.__init__(self)

        self.match_maker = match_maker
        self.variables = match_maker.interview

        self.CHECK_MATCHES = self.match_maker.CHECK_MATCHES
        self.CHECK_FREQUENCY = self.match_maker.CHECK_FREQUENCY

        if self.CHECK_MATCHES:
            self.last_stud_percent = np.zeros(
                (1, self.match_maker.NUM_PREFERENCES_2_CHECK))
            self.last_faculty_percent = np.zeros(
                (1, self.match_maker.NUM_PREFERENCES_2_CHECK))

        self.__solution_count = 0

    def on_solution_callback(self):

        # Get sizes
        num_faculty = self.match_maker.num_faculty
        num_students = self.match_maker.num_students
        num_interviews = self.match_maker.NUM_INTERVIEWS

        # Print objective value
        text = 'Solution %i' % self.__solution_count
        try:
            self.match_maker.print(text)
        except Exception as e:
            print('Error!')
            print(e)
        text = '  objective value = %i' % self.ObjectiveValue()
        try:
            self.match_maker.print(text)
        except Exception as e:
            print('Error!')
            print(e)

        # Determine what matches were made
        if self.CHECK_MATCHES and (
                self.__solution_count %
                self.CHECK_FREQUENCY == 0):

            values = np.empty((num_students, num_faculty, num_interviews))
            for p in range(num_faculty):
                for s in range(num_students):
                    for i in range(num_interviews):
                        the_key = (p, s, i)
                        the_variable = self.variables[the_key]
                        values[s, p, i] = self.Value(the_variable)

            values = np.asarray(values)

            # Determine number of matches made for preferences
            matches = np.sum(values, axis=2)
            self.match_maker.check_preferences(matches)

        self.__solution_count += 1

    def solution_count(self):
        return self.__solution_count


'''
    input_checker
    Checks input to the match_maker class to make sure they are reasonable
    Call input_checker(match_maker) as the last line of match_maker.__init__
    If no errors result, the match_maker program can continue
'''


class input_checker:

    def __init__(self, match_maker):
        self.mm = match_maker

        self.main()

    def check_bool(self, parameter):
        return isinstance(parameter, bool)

    def check_file_exists(self, file_name):
        if not isinstance(file_name, str):
            return False

        full_path = path.join(self.mm.PATH, file_name)
        return path.isfile(full_path)

    def check_positive_int(self, parameter):
        if isinstance(parameter, int):
            return parameter >= 0
        return False

    def check_range_int(self, parameter, lower_bound, upper_bound):
        if isinstance(parameter, int):
            return (parameter >= lower_bound and parameter <= upper_bound)
        return False

    def main(self):

        # Check that files exist
        file_names = [
            self.mm.STUDENT_PREF,
            self.mm.FACULTY_PREF,
            self.mm.TIMES_NAME,
            self.mm.FACULTY_AVAILABILITY_NAME,
            self.mm.STUDENT_AVAILABILITY_NAME]

        for file in file_names:
            if not self.check_file_exists(file):
                raise ValueError(file + ' is not on the path ' + self.mm.PATH)

        # Check bools
        if not self.check_bool(self.mm.USE_INTERVIEW_LIMITS):
            raise ValueError('USE_INTERVIEW_LIMITS' + ' should be a bool')

        if not self.check_bool(self.mm.USE_EXTRA_SLOTS):
            raise ValueError('USE_EXTRA_SLOTS' + ' should be a bool')

        if not self.check_bool(self.mm.USE_RANKING):
            raise ValueError('USE_RANKING' + ' should be a bool')

        if not self.check_bool(self.mm.USE_WORK_LUNCH):
            raise ValueError('USE_WORK_LUNCH' + ' should be a bool')

        if not self.check_bool(self.mm.USE_RECRUITING):
            raise ValueError('USE_RECRUITING' + ' should be a bool')

        if not self.check_bool(self.mm.USE_STUDENT_AVAILABILITY):
            raise ValueError('USE_STUDENT_AVAILABILITY' + ' should be a bool')
            
        if not self.check_bool(self.mm.USE_FACULTY_AVAILABILITY):
            raise ValueError('USE_FACULTY_AVAILABILITY' + ' should be a bool')

        if not self.check_bool(self.mm.USE_FACULTY_SIMILARITY):
            raise ValueError('USE_FACULTY_SIMILARITY' + ' should be a bool')

        if not self.check_bool(self.mm.CHECK_MATCHES):
            raise ValueError('CHECK_MATCHES' + ' should be a bool')

        if not self.check_bool(self.mm.PRINT_STUD_PREFERENCE):
            raise ValueError('PRINT_STUD_PREFERENCE' + ' should be a bool')
            
        if not self.check_bool(self.mm.PRINT_FACULTY_PREFERENCE):
            raise ValueError('PRINT_FACULTY_PREFERENCE' + ' should be a bool')    
        

        # Check positive ints
        if not self.check_positive_int(self.mm.NUM_INTERVIEWS):
            raise ValueError(
                'NUM_INTERVIEWS' +
                ' should be a non-negative integer')

        if not self.check_positive_int(self.mm.MIN_INTERVIEWS):
            raise ValueError(
                'MIN_INTERVIEWS' +
                ' should be a non-negative integer')

        if not self.check_positive_int(self.mm.MAX_INTERVIEWS):
            raise ValueError(
                'MAX_INTERVIEWS' +
                ' should be a non-negative integer')

        if not self.check_positive_int(self.mm.MAX_RANKING):
            raise ValueError(
                'MAX_RANKING' +
                ' should be a non-negative integer')

        if not self.check_positive_int(self.mm.CHOICE_EXPONENT):
            raise ValueError(
                'CHOICE_EXPONENT' +
                ' should be a non-negative integer')

        if not self.check_positive_int(self.mm.LUNCH_PENALTY):
            raise ValueError(
                'LUNCH_PENALTY' +
                ' should be a non-negative integer')

        if not self.check_positive_int(self.mm.RECRUITING_WEIGHT):
            raise ValueError(
                'RECRUITING_WEIGHT' +
                ' should be a non-negative integer')

        if not self.check_positive_int(self.mm.TRACK_WEIGHT):
            raise ValueError(
                'TRACK_WEIGHT' +
                ' should be a non-negative integer')

        if not self.check_positive_int(self.mm.FACULTY_SIMILARITY_WEIGHT):
            raise ValueError(
                'FACULTY_SIMILARITY_WEIGHT' +
                ' should be a non-negative integer')

        if not self.check_positive_int(self.mm.NUM_SUGGESTIONS):
            raise ValueError(
                'NUM_SUGGESTIONS' +
                ' should be a non-negative integer')

        if not self.check_positive_int(self.mm.CHECK_FREQUENCY):
            raise ValueError(
                'CHECK_FREQUENCY' +
                ' should be a non-negative integer')

        if not self.check_positive_int(self.mm.MAX_SOLVER_TIME_SECONDS):
            raise ValueError(
                'MAX_SOLVER_TIME_SECONDS' +
                ' should be a non-negative integer')

        if not self.check_positive_int(self.mm.RAND_NUM_STUDENTS):
            raise ValueError(
                'RAND_NUM_STUDENTS' +
                ' should be a non-negative integer')

        if not self.check_positive_int(self.mm.RAND_NUM_FACULTY):
            raise ValueError(
                'RAND_NUM_FACULTY' +
                ' should be a non-negative integer')

        if not self.check_positive_int(self.mm.RAND_NUM_INTERVIEWS):
            raise ValueError(
                'RAND_NUM_INTERVIEWS' +
                ' should be a non-negative integer')

        if not self.check_positive_int(self.mm.EMPTY_PENALTY):
            raise ValueError(
                'EMPTY_PENALTY' +
                ' should be a non-negative integer')

        # Check ranged ints
        if not self.check_range_int(self.mm.FACULTY_ADVANTAGE, 0, 100):
            raise ValueError(
                'FACULTY_ADVANTAGE' +
                ' should be an integer between ' +
                '0' +
                ' and ' +
                '100')

        if not self.check_range_int(
                self.mm.LUNCH_PERIOD,
                0,
                self.mm.NUM_INTERVIEWS):
            raise ValueError('LUNCH_PERIOD' + ' should be an integer between '
                             + '0' + ' and ' + 'NUM_INTERVIEWS')

        if not self.check_range_int(
                self.mm.NUM_PREFERENCES_2_CHECK,
                0,
                self.mm.NUM_INTERVIEWS):
            raise ValueError(
                'NUM_PREFERENCES_2_CHECK' +
                ' should be an integer between ' +
                '0' +
                ' and ' +
                'NUM_INTERVIEWS')

        # Check other parameters
        if self.mm.AVAILABILITY_VALUE != -1 * 5000:
            warnings.warn(
                'We detected that AVAILABILITY_VALUE does not equal -1 * 5000.  This can cause issues.')
    
class pdf_writer():
    
    def __init__(self):
        
        self.POINT = 1
        self.INCH = 72
        self.FONT_SIZE = 12
        self.COL_START = [1 * self.INCH,
                          3.0 * self.INCH,
                          5.5 * self.INCH]
        self.RECTANGLE_END = (8.5 - 1) * self.INCH
        self.TEXT_HEIGHT = 14 * self.POINT

        
    def make_pdf_file(self, output_filename, text, chart_lines):
        
        c = canvas.Canvas(output_filename, pagesize=(8.5 * self.INCH, 11 * self.INCH))
        c.setStrokeColorRGB(0,0,0)
        c.setFillColorRGB(0,0,0)
        c.setFont("Helvetica", self.FONT_SIZE * self.POINT)
        v = 10 * self.INCH
        for line_num, line in enumerate(text):
            
            # Hightlight alternating lines
            if (line_num >= chart_lines[0] and
                line_num <= chart_lines[1]):
                
                if (line_num - chart_lines[0]) % 2 == 0:
                    c.setFillColorRGB(0.9, 0.9, 0.9) 
                    c.rect(self.COL_START[0],
                           v - 2 * self.POINT,
                           self.RECTANGLE_END - self.COL_START[0],
                           self.TEXT_HEIGHT,
                           stroke=0,
                           fill=1)
            
            # Write the text
            c.setFillColorRGB(0, 0, 0)
            for col_num, col in enumerate(line):
                string = self.clean_string(col)
                c.drawString(self.COL_START[col_num], v, string)
                
            # Find the height of the next line of text
            v -= self.TEXT_HEIGHT
            
        # Create the file
        c.showPage()
        c.save()
        
    def clean_string(self, string):
        
        string = string.replace('[', '')
        string = string.replace(']', '')
        string = string.replace("'", '')
        
        return string
    
class defaults():
    
     def __init__(self):
        ''' Constants '''

        # Text
        self.PATH = "/media/veracrypt1/Users/Cale/Documents/Calers_Writing/PhD/GEC/scheduling_software/2019_data/processed_for_program"
        self.STUDENT_PREF = "CWRU_BME_Open_House-Students.csv"
        self.FACULTY_PREF = "CWRU_BME_Open_House-Faculty.csv"
        self.TIMES_NAME = "interview_times.csv"
        self.FACULTY_TRACK_FILE_NAME = 'faculty_tracks.csv'
        self.STUDENT_TRACK_FILE_NAME = 'student_tracks.csv'
        self.FACULTY_SIMILARITY_FILE_NAME = 'faculty_similarity.csv'
        self.IS_RECRUITING_FILE_NAME = 'faculty_is_recruiting.csv'
        self.LUNCH_FILE_NAME = 'faculty_work_lunch.csv'
        self.FACULTY_AVAILABILITY_NAME = 'faculty_availability.csv'
        self.STUDENT_AVAILABILITY_NAME = 'student_availability.csv'
        self.STUDENT_RANKING_FILE = 'student_ranking.csv'
        self.FACULTY_RANKING_FILE = 'faculty_ranking.csv'
        
        # Checkbox
        self.USE_INTERVIEW_LIMITS = True
        self.USE_EXTRA_SLOTS = True  # Make reccomendations for matches not made
        self.USE_RANKING = True     # True if use preference order instead of binary
        self.USE_WORK_LUNCH = True
        self.USE_RECRUITING = True
        self.USE_STUDENT_AVAILABILITY = True
        self.USE_FACULTY_AVAILABILITY = True
        self.USE_TRACKS = True
        self.USE_FACULTY_SIMILARITY = True
        self.CHECK_MATCHES = True
        self.PRINT_STUD_PREFERENCE = True
        self.PRINT_FACULTY_PREFERENCE = True
        
        # Integers
        self.NUM_INTERVIEWS = 9            # Range [0, Inf) suggested = 10
        self.MIN_INTERVIEWS = 3             # Range [0, self.MAX_INTERVIEWS]
        self.MAX_INTERVIEWS = self.NUM_INTERVIEWS            # Range [0, self.NUM_INTERVIEWS]
        self.NUM_SUGGESTIONS = 2
        self.FACULTY_ADVANTAGE = 70     # Range [0, Inf), suggested = 70
        self.MAX_RANKING = self.NUM_INTERVIEWS
        self.CHOICE_EXPONENT = 4
        self.LUNCH_PENALTY = 50000     # Range [0, Inf), suggested = 10
        self.LUNCH_PERIOD = 4       # Range [0, self.NUM_INTERVIEWS]
        self.RECRUITING_WEIGHT = 30000     # Range [0, Inf), suggested = 200
        self.AVAILABILITY_VALUE = -1
        self.TRACK_WEIGHT = 30000           # Range [0, Inf), suggested = 1
        self.FACULTY_SIMILARITY_WEIGHT = 30000  # Range [0, Inf), suggested = 2
        self.NUM_SIMILAR_FACULTY = 5
        self.NUM_PREFERENCES_2_CHECK = 5
        self.CHECK_FREQUENCY = 20
        self.MAX_SOLVER_TIME_SECONDS = 180   # range [0, inf), suggested = 190
        self.COLUMN_WIDTH = 25
        self.EMPTY_PENALTY = 0
        
# https://stackoverflow.com/questions/2395431/using-tk.inter-in-python-to-edit-the-title-bar
class tk_title(tk.Frame):
    
    def __init__(self,parent=None, title='NoTitle'):
        tk.Frame.__init__(self,parent)
        self.parent = parent
        self.make_widgets(title)
        
    def make_widgets(self, title):
        # don't assume that self.parent is a root window.
        # instead, call `winfo_toplevel to get the root window
        self.winfo_toplevel().title(title)


class gui():
    
    def __init__(self, matchmaker=None):
        
        # Matchmaker
        if matchmaker != None:
            self.mm = matchmaker
        else:
            self.mm = 'None'
        
        # Defaults
        d = defaults()       
        self.d = d
        
        # Set up master
        self.master = tk.Tk()
        tk_title(self.master, 'MatchMakr') 
        
        # Define checkbox variables
        self.var_min_max = tk.BooleanVar(value=d.USE_INTERVIEW_LIMITS)
        self.var_suggestions = tk.BooleanVar(value=d.USE_EXTRA_SLOTS)
        self.var_check_preferences =tk.BooleanVar(value=d.CHECK_MATCHES)
        self.var_stud_match_qualtiy = tk.BooleanVar(value=d.PRINT_STUD_PREFERENCE)
        self.var_faculty_match_quality = tk.BooleanVar(value=d.PRINT_FACULTY_PREFERENCE)
        self.var_ranked_pref = tk.BooleanVar(value=d.USE_RANKING)
        self.var_stud_avail = tk.BooleanVar(value=d.USE_STUDENT_AVAILABILITY)
        self.var_faculty_avail = tk.BooleanVar(value=d.USE_INTERVIEW_LIMITS)
        
        # tk.Checkbuttons
        check_button_column = 3
        tk.Checkbutton(self.master, text="Use min/max interview number",
                    variable=self.var_min_max).grid(row=7, column=check_button_column, sticky=tk.W)
        tk.Checkbutton(self.master, text="Print suggestions for additional interviews",
                    variable=self.var_suggestions).grid(row=21, column=check_button_column, sticky=tk.W)
        tk.Checkbutton(self.master, text="Check if preferences are met (slower)",
                    variable=self.var_check_preferences).grid(row=22, column=check_button_column, sticky=tk.W)   
        tk.Checkbutton(self.master, text="Print match quality for students",
                    variable=self.var_stud_match_qualtiy).grid(row=23, column=check_button_column, sticky=tk.W)
        tk.Checkbutton(self.master, text="Print match quality for faculty",
                    variable=self.var_faculty_match_quality).grid(row=24, column=check_button_column, sticky=tk.W)
        tk.Checkbutton(self.master, text="Use ranked preferences",
                    variable=self.var_ranked_pref).grid(row=10, column=check_button_column, sticky=tk.W)
        tk.Checkbutton(self.master, text="Use student availability",
                    variable=self.var_stud_avail).grid(row=5, column=check_button_column, sticky=tk.W)
        tk.Checkbutton(self.master, text="Use faculty availability",
                    variable=self.var_faculty_avail).grid(row=4, column=check_button_column, sticky=tk.W)
        
        # String variables
        self.path = tk.StringVar(value=d.PATH)
        self.stud_pref = tk.StringVar(value=d.STUDENT_PREF)
        self.faculty_pref = tk.StringVar(value=d.FACULTY_PREF)
        self.interview_times = tk.StringVar(value=d.TIMES_NAME)
        self.faculty_avail = tk.StringVar(value=d.FACULTY_AVAILABILITY_NAME)
        self.stud_avail = tk.StringVar(value=d.STUDENT_AVAILABILITY_NAME)
        
        # String boxes
        label_column = 0
        tk.Entry_column = 1
        tk.Label(self.master,
                    text='File path:').grid(row=0, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.path).grid(row=0, column=tk.Entry_column, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Student preference file name:').grid(row=1, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.stud_pref).grid(row=1, column=tk.Entry_column, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Faculty preference file name:').grid(row=2, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.faculty_pref).grid(row=2, column=tk.Entry_column, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Interview times file name:').grid(row=3, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.interview_times).grid(row=3, column=tk.Entry_column, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Faculty availability file name:').grid(row=4, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.faculty_avail).grid(row=4, column=tk.Entry_column, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Student availability file name:').grid(row=5, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.stud_avail).grid(row=5, column=tk.Entry_column, sticky=tk.W)
        
        # tk.Entry variables
        self.num_interviews = tk.IntVar(value=d.NUM_INTERVIEWS)
        self.min_interviews = tk.IntVar(value=d.MIN_INTERVIEWS)
        self.max_interviews = tk.IntVar(value=d.MAX_INTERVIEWS)
        self.faculty_advantage = tk.IntVar(value=d.FACULTY_ADVANTAGE)
        self.max_ranking = tk.IntVar(value=d.MAX_RANKING)
        self.choice_exponent = tk.IntVar(value=d.CHOICE_EXPONENT)
        self.lunch_penalty = tk.IntVar(value=d.LUNCH_PENALTY)
        
        self.lunch_period = tk.IntVar(value=d.LUNCH_PERIOD)
        self.recruiting_weight = tk.IntVar(value=d.RECRUITING_WEIGHT)
        self.track_weight = tk.IntVar(value=d.TRACK_WEIGHT)
        self.faculty_sim_weight = tk.IntVar(value=d.FACULTY_SIMILARITY_WEIGHT)
        self.num_similar_faculty = tk.IntVar(value=d.NUM_SIMILAR_FACULTY)
        self.num_pref_2_check = tk.IntVar(value=d.NUM_PREFERENCES_2_CHECK)
        self.num_suggestions = tk.IntVar(value=d.NUM_SUGGESTIONS)
        self.check_frequency = tk.IntVar(value=d.CHECK_FREQUENCY)
        self.max_solver_time_seconds = tk.IntVar(value=d.MAX_SOLVER_TIME_SECONDS)
        self.empty_penalty = tk.IntVar(value=d.EMPTY_PENALTY)
        
        
        # tk.Entry boxes
        tk.Label(self.master,
                    text='Number of interview slots (int > 0):').grid(row=6, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.num_interviews).grid(row=6, column=tk.Entry_column, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Minimum number of interviews (int > 0):').grid(row=7, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.min_interviews).grid(row=7, column=tk.Entry_column, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Maximum number of interviews (int >= 0):').grid(row=8, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.max_interviews).grid(row=8, column=tk.Entry_column, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Faculty preference (ints [0, 100]):').grid(row=9, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.faculty_advantage).grid(row=9, column=tk.Entry_column, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Maximum number of matches to consider (int > 0):').grid(row=10, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.max_ranking).grid(row=10, column=tk.Entry_column, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Preference given to first choice (int > 0):').grid(row=11, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.choice_exponent).grid(row=11, column=tk.Entry_column, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Lunch penalty (int > 0):').grid(row=12, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.lunch_penalty).grid(row=12, column=tk.Entry_column, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Interview period containing lunch (int > 0):').grid(row=13, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.lunch_period).grid(row=13, column=tk.Entry_column, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Recruiting faculty advantage (int > 0):').grid(row=14, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.recruiting_weight).grid(row=14, column=tk.Entry_column, sticky=tk.W)
         
        tk.Label(self.master,
                    text='Maximum solver time in seconds (int > 0):').grid(row=15, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.max_solver_time_seconds).grid(row=15, column=tk.Entry_column, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Penalty for empty interview slots \n(int > 0, AVOID USING):').grid(row=16, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.empty_penalty).grid(row=16, column=tk.Entry_column, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Track advantage (int > 0):').grid(row=17, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.track_weight).grid(row=17, column=1, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Faculty similarity advantage (int > 0):').grid(row=18, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.faculty_sim_weight).grid(row=18, column=1, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Number of similar faculty to match (int > 0):').grid(row=19, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.num_similar_faculty).grid(row=19, column=1, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Number of preferences to check \n during matching (int > 0):').grid(row=20, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.num_pref_2_check).grid(row=20, column=1, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Number of interviews to suggest (int > 0):').grid(row=21, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.num_suggestions).grid(row=21, column=1, sticky=tk.W)
        
        tk.Label(self.master,
                    text='Number of iterations to run before checking\nprogress  (int > 0, big = slow):').grid(row=22, column=label_column, sticky=tk.E)
        tk.Entry(self.master,
                    textvariable=self.check_frequency).grid(row=22, column=1, sticky=tk.W)
        
        # Run tk.Checkbutton
        tk.Checkbutton_column = 4
        tk.Button(self.master, text='Run', command=self.start_matchmaking).grid(row=24, column = tk.Checkbutton_column)
        
        # Mainloop
        self.master.mainloop()
    
    def assign_parameters(self):
        
        ''' String Constants '''
        self.mm.PATH = self.path.get()
        self.mm.STUDENT_PREF = self.stud_pref.get()
        self.mm.FACULTY_PREF = self.faculty_pref.get()
        self.mm.TIMES_NAME = self.interview_times.get()
        self.mm.FACULTY_AVAILABILITY_NAME = self.faculty_avail.get()
        self.mm.STUDENT_AVAILABILITY_NAME = self.stud_avail.get()
        
        self.mm.USE_INTERVIEW_LIMITS = self.var_min_max.get()
        self.mm.USE_EXTRA_SLOTS = self.var_suggestions.get()
        self.mm.USE_RANKING = self.var_ranked_pref.get()
        self.mm.USE_WORK_LUNCH = self.lunch_penalty.get() > 0
        self.mm.USE_RECRUITING = self.recruiting_weight.get() > 0
        self.mm.USE_STUDENT_AVAILABILITY = self.var_stud_avail.get()
        self.mm.USE_FACULTY_AVAILABILITY = self.var_faculty_avail.get()
        self.mm.USE_TRACKS = self.track_weight.get() > 0
        self.mm.USE_FACULTY_SIMILARITY = self.faculty_sim_weight.get() > 0
        self.mm.CHECK_MATCHES = self.var_check_preferences.get()
        self.mm.PRINT_STUD_PREFERENCE = self.var_stud_match_qualtiy.get()
        self.mm.PRINT_FACULTY_PREFERENCE = self.var_faculty_match_quality.get()
        
        self.mm.NUM_INTERVIEWS = self.num_interviews.get()
        self.mm.MIN_INTERVIEWS = self.min_interviews.get()
        self.mm.MAX_INTERVIEWS = self.max_interviews.get()
        self.mm.FACULTY_ADVANTAGE = self.faculty_advantage.get()
        self.mm.MAX_RANKING = self.max_ranking.get()
        self.mm.CHOICE_EXPONENT = self.choice_exponent.get()
        self.mm.LUNCH_PENALTY = self.lunch_penalty.get()
        self.mm.LUNCH_PERIOD = self.lunch_period.get()
        self.mm.RECRUITING_WEIGHT = self.recruiting_weight.get()
        self.mm.TRACK_WEIGHT = self.track_weight.get()
        self.mm.FACULTY_SIMILARITY_WEIGHT = self.faculty_sim_weight.get()
        self.mm.NUM_SIMILAR_FACULTY = self.num_similar_faculty.get()
        self.mm.NUM_PREFERENCES_2_CHECK = self.num_pref_2_check.get()
        self.mm.NUM_SUGGESTIONS = self.num_suggestions.get()
        self.mm.CHECK_FREQUENCY = self.check_frequency.get()
        self.mm.MAX_SOLVER_TIME_SECONDS = self.max_solver_time_seconds.get()
        self.mm.EMPTY_PENALTY = self.empty_penalty.get()
        
        ''' Calculated Constants '''
        self.mm.all_interviews = range(self.mm.NUM_INTERVIEWS)
        self.mm.RESULTS_PATH = path.join(self.mm.PATH, 'results')
        if not path.isdir(self.mm.RESULTS_PATH):
            makedirs(self.mm.RESULTS_PATH)

        
        
    def start_matchmaking(self):
        self.assign_parameters()
        self.mm.main()        



if __name__ == '__main__':
    
    mm = match_maker()
    g = gui(mm)
