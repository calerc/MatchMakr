from __future__ import division
from __future__ import print_function
import multiprocessing
import sys
from ortools.sat import sat_parameters_pb2
import warnings
import numpy as np
from os import path, makedirs
import csv
from ortools.sat.python import cp_model


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
        Allows the optimizer to be time-limited


    Future features:

        Code clarity:
            ENSURE THAT THE DOCUMENTATION IS UP TO SNUFF

        Code function:
            RANDOMIZE PREFERENCES TO ENSURE THAT THERE IS NOT STRUCTURE IN HOW THEY ARE ASSIGNED
            CREATE GOOGLE SURVEYS
            FREEZE

        Code accessibility:
            CREATE GUI
            CREATE PUBLIC GITHUB
                Determine license type and copyright
            FIND SOMEWHERE TO HOST BINARIES
            VIDEO TO YOUTUBE


    KNOWN BUGS:
        Sometimes, the same person is suggested more than once
'''


class match_maker():

    ''' Define parameters needed for scheduling '''

    def __init__(self):
        ''' Constants '''

        # Files to load
        self.PATH = "/home/cale/Desktop/open_house/fresh_start"
        self.STUDENT_PREF = "stud_pref_order.csv"
        self.FACULTY_PREF = "faculty_preferences.csv"
        self.TIMES_NAME = "interview_times.csv"
        self.FACULTY_TRACK_FILE_NAME = 'faculty_tracks.csv'
        self.STUDENT_TRACK_FILE_NAME = 'student_tracks.csv'
        self.FACULTY_SIMILARITY_FILE_NAME = 'faculty_similarity.csv'
        self.IS_RECRUITING_FILE_NAME = 'faculty_is_recruiting.csv'
        self.LUNCH_FILE_NAME = 'faculty_work_lunch.csv'
        self.FACULTY_AVAILABILITY_NAME = 'faculty_availability.csv'
        self.STUDENT_AVAILABILITY_NAME = 'student_availability.csv'

        # Number of interviews
        self.NUM_INTERVIEWS = 10            # Range [0, Inf) suggested = 10
        self.all_interviews = range(self.NUM_INTERVIEWS)

        self.USE_INTERVIEW_LIMITS = True
        self.MIN_INTERVIEWS = 3             # Range [0, self.MAX_INTERVIEWS]
        self.MAX_INTERVIEWS = 10            # Range [0, self.NUM_INTERVIEWS]

        self.USE_EXTRA_SLOTS = True  # Make reccomendations for matches not made
        # Number of reccomendations, range = [0, Inf), suggested = 2
        self.NUM_EXTRA_SLOTS = 2

        # Give the faculty an advantage over students range[0, 100], 50 = no
        # advantage
        self.FACULTY_ADVANTAGE = 50     # Range [0, Inf), suggested = 50

        # Use ranked preferences instead of binary(want/don't want)
        self.USE_RANKING = True     # True if use preference order instead of binary
        # What value is given to the first name in a list of preferences
        self.MAX_RANKING = 10
        # What exponent should be used for ranks? If n, first choice is
        # self.MAX_RANKING ^ n, and last choice is 1 ^ n
        self.CHOICE_EXPONENT = 2

        # Penalize the need to work over lunch
        self.USE_WORK_LUNCH = True
        self.LUNCH_PENALTY = 10     # Range [0, Inf), suggested = 10
        self.LUNCH_PERIOD = 4       # Range [0, self.NUM_INTERVIEWS]

        # Give recruiting faculty an advantage over non-recruiting faculty
        self.USE_RECRUITING = True
        self.RECRUITING_WEIGHT = 10     # Range [0, Inf), suggested = 10

        # If some people are not available for some (or all) interviews, use
        # this
        self.USE_AVAILABILITY = True
        # This parameter probably does not need tweeked
        self.AVAILABILITY_VALUE = -1 * 5000

        # A track is a similarity between an interviewer and an interviewee
        # These similarities are represented with integer groups
        # Tracks do not affect well-requested interviews, but are useful for
        # choosing interview matches that may be good, but that were not
        # requested
        self.USE_TRACKS = True
        self.TRACK_WEIGHT = 1           # Range [0, Inf), suggested = 1

        # When interviewers are similar to the interviewers that are "first choices"
        # for the interviewees, they are given a boost in the objective function
        # if the they were not chosen by the interviewee but the interviewee needs
        # more interviews
        self.USE_FACULTY_SIMILARITY = True
        self.FACULTY_SIMILARITY_WEIGHT = 2  # Range [0, Inf), suggested = 2
        # Number of similar faculty objective scores to boost, range [0,
        # self.num_faculty), suggested = 5
        self.NUM_SIMILAR_FACULTY = 5

        # When a solution is found, we will print out how many people got their
        # first, second, ..., n^th choices.  This parameter is n
        # Range [0, self.NUM_INTERVIEWS), suggested = 3
        self.NUM_PREFERENCES_2_CHECK = 3

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
        self.PRINT_PREFERENCE = True

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

        # Set the column width (# characters) for printing names to the
        # schedules
        self.COLUMN_WIDTH = 22

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

    ''' Transform the data into objective matrix'''

    def calc_objective_matrix(self):

        # Determine how to weight faculty preferences over student preferences
        alpha = self.FACULTY_ADVANTAGE
        beta = 100 - alpha
        self.objective_matrix = (
            alpha * self.faculty_pref
            + beta * self.student_pref).astype(int)

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
        if self.USE_AVAILABILITY:

            # Faculty
            i_unavail, f_unavail = np.where(self.faculty_availability == 0)
            self.objective_matrix[i_unavail, :,
                                  f_unavail] = self.AVAILABILITY_VALUE

            # Students
            i_unavail, s_unavail = np.where(self.student_availability == 0)
            self.objective_matrix[i_unavail,
                                  s_unavail, :] = self.AVAILABILITY_VALUE

        # Square the objective matrix to maximize chance of getting first
        # choice
        objective_sign = np.sign(self.objective_matrix)
        self.objective_matrix = np.power(
            self.objective_matrix, self.CHOICE_EXPONENT)
        self.objective_matrix *= objective_sign

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
        print('Fraction of student preferences met: ')
        print(self.student_fraction_preferences_met)

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
        print('Fraction of faculty preferences met: ')
        print(self.faculty_fraction_preferences_met)

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
                if np.shape(bin_needed)[0] == 0:
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

            self.faculty_suggest_matches[p] = self.student_names[matches]

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

            self.stud_suggest_matches[s] = self.faculty_names[matches]

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
        print('The following '
              + str(len(self.student_names))
              + ' student names have been detected:')

        print(self.student_names)

    '''
        Load the availability data for students or faculty
    '''

    def load_availability(self, filename, num_expected_available):

        # Load the availability
        availability = self.load_data_from_human_readable(
            filename, False).astype(int)

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

    def load_faculty_similarity(self):

        # Load the matrix data
        self.faculty_similarity = self.load_matrix_data(
            self.FACULTY_SIMILARITY_FILE_NAME)

        # Convert to an array
        self.faculty_similarity = np.asarray(
            self.faculty_similarity, dtype=int)

        # Check that the array size is correct
        num_rows, num_columns = np.shape(self.faculty_similarity)
        if num_rows != self.num_faculty or num_columns != self.num_faculty:
            raise ValueError(
                'Faculty similarity size does not match the number of faculty')

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

        # Load the interview times
        self.load_interview_times()

        # Load the preference data
        self.load_preference_data()

        # Load the track data
        if self.USE_TRACKS:
            self.load_track_data()

        # Load the faculty similarity data
        if self.USE_FACULTY_SIMILARITY:
            self.load_faculty_similarity()

        # Load the lunch and recruiting weight data
        if self.USE_RECRUITING:
            self.is_recruiting = self.load_data_from_human_readable(
                self.IS_RECRUITING_FILE_NAME, False)
            self.is_recruiting = self.response_to_weight(self.is_recruiting)

        if self.USE_WORK_LUNCH:
            self.will_work_lunch = self.load_data_from_human_readable(
                self.LUNCH_FILE_NAME, False)
            self.will_work_lunch = self.response_to_weight(
                self.will_work_lunch)

        # Load the availability data
        if self.USE_AVAILABILITY:

            # Student
            self.student_availability, self.students_avail = self.load_availability(
                self.STUDENT_AVAILABILITY_NAME, len(self.student_names))

            # Faculty
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
        student_names = stud_match_data[1, 1:]
        student_names = student_names[np.where(student_names != '')]
        faculty_names = faculty_match_data[1, 1:]
        faculty_names = faculty_names[np.where(faculty_names != '')]

        # Extract the preferences
        student_pref = stud_match_data[3:, 1:]
        faculty_pref = faculty_match_data[3:, 1:]

        # Randomize preferences, if necessary
        if self.RANDOMIZE_PREFERENCES:
            stud_match_data = self.randomize_preferences(stud_match_data)
            faculty_match_data = self.randomize_preferences(faculty_match_data)

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
                student_pref[count, count2] = student_pref[count,
                                                           count2].replace(' ', '')

        for count, pref in enumerate(faculty_pref):
            for count2, pref2 in enumerate(pref):
                faculty_pref[count, count2] = faculty_pref[count,
                                                           count2].replace(' ', '')

        # Fill-in faculty preferences
        self.faculty_pref = np.zeros((self.num_students, self.num_faculty))

        for p in self.all_faculty:
            temp_pref = faculty_pref[np.where(
                faculty_pref[:, p] != ''), p].flatten()
            for count, pref in enumerate(temp_pref):
                student_num = np.where(student_names == pref)
                self.faculty_pref[student_num, p] = self.MAX_RANKING - count

        # Fill-in student preferences
        self.student_pref = np.zeros((self.num_students, self.num_faculty))

        for s in self.all_students:
            temp_pref = student_pref[np.where(
                student_pref[:, s] != ''), s].flatten()
            for count, pref in enumerate(temp_pref):
                faculty_num = np.where(faculty_names == pref)
                self.student_pref[s, faculty_num] = self.MAX_RANKING - count

        # Assign object names
        self.student_names = student_names
        self.faculty_names = faculty_names

    '''
        Load track data
        A "track" is a field of specialty.  The idea is to match students and
        faculty who have the same specialty.
    '''

    def load_track_data(self):

        # Get the track data from files
        self.faculty_tracks = self.load_data_from_human_readable(
            self.FACULTY_TRACK_FILE_NAME)
        self.student_tracks = self.load_data_from_human_readable(
            self.STUDENT_TRACK_FILE_NAME)

        # Find students and faculty that are in the same track
        all_tracks = np.concatenate(
            (self.faculty_tracks, self.student_tracks), axis=1)
        unique_tracks, unique_idx = np.unique(all_tracks, return_inverse=True)

        self.same_track = np.zeros((self.num_students, self.num_faculty))
        for count, track in enumerate(unique_tracks):
            if track != 'None' and track != '':
                same_track = np.asarray(np.where(unique_idx == count))
                faculty_nums = np.reshape(
                    same_track[same_track < self.num_faculty], (1, -1))
                student_nums = np.reshape(
                    same_track[same_track >= self.num_faculty] - self.num_faculty, (-1, 1))

                self.same_track[student_nums, faculty_nums] = 1

    '''
        Make the matches
    '''

    def main(self):

        # Creates the model.
        model = cp_model.CpModel()

        # Get objective matrix
        # self.define_random_matches()
        self.load_data()
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
                if not self.USE_AVAILABILITY:
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
                if not self.USE_AVAILABILITY:
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
        print('Building Maximization term...')
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
        print('Building Model...', flush=True)
        solver = cp_model.CpSolver()
        solution_printer = VarArrayAndObjectiveSolutionPrinter(self)

        print('Setting up workers...', flush=True)
        self.get_cpu_2_use()
        solver.parameters = sat_parameters_pb2.SatParameters(
            num_search_workers=self.num_cpus)
        solver.parameters.max_time_in_seconds = self.MAX_SOLVER_TIME_SECONDS

        print('Solving model...', flush=True)
        status = solver.SolveWithSolutionCallback(model, solution_printer)

        print(solver.StatusName(status))

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
            np.savetxt(path.join(self.PATH, 'matches.csv'),
                       self.matches, delimiter=",",
                       fmt='%i')

            # Check the percentage of preferences met
            self.check_preferences(self.matches)

        else:
            print('-------- Solver failed! --------')

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
                        temp_list.append(self.student_names[s])
                        found_match = True
                        temp_objective[i] = self.objective_matrix[i, s, p]
                    s += 1
                if not found_match:
                    temp_list.append('Free')
            self.faculty_schedule.append(temp_list)
            faculty_objective[p] = temp_objective

        self.print_schedules('Faculty', 'faculty_schedules',
                             self.faculty_names, self.faculty_schedule,
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
                        temp_list.append(self.faculty_names[p])
                        found_match = True
                        temp_objective[i] = self.objective_matrix[i, s, p]
                    p += 1
                if not found_match:
                    temp_list.append('Free')
            self.student_schedule.append(temp_list)
            student_objective[s] = temp_objective

        self.print_schedules('Student', 'student_schedules',
                             self.student_names, self.student_schedule,
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

    '''
        Print schedules
        names1 = people who the schedules are for
        names2 = people on the scheudles
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

        # Get the interview times
        times = np.asarray(self.interview_times)
        times.flatten()
        schedule = np.asarray(schedule)

        # Make the folder, if it doesn't exist
        if not path.exists(path.join(self.PATH, folder_name)):
            makedirs(path.join(self.PATH, folder_name))

        # Print the results
        for count, name in enumerate(names1):

            # Determine the file name
            file_name = name + '.txt'
            file_name = path.join(self.PATH, folder_name, file_name)

            # Open the file for editing
            with open(file_name, 'w') as file:

                # Header
                file.writelines(
                    person_string +
                    ' interivew schedule for:         ' +
                    name +
                    '\n')
                file.writelines('\n')
                file.writelines('\n')

                # Schedule
                if self.PRINT_PREFERENCE:
                    file.writelines(
                        'Time:                     Person:                 Match Quality:\n')
                    for i in self.all_interviews:

                        if self.is_odd(i):
                            sep_char = '+'
                            sep_string = ' +++++++++ '
                        else:
                            sep_char = '-'
                            sep_string = ' --------- '

                        num_spaces_needed = self.COLUMN_WIDTH - \
                            len(schedule[count, i])
                        if num_spaces_needed > 0:
                            space_string = ' ' + sep_char * num_spaces_needed + ' '
                        else:
                            space_string = ' ' + ' '

                        # Change the objective value to something easier to understand
                        # Also, make it strictly positive so that it looks like
                        # there is "always a benefit"
                        obj = objective[count][i]

                        if obj < 1:
                            obj = 1
                        else:
                            try:
                                obj = int(np.log10(obj))
                            except BaseException:
                                warnings.warn('NaN problem for '
                                              + name + ', obj = '
                                              + str(obj))
                                obj = 1

                        if obj < 1:
                            obj = 1

                        if obj == 1:
                            if schedule[count, i] == 'Free':
                                match_string = 'Free'
                            else:
                                match_string = 'Informational Interview'
                        elif obj >= 2 and obj < 4:
                            match_string = 'Moderate Match'
                        elif obj >= 4:
                            match_string = 'Strong Match'

                        file.writelines(np.array_str(times[i]) + sep_string
                                        + schedule[count, i] + space_string
                                        + match_string + '\n')
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
                file.writelines(
                    'During the open interview periods, we suggest you meet with: \n')

                for match_count, match in enumerate(good_matches[count]):
                    if match_count == 0:
                        file.writelines(match)
                    else:
                        file.writelines(', ' + match)

                file.writelines('\n')
                file.writelines('\n')

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
        print('Solution %i' % self.__solution_count, flush=True)
        print('  objective value = %i' % self.ObjectiveValue(), flush=True)

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
            return (parameter >= lower_bound and parameter < upper_bound)
        return False

    def main(self):

        # Check that files exist
        file_names = [
            self.mm.STUDENT_PREF,
            self.mm.FACULTY_PREF,
            self.mm.TIMES_NAME,
            self.mm.FACULTY_TRACK_FILE_NAME,
            self.mm.STUDENT_TRACK_FILE_NAME,
            self.mm.FACULTY_SIMILARITY_FILE_NAME,
            self.mm.IS_RECRUITING_FILE_NAME,
            self.mm.LUNCH_FILE_NAME,
            self.mm.FACULTY_AVAILABILITY_NAME,
            self.mm.STUDENT_AVAILABILITY_NAME]

        for file in file_names:
            if not self.check_file_exists(file):
                raise ValueError(file + ' is not on the path ' + self.PATH)

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

        if not self.check_bool(self.mm.USE_AVAILABILITY):
            raise ValueError('USE_AVAILABILITY' + ' should be a bool')

        if not self.check_bool(self.mm.USE_FACULTY_SIMILARITY):
            raise ValueError('USE_FACULTY_SIMILARITY' + ' should be a bool')

        if not self.check_bool(self.mm.CHECK_MATCHES):
            raise ValueError('CHECK_MATCHES' + ' should be a bool')

        if not self.check_bool(self.mm.PRINT_PREFERENCE):
            raise ValueError('PRINT_PREFERENCE' + ' should be a bool')

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

        if not self.check_positive_int(self.mm.NUM_EXTRA_SLOTS):
            raise ValueError(
                'NUM_EXTRA_SLOTS' +
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

        if not self.check_positive_int(self.mm.COLUMN_WIDTH):
            raise ValueError(
                'COLUMN_WIDTH' +
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


if __name__ == '__main__':

    mm = match_maker()
    mm.main()
