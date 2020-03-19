from os import path
import warnings

'''
    InputChecker
    Checks input to the match_maker class to make sure they are reasonable
    Call input_checker(match_maker) as the last line of match_maker.__init__
    If no errors result, the match_maker program can continue
'''
class InputChecker:

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



'''
    InputCheckerNoThrow
    An InputChecker, but prints errors instead of throwing them
'''
class InputCheckerNoThrow(InputChecker):
    
    def __init__(self, match_maker):
        self.mm = match_maker
        try:
        # if True:
            self.main()
            self.can_continue = True
        except Exception as e:
            print(e)
            self.can_continue = False