from os.path import join


'''
    Translate settings from the MatchMaker gui for the function match_maker
    Called from MathMakr
'''
class Makr2Maker():

    def __init__(self, matchMakr):
        self.match_maker = matchMakr.match_maker
        self.matchMakr = matchMakr
        
    def apply_settings(self):
        settings_dict = self.matchMakr.get_settings_dict()
        
        conversion_dict = {
                            'USE_FACULTY_AVAILABILITY' : 'cb_fac_avail',
                            'PRINT_FACULTY_PREFERENCE' : 'cb_print_fac_pref',
                            'PRINT_STUDENT_PREFERENCE' : 'cb_print_stud_pref',
                            'USE_INTERVIEW_LIMITS' : 'cb_use_inter_limits',
                            'USE_STUDENT_AVAILABILITY' : 'cb_stud_avail',
                            'USE_RANKING' : 'cb_use_rank',
                            'CHECK_FREQUENCY' : 'sb_check_freq',
                            'EMPTY_PENALTY' : 'sb_empty_penalty',
                            'CHOICE_EXPONENT' : 'sb_exp',
                            'FACULTY_ADVANTAGE' : 'sb_fac_advantage',
                            'FACULTY_SIMILARITY_WEIGHT' : 'sb_fac_sim_weight',
                            'LUNCH_PENALTY' :'sb_lunch_penalty',
                            'LUNCH_PERIOD' :'sb_lunch_period',
                            'MAX_INTERVIEWS' : 'sb_max_num_inter',
                            'MIN_INTERVIEWS' : 'sb_min_num_inter',
                            'NUM_SUGGESTIONS' : 'sb_num_extra_matches',
                            'NUM_INTERVIEWS' : 'sb_num_inter',
                            'NUM_PREFERENCES_2_CHECK' : 'sb_num_pref_2_check',
                            'NUM_SIMILAR_FACULTY' : 'sb_num_sim_fac',
                            'RECRUITING_WEIGHT' : 'sb_recruit_weight',
                            'TRACK_WEIGHT' : 'sb_track_weight',
                            'FACULTY_AVAILABILITY_NAME' : 'tb_fac_avail',
                            'FACULTY_PREF' : 'tb_fac_pref',
                            'FACULTY_SCHEDULES_DIR' : 'tb_fac_sched_dir',
                            'LOG_FILE_NAME' : 'tb_log_name',
                            'PATH' : 'tb_path',
                            'STUDENT_AVAILABILITY_NAME' : 'tb_stud_avail',
                            'STUDENT_PREF' : 'tb_stud_pref',
                            'STUDENT_SCHEDULES_DIR' : 'tb_stud_sched_dir'
                           }
        
        for key, value in zip(conversion_dict.keys(), conversion_dict.values()):
            val = settings_dict[value]            
            setattr(self.match_maker, key, val)
            
        # Set parameters that can't be done programmatically
        self.match_maker.MATCHES_CSV_FILE = settings_dict['tb_match'] + '.csv'
        self.match_maker.MATCHES_TXT_FILE = settings_dict['tb_match'] + '.txt'
        self.match_maker.RESULTS_PATH = join(settings_dict['tb_path'], settings_dict['tb_results_dir'])
        
        # Set use parameters based on weights
        if self.match_maker.FACULTY_SIMILARITY_WEIGHT == 0:
            self.match_maker.USE_FACULTY_SIMILARITY = False
        else:
            self.match_maker.USE_FACULTY_SIMILARITY = True
        
        if self.match_maker.LUNCH_PENALTY == 0:
            self.match_maker.USE_WORK_LUNCH = False
        else:
            self.match_maker.USE_WORK_LUNCH = True
        
        if self.match_maker.NUM_PREFERENCES_2_CHECK == 0:
            self.match_maker.USE_EXTRA_SLOTS = False
        else:
            self.match_maker.USE_EXTRA_SLOTS = True
            
        if self.match_maker.NUM_SIMILAR_FACULTY == 0:
            self.match_maker.USE_FACULTY_SIMILARITY = False
        else:
            self.match_maker.USE_FACULTY_SIMILARITY = True 

