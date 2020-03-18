from tkinter import Checkbutton, IntVar, Tk, W, StringVar, Entry, Label, Frame, E, Button



class defaults():
    
     def __init__(self):
        ''' Constants '''

        # Text
        self.PATH = "/media/veracrypt1/Users/Cale/Documents/Calers_Writing/PhD/GEC/scheduling_software/2019_data/processed_for_program"
        self.STUDENT_PREF = "student_preferences.csv"
        self.FACULTY_PREF = "faculty_preferences.csv"
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
        
# https://stackoverflow.com/questions/2395431/using-tkinter-in-python-to-edit-the-title-bar
class tk_title(Frame):
    
    def __init__(self,parent=None, title='NoTitle'):
        Frame.__init__(self,parent)
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
        self.master = Tk()
        tk_title(self.master, 'MatchMakr') 
        
        # Define checkbox variables
        var_min_max = IntVar(value=d.USE_INTERVIEW_LIMITS)
        var_suggestions = IntVar(value=d.USE_EXTRA_SLOTS)
        var_check_preferences =IntVar(value=d.CHECK_MATCHES)
        var_stud_match_qualtiy = IntVar(value=d.PRINT_STUD_PREFERENCE)
        var_faculty_match_quality = IntVar(value=d.PRINT_FACULTY_PREFERENCE)
        var_ranked_pref = IntVar(value=d.USE_RANKING)
        var_stud_avail = IntVar(value=d.USE_STUDENT_AVAILABILITY)
        var_faculty_avail = IntVar(value=d.USE_INTERVIEW_LIMITS)
        
        # Checkbuttons
        check_button_column = 3
        Checkbutton(self.master, text="Use min/max interview number",
                    variable=var_min_max).grid(row=7, column=check_button_column, sticky=W)
        Checkbutton(self.master, text="Print suggestions for additional interviews",
                    variable=var_suggestions).grid(row=21, column=check_button_column, sticky=W)
        Checkbutton(self.master, text="Check if preferences are met (slower)",
                    variable=var_check_preferences).grid(row=22, column=check_button_column, sticky=W)   
        Checkbutton(self.master, text="Print match quality for students",
                    variable=var_stud_match_qualtiy).grid(row=23, column=check_button_column, sticky=W)
        Checkbutton(self.master, text="Print match quality for faculty",
                    variable=var_faculty_match_quality).grid(row=24, column=check_button_column, sticky=W)
        Checkbutton(self.master, text="Use ranked preferences",
                    variable=var_ranked_pref).grid(row=10, column=check_button_column, sticky=W)
        Checkbutton(self.master, text="Use student availability",
                    variable=var_stud_avail).grid(row=5, column=check_button_column, sticky=W)
        Checkbutton(self.master, text="Use faculty availability",
                    variable=var_faculty_avail).grid(row=4, column=check_button_column, sticky=W)
        
        # String variables
        path = StringVar(value=d.PATH)
        stud_pref = StringVar(value=d.STUDENT_PREF)
        faculty_pref = StringVar(value=d.FACULTY_PREF)
        interview_times = StringVar(value=d.TIMES_NAME)
        faculty_avail = StringVar(value=d.FACULTY_AVAILABILITY_NAME)
        stud_avail = StringVar(value=d.STUDENT_AVAILABILITY_NAME)
        
        # String boxes
        label_column = 0
        entry_column = 1
        Label(self.master,
                    text='File path:').grid(row=0, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=path).grid(row=0, column=entry_column, sticky=W)
        
        Label(self.master,
                    text='Student preference file name:').grid(row=1, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=stud_pref).grid(row=1, column=entry_column, sticky=W)
        
        Label(self.master,
                    text='Faculty preference file name:').grid(row=2, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=faculty_pref).grid(row=2, column=entry_column, sticky=W)
        
        Label(self.master,
                    text='Interview times file name:').grid(row=3, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=interview_times).grid(row=3, column=entry_column, sticky=W)
        
        Label(self.master,
                    text='Faculty availability file name:').grid(row=4, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=faculty_avail).grid(row=4, column=entry_column, sticky=W)
        
        Label(self.master,
                    text='Student availability file name:').grid(row=5, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=stud_avail).grid(row=5, column=entry_column, sticky=W)
        
        # Entry variables
        num_interviews = IntVar(value=d.NUM_INTERVIEWS)
        min_interviews = IntVar(value=d.MIN_INTERVIEWS)
        max_interviews = IntVar(value=d.MAX_INTERVIEWS)
        faculty_advantage = IntVar(value=d.FACULTY_ADVANTAGE)
        max_ranking = IntVar(value=d.MAX_RANKING)
        choice_exponent = IntVar(value=d.CHOICE_EXPONENT)
        lunch_penalty = IntVar(value=d.LUNCH_PENALTY)
        
        lunch_period = IntVar(value=d.LUNCH_PERIOD)
        recruiting_weight = IntVar(value=d.RECRUITING_WEIGHT)
        track_weight = IntVar(value=d.TRACK_WEIGHT)
        faculty_sim_weight = IntVar(value=d.FACULTY_SIMILARITY_WEIGHT)
        num_similar_faculty = IntVar(value=d.NUM_SIMILAR_FACULTY)
        num_pref_2_check = IntVar(value=d.NUM_PREFERENCES_2_CHECK)
        num_suggestions = IntVar(value=d.NUM_SUGGESTIONS)
        check_frequency = IntVar(value=d.CHECK_FREQUENCY)
        max_solver_time_seconds = IntVar(value=d.MAX_SOLVER_TIME_SECONDS)
        empty_penalty = IntVar(value=d.EMPTY_PENALTY)
        
        
        # Entry boxes
        Label(self.master,
                    text='Number of interview slots (int > 0):').grid(row=6, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=num_interviews).grid(row=6, column=entry_column, sticky=W)
        
        Label(self.master,
                    text='Minimum number of interviews (int > 0):').grid(row=7, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=min_interviews).grid(row=7, column=entry_column, sticky=W)
        
        Label(self.master,
                    text='Maximum number of interviews (int >= 0):').grid(row=8, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=max_interviews).grid(row=8, column=entry_column, sticky=W)
        
        Label(self.master,
                    text='Faculty preference (ints [0, 100]):').grid(row=9, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=faculty_advantage).grid(row=9, column=entry_column, sticky=W)
        
        Label(self.master,
                    text='Maximum number of matches to consider (int > 0):').grid(row=10, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=max_ranking).grid(row=10, column=entry_column, sticky=W)
        
        Label(self.master,
                    text='Preference given to first choice (int > 0):').grid(row=11, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=choice_exponent).grid(row=11, column=entry_column, sticky=W)
        
        Label(self.master,
                    text='Lunch penalty (int > 0):').grid(row=12, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=lunch_penalty).grid(row=12, column=entry_column, sticky=W)
        
        Label(self.master,
                    text='Interview period containing lunch (int > 0):').grid(row=13, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=lunch_period).grid(row=13, column=entry_column, sticky=W)
        
        Label(self.master,
                    text='Recruiting faculty advantage (int > 0):').grid(row=14, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=recruiting_weight).grid(row=14, column=entry_column, sticky=W)
         
        Label(self.master,
                    text='Maximum solver time in seconds (int > 0):').grid(row=15, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=max_solver_time_seconds).grid(row=15, column=entry_column, sticky=W)
        
        Label(self.master,
                    text='Penalty for empty interview slots \n(int > 0, AVOID USING):').grid(row=16, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=empty_penalty).grid(row=16, column=entry_column, sticky=W)
        
        Label(self.master,
                    text='Track advantage (int > 0):').grid(row=17, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=track_weight).grid(row=17, column=1, sticky=W)
        
        Label(self.master,
                    text='Faculty similarity advantage (int > 0):').grid(row=18, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=faculty_sim_weight).grid(row=18, column=1, sticky=W)
        
        Label(self.master,
                    text='Number of similar faculty to match (int > 0):').grid(row=19, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=num_similar_faculty).grid(row=19, column=1, sticky=W)
        
        Label(self.master,
                    text='Number of preferences to check \n during matching (int > 0):').grid(row=20, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=num_pref_2_check).grid(row=20, column=1, sticky=W)
        
        Label(self.master,
                    text='Number of interviews to suggest (int > 0):').grid(row=21, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=num_suggestions).grid(row=21, column=1, sticky=W)
        
        Label(self.master,
                    text='Number of iterations to run before checking\nprogress  (int > 0, big = slow):').grid(row=22, column=label_column, sticky=E)
        Entry(self.master,
                    textvariable=check_frequency).grid(row=22, column=1, sticky=W)
        
        # Run Button
        button_column = 4
        Button(self.master, text='Run', command=self.start_matchmaking).grid(row=24, column = button_column)
        
        # Feedback box
#        self.feedback =Label(self.master, text='test', width=20, height=10, bg='red').grid(row=0, column=4, sticky=W)
        
        # Mainloop
        self.master.mainloop()
        
    def start_matchmaking(self):
        fb = feedback()
        self.mm.main(fb)        
        
        
    
class feedback():
    
    def __init__(self):
        
        # Constants
        self.NUM_TEXT_LINES = 200
        self.text_var = StringVar(value='test')
        
        # Make master form
        self.master = Tk()
        tk_title(self.master, 'MatchMakr - Feedback') 
        
        self.label = Label(self.master, text='Test', width=100, height=50)
        self.label.pack()
        
        self.master.update()
        
    def update_text(self, text):
        self.label.config(text=text)    
        self.update()


if __name__=='__main__':
    g = gui()
