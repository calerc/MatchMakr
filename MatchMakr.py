import sys
from PyQt5.QtCore import Qt, pyqtSlot, QObject, pyqtSignal, QTextStream, QThread
from PyQt5.QtWidgets import QMainWindow, QDockWidget, QListWidget, QHBoxLayout, QVBoxLayout, QSpacerItem, QListWidgetItem
from PyQt5.QtWidgets import QApplication, QTextEdit, QAction, QPushButton, QFrame, QGridLayout, QSizePolicy, QLabel
from PyQt5.QtWidgets import QStackedWidget, QFileDialog, QSpinBox, QCheckBox, QLineEdit, QMessageBox, QWidget, QStatusBar
from ipdb import set_trace
from os import getcwd
from os.path import join
import yaml
import shutil
from match_maker import match_maker
import threading
from queue import Queue
from time import sleep

'''
    TODO:
        Harden match_maker
        Stop freezing on program close
        Stop Freezing of interrupt
        Add re_match_maker
        Create icon
        Clean up
        Create Exectuable and Documentation
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

class Dock(QListWidget):
    
    def __init__(self, q_main_window):
        super(Dock, self).__init__()
        self.q_main_window = q_main_window
        self.add_items()
        self.itemClicked.connect(self.item_click)
        
    def add_items(self):
        self.settings = self.addItem("Settings")
        self.advanced_settings = self.addItem("Advanced Settings")
        self.run = self.addItem("Run")
        
    def item_click(self, item):
        item_text = item.text()
        function_handle = self.function_dict(item_text)
        function_handle()
        
        
    def function_dict(self, text):
        
        callback_dict = {'Settings'  : self.q_main_window.settings_callback,
                         'Advanced Settings': self.q_main_window.advanced_settings_callback,
                         'Run'       : self.q_main_window.run_callback}
    
        return callback_dict[text]
    
class LineBox(QLineEdit):
    
    def __init__(self, parent):
        super(LineBox, self).__init__(parent)
        self.parent = parent
        self.get_file = False
    
    def mouseDoubleClickEvent(self, event):
                
        if self.get_file:
            dir_name = QFileDialog.getExistingDirectory(self, 'Select Directory', self.text())
            if dir_name != '':
                self.setText(dir_name)
      
class SettingsFrame(QFrame):
    
    def __init__(self, q_main_window):
        super(SettingsFrame, self).__init__()
        
        # Split the frame
        self.split_frame = QVBoxLayout(self)
        self.settings_frame = QFrame()
        self.button_frame = QFrame()
        self.split_frame.addWidget(self.settings_frame)
        self.split_frame.addStretch()
        self.split_frame.addWidget(self.button_frame)
        self.settings_grid = QGridLayout(self.settings_frame)
        self.button_layout = QHBoxLayout(self.button_frame)
        
        # Store the parent
        self.q_main_window = q_main_window
        
        # Populate the settings
        self.label_counter = -1
        self.io_counter = -1
        self.define_settings()
             
    def save_settings(self):
        self.q_main_window.parent().statusBar.showMessage('Saving Settings...')
        self.q_main_window.parent().save_settings()
        self.q_main_window.parent().statusBar.showMessage('Done')
        
    def load_settings(self):
        self.q_main_window.parent().statusBar.showMessage('Loading Settings...')
        self.q_main_window.parent().load_settings()
        self.q_main_window.parent().statusBar.showMessage('Done')

    def define_settings(self):
        self.define_labels()
        self.define_io()
        self.define_controls()        
    
    def add_label(self, text):
        self.label_counter += 1
        label = QLabel(self.settings_frame)
        label.setText(text)
        label.setAlignment(Qt.AlignRight)
        self.settings_grid.addWidget(label, self.label_counter, 0)
        return label
        
        
    def define_labels(self):        
        self.add_label("Path:")
        self.add_label("Student Preferences:")
        self.add_label("Faculty Preferences:")
        self.add_label("Student Availability:")
        self.add_label("Faculty Availability:")
        self.add_label("Results Folder:")
        self.add_label("Student Schedules Folder Name:")
        self.add_label("Faculty Schedules Folder Name:")        
        self.add_label("Number of Interviews:")
        self.add_label("Minimum Number of Interviews:")
        self.add_label("Maximum Number of Interviews:")
        self.add_label("Reccomend Extra Matches:")
        self.add_label("Faculty Advantage Factor:")
                
    def path_callback(self, parent):
        print(parent)
        
    def add_text_box(self, feature_type, text):
            self.io_counter += 1
            widget = LineBox(self.settings_frame)
            widget.setText(text)
            self.settings_grid.addWidget(widget, self.io_counter, 1)
            return widget
        
    def add_widget(self, wid_type):
            self.io_counter += 1
            widget = wid_type(self.settings_frame)
            self.settings_grid.addWidget(widget, self.io_counter, 1)
            return widget
        
    def set_min_max_def(self, spinbox, minimum, maximum, default):
        spinbox.setMinimum(minimum)
        spinbox.setMaximum(maximum)
        spinbox.setValue(default)
                
                
    def define_io(self):
           
        # Current Directory
        current_dir = getcwd()
        
        # Directory settings
        self.tb_path = self.add_text_box(QLineEdit, current_dir)
        self.tb_stud_pref = self.add_text_box(QLineEdit, 'student_preferences.csv')
        self.tb_fac_pref = self.add_text_box(QLineEdit, 'faculty_preferences.csv')
        self.tb_stud_avail = self.add_text_box(QLineEdit, 'student_availability.csv')
        self.tb_fac_avail = self.add_text_box(QLineEdit, 'faculty_availability.csv')
        self.tb_results_dir = self.add_text_box(QLineEdit, 'results')
        self.tb_stud_sched_dir = self.add_text_box(QLineEdit, 'student_schedules')
        self.tb_fac_sched_dir = self.add_text_box(QLineEdit, 'faculty_schedules')
        
        # Numbers and check boxes
        self.sb_num_inter = self.add_widget(QSpinBox)
        self.sb_min_num_inter = self.add_widget(QSpinBox)
        self.sb_max_num_inter = self.add_widget(QSpinBox)
        self.sb_num_extra_matches = self.add_widget(QSpinBox)
        self.sb_fac_advantage = self.add_widget(QSpinBox)
        
        # Configuration
        self.configure_io()
        
    def configure_io(self):
        
        # Open File Dialog
        self.tb_path.get_file = True
        
        # Spin Boxes
        self.set_min_max_def(self.sb_num_inter, 0, 99, 9)
        self.set_min_max_def(self.sb_min_num_inter, 0, 99, 3)
        self.set_min_max_def(self.sb_max_num_inter, 0, 99, 9)
        self.set_min_max_def(self.sb_num_extra_matches, 0, 99, 2)
        self.set_min_max_def(self.sb_fac_advantage, 0, 999999, 70)      
    
    
    def define_controls(self):
        
        def add_button(self, text, callback):
            button = QPushButton(self.button_frame)
            button.clicked.connect(callback)
            button.setText(text)
            self.button_layout.addWidget(button)
            return button
        
        # Buttons
        self.bt_load = add_button(self, 'Load Settings', self.load_settings)
        self.bt_save = add_button(self, 'Save Settings', self.save_settings)
          
class AdvancedSettingsFrame(SettingsFrame):
    
    def __init__(self, q_main_window):
        super(AdvancedSettingsFrame, self).__init__(q_main_window)
    
    def define_labels(self):
        
        # Text Boxes
        self.add_label("Log Name:")
        self.add_label("Matches File Name:")
        
        
        # Check Boxes
        self.add_label("Use Ranking:")       
        self.add_label("Use Student Availability::")
        self.add_label("Use Faculty Availabitily:")
        self.add_label("Print Student Preferences:")
        self.add_label("Print Faculty Preferences:")
        self.add_label("Use Interview Limits:")
        
        self.add_label("Choice Exponent:")
        self.add_label("Lunch Penalty:")
        self.add_label("Lunch Period:")
        self.add_label("Recruiting Weight:")
        self.add_label("Track Weight:")
        
        self.add_label("Faculty Similarity Weight:")
        self.add_label("Number of Similar Faculty:")
        self.add_label("Number of Preferences to Check:")
        self.add_label("Check Frequency:")
        self.add_label("Empty Penalty:")
        
        
    def define_io(self):
        
        # Directory settings
        self.tb_log_name = self.add_text_box(QLineEdit, 'log.txt')
        self.tb_match = self.add_text_box(QLineEdit, 'matches')
        
        # Check boxes
        self.cb_use_rank = self.add_widget(QCheckBox)
        self.cb_stud_avail = self.add_widget(QCheckBox)
        self.cb_fac_avail = self.add_widget(QCheckBox)
        self.cb_print_stud_pref = self.add_widget(QCheckBox)
        self.cb_print_fac_pref = self.add_widget(QCheckBox)
        self.cb_use_inter_limits = self.add_widget(QCheckBox)
        
        # Spin Boxes        
        self.sb_exp = self.add_widget(QSpinBox)
        self.sb_lunch_penalty = self.add_widget(QSpinBox)
        self.sb_lunch_period = self.add_widget(QSpinBox)
        self.sb_recruit_weight = self.add_widget(QSpinBox)        
        self.sb_track_weight = self.add_widget(QSpinBox)
        self.sb_fac_sim_weight = self.add_widget(QSpinBox)
        self.sb_num_sim_fac = self.add_widget(QSpinBox)
        self.sb_num_pref_2_check = self.add_widget(QSpinBox)
        self.sb_check_freq = self.add_widget(QSpinBox)
        self.sb_empty_penalty = self.add_widget(QSpinBox)
        
        # Configuration
        self.configure_io()
        
    def configure_io(self):
        
        # Check boxes
        self.cb_use_rank.setChecked(True)
        self.cb_stud_avail.setChecked(False)
        self.cb_fac_avail.setChecked(True)
        self.cb_print_stud_pref.setChecked(False)
        self.cb_print_fac_pref.setChecked(True)
        self.cb_use_inter_limits.setChecked(True)
        
        # Spin boxes (numbers)
        self.set_min_max_def(self.sb_exp, 0, 10, 4)
        self.set_min_max_def(self.sb_lunch_penalty, 0, 9999999, 50000)
        self.set_min_max_def(self.sb_lunch_period, 0, 99, 4)
        self.set_min_max_def(self.sb_recruit_weight, 0, 9999999, 30000)
        self.set_min_max_def(self.sb_track_weight, 0, 9999999, 30000)
        self.set_min_max_def(self.sb_fac_sim_weight, 0, 9999999, 30000)
        self.set_min_max_def(self.sb_num_sim_fac, 0, 99, 5)
        self.set_min_max_def(self.sb_num_pref_2_check, 0, 99, 5)
        self.set_min_max_def(self.sb_check_freq, 0, 10000, 100)
        self.set_min_max_def(self.sb_empty_penalty, 0, 9999999, 0)
       
class RunFrame(QFrame):
    
    def __init__(self, q_main_window):
        super(RunFrame, self).__init__()
        
        # Split the frame
        self.split_frame = QVBoxLayout(self)
        self.output_frame = QFrame()
        self.button_frame = QFrame()
        self.split_frame.addWidget(self.output_frame, 2)
        self.split_frame.addWidget(self.button_frame)
        self.output = QTextEdit(self.output_frame)
        self.button_layout = QHBoxLayout(self.button_frame)
        
        # Store the parent
        self.q_main_window = q_main_window
        
        # Define Layout
        self.define_settings()
        self.define_text_output()
        
        # Running
        # self.stop_running = False
        self.m2m = Makr2Maker(self.q_main_window)
    
    def update_text_listener(self, is_same, new_string):        
        if is_same == '0':
            self.new_text(new_string)
    
    def new_text(self, text):
        self.output.setText(text)
        QApplication.processEvents()
        
    def define_settings(self):
        self.define_controls()
        
    def define_controls(self):
        
        def add_button(self, text, callback):
            button = QPushButton(self.button_frame)
            button.clicked.connect(callback)
            button.setText(text)
            self.button_layout.addWidget(button)
            return button
        
        self.bt_validate = add_button(self, 'Validate', self.validate)
        self.bt_run = add_button(self, 'Run', self.run)
        self.bt_interrupt = add_button(self, 'Interrupt', self.interrupt)
        self.bt_clear = add_button(self, 'Clear Output', self.clear_output)
        self.bt_remove_results = add_button(self, 'Remove Results', self.remove_results)    
    
    def interrupt(self):
        self.q_main_window.is_interrupting = True
        self.q_main_window.statusBar.showMessage('Interrupting')
        self.q_main_window.match_maker.stopSearch()
        self.q_main_window.match_maker.is_running = False
        self.q_main_window.statusBar.showMessage('Matchmaking Interrupted')
        self.q_main_window.is_interrupting = False
        
    def validate(self):
        self.q_main_window.statusBar.showMessage('Validating...')
        self.m2m.apply_settings()
        t = threading.Thread(target=self.q_main_window.match_maker.validate)
        t.start()      
    
    def run(self):
        self.m2m.apply_settings()
        t = threading.Thread(target=self.q_main_window.match_maker.main)
        t.start()
        # self.q_main_window.is_running = False
    
    def clear_output(self):
        self.output.setText('')
    
    def remove_results(self):
        
        working_dir = self.q_main_window.settings_frame.tb_path.text()
        results_dir = self.q_main_window.settings_frame.tb_results_dir.text()
        self.dir_to_remove = join(working_dir, results_dir)
        
        def callback(button_pressed):
            if button_pressed.text() == '&OK':
                try:
                    shutil.rmtree(self.dir_to_remove)
                    print('Directory Removed: ' + self.dir_to_remove)
                except:
                    print('Directory not found: ' + self.dir_to_remove)
            else:
                return
        
        dialog = QMessageBox()
        dialog.setIcon(QMessageBox.Critical)
        dialog.setText('You are about to delete the results of the optimization.')
        dialog.setInformativeText(self.dir_to_remove)
        dialog.setWindowTitle('Delete Optimization Results?')
        dialog.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        dialog.buttonClicked.connect(callback)
        dialog.exec_()
        
    def define_text_output(self):
        self.output.setReadOnly(True)
        self.resize_text_output()
        
    def resize_text_output(self):
        frame_width = self.output_frame.width()
        frame_height = self.output_frame.height()
        self.output.resize(frame_width, frame_height)
  

class MatchMakr(QMainWindow):
    
    def __init__(self, match_maker, parent=None):
        
        super(MatchMakr, self).__init__(parent)
		
        # Sizes
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        
        # Match_maker
        self.match_maker = match_maker
        
        # Workflow
        self.items = QDockWidget("Workflow", self)
        
        self.listWidget = Dock(self)
        self.items.setWidget(self.listWidget)
        self.items.setFloating(False)
        self.items.setFeatures(QDockWidget.DockWidgetMovable)
        # self.items.DockWidgetClosable = False
        self.addDockWidget(Qt.LeftDockWidgetArea, self.items)
        
        # Widget Stack
        self.stacked_widget = QStackedWidget(self) 
        
        # Output
        self.run_frame = RunFrame(self)
        
        # Settings Frame               
        self.settings_frame = SettingsFrame(self.stacked_widget)
        self.advanced_settings_frame = AdvancedSettingsFrame(self.stacked_widget)
        self.stacked_widget.addWidget(self.settings_frame)
        self.stacked_widget.addWidget(self.advanced_settings_frame)
        self.stacked_widget.addWidget(self.run_frame)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('Waiting...')
        self.is_interrupting = False
        self.is_running = False
        
        # Other GUI Setup
        t = threading.Thread(target=self.thread_update_progress_bar)
        t.start()
        self.setCentralWidget(self.stacked_widget)
        self.stacked_widget.setCurrentWidget(self.settings_frame)
        self.setWindowTitle("MatchMakr - Match Interviewers with Interviewees")
        
                
    
    def settings_callback(self):
        self.stacked_widget.setCurrentWidget(self.settings_frame)
    
    def advanced_settings_callback(self):
        self.stacked_widget.setCurrentWidget(self.advanced_settings_frame)
        
    def run_callback(self):
        self.stacked_widget.setCurrentWidget(self.run_frame)
        self.run_frame.resize_text_output()
        
    def load_settings(self):
        settings = [self.settings_frame, self.advanced_settings_frame]
        
        filename = QFileDialog.getOpenFileName(self, 'Select File', getcwd(), 'YAML Files (*.yaml)')
        filename = filename[0]
        if filename == '':
            return
        print('Loading Settings from file: ' + filename)
        
        with open(filename, 'r') as f:
            settings_dictionary = yaml.safe_load(f)
        yaml_keys = settings_dictionary.keys()
        yaml_values = settings_dictionary.values()
            
        for s in settings:
            s_keys = s.__dict__.keys()
            for key, value in zip(yaml_keys, yaml_values):
                if key in s_keys:
                    prefix = key[0:3]
                    widget = getattr(s, key)
                    if prefix == 'tb_':                        
                        widget.setText(value)
                    elif prefix == 'cb_':
                        widget.setChecked(value)
                    elif prefix == 'sb_':
                        widget.setValue(value)
                        
    def get_settings_dict(self):
        settings = [self.settings_frame, self.advanced_settings_frame]        
        settings_dictionary = {}
        
        for s in settings:            
            keys = s.__dict__.keys()
            values = s.__dict__.values()        
            for key, value in zip(keys, values):
                prefix = key[0:3]
                if prefix == 'tb_':
                    settings_dictionary[key] = value.text()
                elif prefix == 'cb_':
                    # set_trace()
                    settings_dictionary[key] = value.isChecked()
                elif prefix == 'sb_':
                    settings_dictionary[key] = value.value()
                    
        return settings_dictionary
        
    def save_settings(self):      
        
        settings_dictionary = self.get_settings_dict()        
        
        filename = QFileDialog.getSaveFileName(self, 'Select File', getcwd(), 'YAML Files (*.yaml)')
        filename = filename[0]
        if filename == '':
            return
        print('Saving to file: ' + filename)
        
        with open(filename, 'w') as f:
            yaml.safe_dump(settings_dictionary, f)
            
    @pyqtSlot(str)        
    def append_text(self, text):
    #     t = threading.Thread(target=self.thread_append_text, args=(text,))
    #     t.start()
        
    # def thread_append_text(self, text):
        bottom = self.run_frame.output.verticalScrollBar().maximum()
        self.run_frame.output.verticalScrollBar().setValue(bottom)
        self.run_frame.output.insertPlainText(text)
        
        
    
    def thread_update_progress_bar(self):
        SLEEP_TIME = 0.1
        was_running = False
        
        while True:
            if self.match_maker.is_running and not self.is_interrupting:
                self.statusBar.showMessage('Running')
                sleep(SLEEP_TIME)
                self.statusBar.showMessage('Running.')
                sleep(SLEEP_TIME)
                self.statusBar.showMessage('Running..')
                sleep(SLEEP_TIME)
                self.statusBar.showMessage('Running...')
                sleep(SLEEP_TIME)
                
                was_running = True
                
            elif was_running:
                was_running = False
                self.statusBar.showMessage('Done')
        
    def close_application(self):
        pass
    
    def resizeEvent(self, event):
        self.run_frame.resize_text_output()
 
'''
    ----------------------------------------------------------------------------
'''

# The new Stream Object which replaces the default stream associated with sys.stdout
# This object just puts data in a queue!
class WriteStream(object):
    def __init__(self,queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)
        
    def flush(self):
        pass

# A QObject (to be run in a QThread) which sits waiting for data to come through a Queue.Queue().
# It blocks until data is available, and one it has got something from the queue, it sends
# it to the "MainThread" by emitting a Qt Signal 
class MyReceiver(QObject):
    mysignal = pyqtSignal(str)

    def __init__(self,queue,*args,**kwargs):
        QObject.__init__(self,*args,**kwargs)
        self.queue = queue

    @pyqtSlot()
    def run(self):
        while True:
            text = self.queue.get()
            self.mysignal.emit(text)

'''
    ----------------------------------------------------------------------------
'''


if __name__ == '__main__':
    
    # My app
    app = QApplication(sys.argv)
    m_m = match_maker()
    mm = MatchMakr(m_m)
    mm.show()    
    
    # Create Queue and redirect sys.stdout to this queue
    queue = Queue()
    sys.stdout = WriteStream(queue)
    sys.stderr = WriteStream(queue)
    
    # Create thread that will listen on the other end of the queue, and send the text to the textedit in our application
    thread = QThread()
    my_receiver = MyReceiver(queue)
    my_receiver.mysignal.connect(mm.append_text)
    my_receiver.moveToThread(thread)
    thread.started.connect(my_receiver.run)
    thread.start()
    
    sys.exit(app.exec_())
