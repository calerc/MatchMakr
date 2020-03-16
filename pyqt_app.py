import sys
from PyQt5.QtCore import Qt, pyqtSlot, QObject, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QDockWidget, QListWidget, QHBoxLayout, QVBoxLayout, QSpacerItem, QListWidgetItem
from PyQt5.QtWidgets import QApplication, QTextEdit, QAction, QPushButton, QFrame, QGridLayout, QSizePolicy, QLabel
from PyQt5.QtWidgets import QStackedWidget, QFileDialog, QSpinBox, QCheckBox, QLineEdit, QMessageBox
from itertools import product
from ipdb import set_trace
from os import getcwd
from os.path import join
import yaml
import shutil
from open_house_scheduling import match_maker
from contextlib import redirect_stdout
import io
import time
import threading

# class Communicate(QObject):
#     detect_change = pyqtSignal(str)

# def detect_change(callback_function, string_io, frame):
#     signal_src = Communicate()
#     signal_src.detect_change.connect(callback_function)
    
#     old_value = string_io.getvalue()
    
#     while frame:
#         string_io.flush()
#         new_value = string_io.getvalue()
#         if new_value == old_value:
#             is_same = str(1)
#         else:
#             is_same = str(0)
#             old_value = new_value
            
#         # time.sleep(0.0001)
        
#         signal_src.detect_change.emit(is_same)

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
        
        print('Double Clicked:' + self.text())
        
        if self.get_file:
            self.setText('Double Clicked')
      

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
        self.q_main_window.parent().save_settings()
        
    def load_settings(self):
        self.q_main_window.parent().load_settings()

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
    
    def __init__(self, q_main_window, f):
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
        
        # Logging
        # sys.stdout.write = self.output.insertPlainText
        # print('testing')
        # self.f = f
        # self.start_detect_change()
    
    def update_text_listener(self, is_same):
        if is_same == '0':
            self.update_text()
    
    # def start_detect_change(self):
    #     self.thread = threading.Thread(target=detect_change, args=(self.update_text_listener, self.f, self))
    #     self.thread.start()
        
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
        self.bt_clear = add_button(self, 'Clear Output', self.clear_output)
        self.bt_remove_results = add_button(self, 'Remove Results', self.remove_results)
    
    def update_text(self):
        pass
        # self.f.flush()
        # text = self.f.getvalue()
        # self.output.setText(text)
        # time.sleep(1)
        
    def validate(self):
        self.q_main_window.match_maker.validate()
        self.update_text()
        
    
    def run(self):
        self.q_main_window.match_maker.main()
    
    def clear_output(self):
        self.q_main_window.f.truncate(0)
        self.q_main_window.f.seek(0)
        self.output.setText('')
    
    def remove_results(self):
        
        working_dir = self.q_main_window.settings_frame.tb_path.text()
        results_dir = self.q_main_window.settings_frame.tb_results_dir.text()
        self.dir_to_remove = join(working_dir, results_dir)
        
        def callback(button_pressed):
            if button_pressed.text() == 'Ok':
                shutil.rmtree(self.dir_to_remove)
                print('Results Removed')
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
        # text = "Her extensive perceived may any sincerity extremity. Indeed add rather may pretty see. Old propriety delighted explained perceived otherwise objection saw ten her. Doubt merit sir the right these alone keeps. By sometimes intention smallness he northward. Consisted we otherwise arranging commanded discovery it explained. Does cold even song like two yet been. Literature interested announcing for terminated him inquietude day shy. Himself he fertile chicken perhaps waiting if highest no it. Continued promotion has consulted fat improving not way.  Folly was these three and songs arose whose. Of in vicinity contempt together in possible branched. Assured company hastily looking garrets in oh. Most have love my gone to this so. Discovered interested prosperous the our affronting insipidity day. Missed lovers way one vanity wishes nay but. Use shy seemed within twenty wished old few regret passed. Absolute one hastened mrs any sensible.   Or kind rest bred with am shed then. In raptures building an bringing be. Elderly is detract tedious assured private so to visited. Do travelling companions contrasted it. Mistress strongly remember up to. Ham him compass you proceed calling detract. Better of always missed we person mr. September smallness northward situation few her certainty something.   Sportsman delighted improving dashwoods gay instantly happiness six. Ham now amounted absolute not mistaken way pleasant whatever. At an these still no dried folly stood thing. Rapid it on hours hills it seven years. If polite he active county in spirit an. Mrs ham intention promotion engrossed assurance defective. Confined so graceful building opinions whatever trifling in. Insisted out differed ham man endeavor expenses. At on he total their he songs. Related compact effects is on settled do.   Do am he horrible distance marriage so although. Afraid assure square so happen mr an before. His many same been well can high that. Forfeited did law eagerness allowance improving assurance bed. Had saw put seven joy short first. Pronounce so enjoyment my resembled in forfeited sportsman. Which vexed did began son abode short may. Interested astonished he at cultivated or me. Nor brought one invited she produce her.   Barton did feebly change man she afford square add. Want eyes by neat so just must. Past draw tall up face show rent oh mr. Required is debating extended wondered as do. New get described applauded incommode shameless out extremity but. Resembled at perpetual no believing is otherwise sportsman. Is do he dispatched cultivated travelling astonished. Melancholy am considered possession on collecting everything.   May indulgence difficulty ham can put especially. Bringing remember for supplied her why was confined. Middleton principle did she procuring extensive believing add. Weather adapted prepare oh is calling. These wrong of he which there smile to my front. He fruit oh enjoy it of whose table. Cultivated occasional old her unpleasing unpleasant. At as do be against pasture covered viewing started. Enjoyed me settled mr respect no spirits civilly.   Dashwood contempt on mr unlocked resolved provided of of. Stanhill wondered it it welcomed oh. Hundred no prudent he however smiling at an offence. If earnestly extremity he he propriety something admitting convinced ye. Pleasant in to although as if differed horrible. Mirth his quick its set front enjoy hoped had there. Who connection imprudence middletons too but increasing celebrated principles joy. Herself too improve gay winding ask expense are compact. New all paid few hard pure she.   That know ask case sex ham dear her spot. Weddings followed the all marianne nor whatever settling. Perhaps six prudent several her had offence. Did had way law dinner square tastes. Recommend concealed yet her procuring see consulted depending. Adieus hunted end plenty are his she afraid. Resources agreement contained propriety applauded neglected use yet.  Up unpacked friendly ecstatic so possible humoured do. Ample end might folly quiet one set spoke her. We no am former valley assure. Four need spot ye said we find mile. Are commanded him convinced dashwoods did estimable forfeited. Shy celebrated met sentiments she reasonably but. Proposal its disposed eat advanced marriage sociable. Drawings led greatest add subjects endeavor gay remember. Principles one yet assistance you met impossible."
        # self.output.setText(text)
        self.output.setReadOnly(True)
        self.resize_text_output()
        
    def resize_text_output(self):
        frame_width = self.output_frame.width()
        frame_height = self.output_frame.height()
        self.output.resize(frame_width, frame_height)



class IOChecker(QTextEdit):
    has_changed= pyqtSignal('QString')
    
    def __init__(self, parent, f):
        super(IOChecker, self).__init__(parent)
        self.f = f
        
    def connect_and_emit_has_changed(self, string):
        self.has_changed.connect(self.detect_change)
        self.has_changed.emit()
        
    def detect_change(self, f):
        print("Slot Called")
        if f != self.f:
            print("Has Changed")
    

class MatchMakr(QMainWindow):
    
    def __init__(self, match_maker, f, parent=None):
        
        super(MatchMakr, self).__init__(parent)
		
        # Sizes
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        
        # Workflow
        self.items = QDockWidget("Workflow", self)
        self.listWidget = Dock(self)
        self.items.setWidget(self.listWidget)
        self.items.setFloating(False)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.items)
        
        # Standard Output Redirect
        self.f = f
        
        # Widget Stack
        self.stacked_widget = QStackedWidget(self) 
        
        # Output
        self.run_frame = RunFrame(self, self.f)
        # self.output = QTextEdit(self.stacked_widget)
        
        # Settings Frame               
        self.settings_frame = SettingsFrame(self.stacked_widget)
        self.advanced_settings_frame = AdvancedSettingsFrame(self.stacked_widget)
        self.stacked_widget.addWidget(self.settings_frame)
        self.stacked_widget.addWidget(self.advanced_settings_frame)
        self.stacked_widget.addWidget(self.run_frame)
        
        # Other GUI Setup
        self.setCentralWidget(self.stacked_widget)
        self.stacked_widget.setCurrentWidget(self.settings_frame)
        self.setWindowTitle("MatchMakr - Match Interviewers with Interviewees")
        
        # Match_maker
        self.match_maker = match_maker
        
        
    
    def settings_callback(self):
        self.stacked_widget.setCurrentWidget(self.settings_frame)
        # print("Settings")
    
    def advanced_settings_callback(self):
        self.stacked_widget.setCurrentWidget(self.advanced_settings_frame)
        # print("Advanced Settings")
        
    def run_callback(self):
        self.stacked_widget.setCurrentWidget(self.run_frame)
        self.run_frame.resize_text_output()
        # print('Run')
        
    def load_settings(self):
        settings = [self.settings_frame, self.advanced_settings_frame]
        
        filename = QFileDialog.getOpenFileName(self, 'Select File', getcwd(), 'YAML Files (*.yaml)')
        filename = filename[0]
        if filename == '':
            return
        # filename = join(getcwd(), 'settings.yaml')
        print('Saving to file: ' + filename)
        
        with open(filename, 'r') as f:
            settings_dictionary = yaml.safe_load(f)
        yaml_keys = settings_dictionary.keys()
        yaml_values = settings_dictionary.values()
            
        for s in settings:
            s_keys = s.__dict__.keys()
            # s_values = s.__dict__.values()
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
        
    def save_settings(self):
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
        
        filename = QFileDialog.getSaveFileName(self, 'Select File', getcwd(), 'YAML Files (*.yaml)')
        filename = filename[0]
        if filename == '':
            return
        print('Saving to file: ' + filename)
        
        with open(filename, 'w') as f:
            yaml.safe_dump(settings_dictionary, f)
                        
                
        
    def close_application(self):
        pass
    
    def resizeEvent(self, event):
        self.run_frame.resize_text_output()
        

class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        # self.widget.configure(state="normal")
        self.widget.append(str)
        # self.widget.configure(state="disabled")


if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    m_m = match_maker()
   
    f = io.StringIO()    
    # with redirect_stdout(f):
    mm = MatchMakr(m_m, f)
    m_m.printer = mm.run_frame.output.insertPlainText
    
    # sys.stdout.write = mm.run_frame.output.insertPlainText
    # sys.stderr.write = mm.run_frame.output.insertPlainText
    # sys.stdout = TextRedirector(mm.run_frame.output, "stdout")
    # def redirect(func):
    #     def inner(s):
    #         try:
    #             mm.run_frame.output.setText(s)
    #             return func
    #         except:
    #             return inner
    # sys.stdout.write=redirect(sys.stdout.write)
            
    # def redirect(s):
    #     mm.run_frame.output.setText(s)
    # sys.stdout.write = redirect
    print('test')
    
    mm.show()
    sys.exit(app.exec_())
