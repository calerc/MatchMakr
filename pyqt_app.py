import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QDockWidget, QListWidget, QHBoxLayout, QVBoxLayout, QSpacerItem, QListWidgetItem
from PyQt5.QtWidgets import QApplication, QTextEdit, QAction, QPushButton, QFrame, QGridLayout, QSizePolicy, QLabel
from PyQt5.QtWidgets import QStackedWidget, QFileDialog, QSpinBox, QCheckBox, QLineEdit
from itertools import product
from ipdb import set_trace
from os import getcwd
from os.path import join


class Dock(QListWidget):
    
    def __init__(self, q_main_window):
        super(Dock, self).__init__()
        self.q_main_window = q_main_window
        self.add_items()
        self.itemClicked.connect(self.item_click)
        
    def add_items(self):
        self.settings = self.addItem("Settings")
        self.advanced_settings = self.addItem("Advanced Settings")
        self.validate = self.addItem("Validate")
        self.run = self.addItem("Run")
        
    def item_click(self, item):
        item_text = item.text()
        function_handle = self.function_dict(item_text)
        function_handle()
        
        
    def function_dict(self, text):
        
        callback_dict = {'Settings'  : self.q_main_window.settings_callback,
                         'Advanced Settings': self.q_main_window.advanced_settings_callback,
                         'Validate'  : self.q_main_window.validate_callaback,
                         'Run'       : self.q_main_window.run_callback}
    
        return callback_dict[text]
    
class SettingsFrame(QFrame):
    
    def __init__(self, q_main_window):
        super(SettingsFrame, self).__init__()
        
        # Split the frame
        self.split_frame = QVBoxLayout(self)
        self.settings_frame = QFrame()
        self.button_frame = QFrame()
        self.split_frame.addWidget(self.settings_frame)
        self.split_frame.addWidget(self.button_frame)
        self.settings_grid = QGridLayout(self.settings_frame)
        self.button_layout = QHBoxLayout(self.button_frame)
        
        # Store the parent
        self.q_main_window = q_main_window
        # self.define_labels()
        # self.define_text_boxes()
        # self.define_buttons()
        
        # Populate the settings
        self.define_settings()
        # self.populate_settings()
        
    def define_settings(self):
        self.define_labels()
        self.define_io()
        self.define_controls()        
    
        
    def define_labels(self):        
        self.labels = {(0, 0): "Student Preferences:",
                       (1, 0): "Faculty Preferences:",
                       (2, 0): "Student Availability:",
                       (3, 0): "Faculty Availability:",
                       (4, 0): "Results Path:",
                       (5, 0): "Student Schedules Path:",
                       (6, 0): "Faculty Schedules Path:",
                       (7, 0): "Number of Interviews:",
                       (8, 0): "Minimum Number of Interviews:",
                       (9, 0): "Maximum Number of Interviews:",
                       (10, 0): "Reccomend Extra Matches:",
                       (11, 0): "Faculty Advantage Factor:"}
        
        for pos, name in self.labels.items():
                x, y = pos
                ob = QLabel(self.settings_frame)
                ob.setText(name)
                ob.setAlignment(Qt.AlignRight)
                self.settings_grid.addWidget(ob, x, y)
                
                
    def define_io(self):
        
        self.io_counter = -1
        def add_text_box(self, feature_type, text):
            self.io_counter += 1
            widget = feature_type(self.settings_frame)
            widget.setText(text)
            self.settings_grid.addWidget(widget, self.io_counter, 1)
            return widget
        
        def add_widget(self, wid_type):
            self.io_counter += 1
            widget = wid_type(self.settings_frame)
            self.settings_grid.addWidget(widget, self.io_counter, 1)
            return widget
            
        
        current_dir = getcwd()
        self.tb_stud_pref = add_text_box(self, QLineEdit, join(current_dir, 'student_preferences.csv'))
        self.tb_fac_pref = add_text_box(self, QLineEdit, join(current_dir, 'faculty_preferences.csv'))
        self.tb_stud_avail = add_text_box(self, QLineEdit, join(current_dir, 'student_availability.csv'))
        self.tb_fac_avail = add_text_box(self, QLineEdit, join(current_dir, 'faculty_availability.csv'))
        self.tb_results_dir = add_text_box(self, QLineEdit, join(current_dir, 'results'))
        self.tb_stud_sched_dir = add_text_box(self, QLineEdit, join(current_dir, 'results', 'student_schedules'))
        self.tb_fac_sched_dir = add_text_box(self, QLineEdit, join(current_dir, 'results', 'faculty_schedules'))
        self.sb_num_inter = add_widget(self, QSpinBox)
        self.sb_min_num_inter = add_widget(self, QSpinBox)
        self.sb_max_num_inter = add_widget(self, QSpinBox)
        self.cb_extra_match = add_widget(self, QCheckBox)
        self.sb_fac_advantage = add_widget(self, QSpinBox)
        
        # self tb_stud_pref =
    
    
    def define_controls(self):
        
        def add_button(self, text):
            button = QPushButton(self.button_frame)
            button.setText(text)
            self.button_layout.addWidget(button)
            return button
        
        self.bt_save = add_button(self, 'Save Settings')
        self.bt_load = add_button(self, 'Load Settings')
        
        
        # self.bt_save = QPushButton(self.button_frame)
        # self.bt_save
        
        
        # self.buttons = {(0, 0): 'Save Settings',
        #                 (0, 1): 'Load Settings'}
        
        # for pos, name in self.buttons.items():
        #         x, y = pos
        #         ob = QPushButton(self.button_frame)
        #         ob.setText(name)
        #         self.button_layout.addWidget(ob)
    
        
    
    def define_text_boxes(self):
        current_dir = getcwd()
        
    
    # def define_buttons(self):
        
            
    # def populate_frame(self):
    #     widgets_to_populate = [self.labels, self.text_boxes, self.buttons]
    #     widgets_types = [QLabel, QTextEdit, QPushButton]
        
    #     for wid, wid_type in zip(widgets_to_populate, widgets_types):
            
                
                
                
class AdvancedSettingsFrame(SettingsFrame):
    
    def __init__(self, q_main_window):
        super(AdvancedSettingsFrame, self).__init__(q_main_window)
    
    def define_labels(self):
        self.labels = {}
    
    def define_text_boxes(self):
        self.text_boxes = {}
    
    def define_buttons(self):
        self.buttons = {}
        

class dockdemo(QMainWindow):
    
    def __init__(self, parent = None):
        
        super(dockdemo, self).__init__(parent)
		
        # Sizes
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        
        # Menus
        self.define_file_menu()
        
        # Workflow
        self.items = QDockWidget("Workflow", self)
        self.listWidget = Dock(self)
        self.items.setWidget(self.listWidget)
        self.items.setFloating(False)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.items)
        
        # Widget Stack
        self.stacked_widget = QStackedWidget(self) 
        
        # Output
        self.output = QTextEdit(self.stacked_widget)
        
        # Settings Frame               
        self.settings_frame = SettingsFrame(self.stacked_widget)
        self.advanced_settings_frame = AdvancedSettingsFrame(self.stacked_widget)
        self.stacked_widget.addWidget(self.settings_frame)
        self.stacked_widget.addWidget(self.advanced_settings_frame)
        self.stacked_widget.addWidget(self.output)
        
        # Other GUI Setup
        self.setCentralWidget(self.stacked_widget)
        self.stacked_widget.setCurrentWidget(self.settings_frame)
        self.setWindowTitle("MatchMakr - Match Interviewers with Interviewees")
    
    def settings_callback(self):
        self.stacked_widget.setCurrentWidget(self.settings_frame)
        print("Settings")
    
    def advanced_settings_callback(self):
        self.stacked_widget.setCurrentWidget(self.advanced_settings_frame)
        print("Advanced Settings")
    
    def validate_callaback(self):
        self.stacked_widget.setCurrentWidget(self.output)
        print('Validate')
        
    def run_callback(self):
        self.stacked_widget.setCurrentWidget(self.output)
        print('Run')
        
    def define_file_menu(self):
        
        load_settings = QAction('&Load Settings', self)
        load_settings.triggered.connect(self.load_settings)
        load_settings.setShortcut("Ctrl+O")
        load_settings.setStatusTip("Load pre-defined settings")
        
        save_settings = QAction('&Save Settings', self)
        save_settings.triggered.connect(self.save_settings)
        save_settings.setShortcut("Ctrl+S")
        save_settings.setStatusTip("Save the current settings")
        
        close_action = QAction('&Close', self)
        close_action.triggered.connect(self.close_application)
        close_action.setStatusTip('Close the application')
        
        self.menu_bar = self.menuBar()
        self.file_menu = self.menu_bar.addMenu('&File')
        self.file_menu.addAction(load_settings)
        self.file_menu.addAction(save_settings)
        self.file_menu.addAction(close_action)
        
    def load_settings(self):
        print('Loading Settings')
        
    def save_settings(self):
        print('Saving Settings')
        
    def close_application(self):
        pass
	
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = dockdemo()
    ex.show()
    sys.exit(app.exec_())
