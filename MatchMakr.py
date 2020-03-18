import sys
from PyQt5.QtGui import QTextCursor, QIcon
from PyQt5.QtCore import Qt, pyqtSlot, QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import QMainWindow, QDockWidget, QApplication, QStackedWidget, QFileDialog, QStatusBar 
from os import getcwd
import yaml
from match_maker import match_maker
import threading
from queue import Queue
from time import sleep
from os import path
from makr_components import SettingsFrame, AdvancedSettingsFrame, RunFrame, Dock

global stop_threads
stop_threads = False

'''
    TODO:
        move classes out of MatchMakr
        Clean up
        Create Exectuable and Documentation
'''
           
class MatchMakr(QMainWindow):
    
    def __init__(self, match_maker, parent=None):
        
        super(MatchMakr, self).__init__(parent)
		
        # Sizes
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        
        # Icon
        scriptDir = path.dirname(path.realpath(__file__))
        self.setWindowIcon(QIcon(path.join(scriptDir, 'MatchMakr_Icon.png')))
        
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
        self.update_status_thread = threading.Thread(target=self.thread_update_progress_bar)
        self.update_status_thread.start()
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
        self.run_frame.output.moveCursor(QTextCursor.End)
        self.run_frame.output.insertPlainText(text)
        
        
    
    def thread_update_progress_bar(self):
        SLEEP_TIME = 0.1
        was_running = False
        
        global stop_threads
        
        while not stop_threads:
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
    
    def resizeEvent(self, event):
        self.run_frame.resize_text_output()
        
    def closeEvent(self, event):
        
        # Stop Matchmaking
        self.run_frame.interrupt()
        
        # Stop other threads
        global stop_threads
        stop_threads = True
        self.update_status_thread.join()
        
        # Exit
        event.accept()
 
'''
    The new Stream Object which replaces the default stream associated with sys.stdout
    This object just puts data in a queue!
'''


class WriteStream(object):
    def __init__(self,queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)
        
    def flush(self):
        pass


'''
    A QObject (to be run in a QThread) which sits waiting for data to come through a Queue.Queue().
    It blocks until data is available, and one it has got something from the queue, it sends
    it to the "MainThread" by emitting a Qt Signal 
'''

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
    Main Function
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
