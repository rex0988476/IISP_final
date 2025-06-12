from PyQt5 import QtWidgets
import sys
import threading
import cv2
import v5_part3

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Face swap')
        self.width_=600
        self.height_=600
        self.resize(self.width_, self.height_)
        self.is_start=True
        self.link_file_name=""
        self.ori_file_name=""
        self.style()
        self.ui()
        

    def style(self):
        self.style_box = '''
            background:#1a1a1a;
            border:1px solid #000;
            font-size:20px;
            color:white;
            font-family:Verdana;
            border-color:#1a1a1a;
            margin:1px
        '''
        self.style_btn = '''
            QPushButton{
                background:#1f538d;
                border:1px solid #000;
                border-radius:10px;
                padding:5px;
            }
            QPushButton:pressed{
                background:#a5bad1;
            }
            QPushButton:disabled{
                background:#a5bad1;
                color:#999999;
            }
        '''

    def testDevice(self):
        i=0
        while True:
            cap = cv2.VideoCapture(i) 
            if cap is None or not cap.isOpened():
                i-=1
                break
            i+=1
        return i

    def ui(self):
        self.main_box = QtWidgets.QWidget(self)
        #self.main_box.setGeometry(0,0,600,800)
        self.main_box.setStyleSheet(self.style_box)

        self.main_layout = QtWidgets.QFormLayout(self.main_box)

        self.change_face_btn=QtWidgets.QPushButton(self)
        self.change_face_btn.setText('換臉圖片')
        self.change_face_btn.setStyleSheet(self.style_btn)
        self.change_face_btn.clicked.connect(self.choose_swap_picture)

        self.face_name_label=QtWidgets.QLabel(self)

        #self.camera_label=QtWidgets.QLabel(self)
        #self.camera_label.setText('鏡頭:')
        #self.camera_box=QtWidgets.QComboBox(self)
        #max_camera_num = self.testDevice()
        #i=0
        #while i<=max_camera_num:
        #    self.camera_box.addItem(f"視訊鏡頭{i}")
        #    i+=1

        #info = self.audio.get_host_api_info_by_index(0)
        #num_devices = info.get('deviceCount')
        #devices = [self.audio.get_device_info_by_host_api_device_index(0, i) for i in range(num_devices)]
        #for i, device in enumerate(devices):
        #    print(f"device {i}: {device['name']}")

        self.face_enhance_check_box=QtWidgets.QCheckBox(self)
        self.face_enhance_check_box.toggle()
        self.face_enhance_label=QtWidgets.QLabel(self)
        self.face_enhance_label.setText('啟用人臉強化:')

        self.start_stream_btn=QtWidgets.QPushButton(self)
        self.start_stream_btn.setText('開始直播')
        self.start_stream_btn.setStyleSheet(self.style_btn)
        self.start_stream_btn.clicked.connect(self.start_stream)

        self.end_stream_btn=QtWidgets.QPushButton(self)
        self.end_stream_btn.setText('結束直播')
        self.end_stream_btn.setStyleSheet(self.style_btn)
        self.end_stream_btn.clicked.connect(self.close_stream)

        self.main_layout.addRow(self.change_face_btn,self.face_name_label)
        #self.main_layout.addRow(self.camera_label,self.camera_box)
        self.main_layout.addRow(self.face_enhance_label,self.face_enhance_check_box)
        self.main_layout.addRow(self.start_stream_btn)
        self.main_layout.addRow(self.end_stream_btn)

    def ui_set_enable(self,b):
        self.change_face_btn.setEnabled(b)
        self.start_stream_btn.setEnabled(b)
        self.end_stream_btn.setEnabled(b)

    def start_stream(self):
        if self.is_start and self.face_name_label.text()!="":
            self.is_start=False
            self.t_app=threading.Thread(target=self.run_app)
            self.t_app.setDaemon(True)
            v5_part3.RUN=True
            self.t_app.start()
            self.change_face_btn.setEnabled(False)
            self.start_stream_btn.setEnabled(False)
    
    def run_app(self):
        self.end_stream_btn.setEnabled(True)
        #int(self.camera_box.currentText()[-1])
        print(self.link_file_name)
        print(self.face_enhance_check_box.isChecked())
        v5_part3.app(source_face_path=self.link_file_name,face_enhance=self.face_enhance_check_box.isChecked())
        self.is_start=True
        self.change_face_btn.setEnabled(True)
        self.start_stream_btn.setEnabled(True)
        self.end_stream_btn.setEnabled(False)

    def close_stream(self):
        v5_part3.RUN=False
        #v3_play_video.play_video_RUN=False
        #pyautogui.press('q')

    def choose_swap_picture(self):
        filePath , filterType = QtWidgets.QFileDialog.getOpenFileNames(filter="Images (*.png *.jpg)")  # 選擇檔案對話視窗
        if len(filePath)>0:
            self.link_file_name=filePath[0]
            self.ori_file_name=filePath[0].split('/')[-1]
            if len(self.ori_file_name)>12:
                file_name=self.ori_file_name[:12]+"..."
            else:
                file_name=self.ori_file_name
            self.face_name_label.setText(file_name)
     
    def resizeEvent(self,event):
        width, height = event.size().width(), event.size().height()
        self.main_box.setGeometry(0,0,width,height)
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    Form = MyWidget()
    Form.show()
    sys.exit(app.exec_())