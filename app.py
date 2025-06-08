from threading import Thread
import cv2, time
import os
import sys
# single thread doubles performance of gpu-mode - needs to be set before torch import
if any(arg.startswith('--gpu-vendor') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
from v3_swapper import  get_face_swapper
from v3_analyser import get_face_single
from queue import Queue
import mediapipe as mp

def detect_face_area_ratio(image):
    h, w = image.shape[:2]
    
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        # MediaPipe 使用 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if not results.detections:
            return None, 0.0

        # 假設只取第一張臉
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        # 相對座標轉為絕對座標
        x, y = int(bbox.xmin * w), int(bbox.ymin * h)
        box_w, box_h = int(bbox.width * w), int(bbox.height * h)

        face_area = box_w * box_h
        image_area = w * h
        ratio = face_area / image_area

        return (x, y, box_w, box_h), ratio

def add_padding(image, target_ratio=0.3):
    bbox, ratio = detect_face_area_ratio(image)
    if bbox is None:
        print("找不到人臉")
        return image

    if ratio <= target_ratio:
        print(f"人臉比例 {ratio:.2f} 已小於閾值 {target_ratio}")
        return image

    print(f"人臉比例 {ratio:.2f} 過高，執行 padding")

    # 根據比例決定 padding 倍數
    scale = (ratio / target_ratio) ** 0.5
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)

    # 建立新的畫布並將原圖置中
    top = (new_h - h) // 2
    bottom = new_h - h - top
    left = (new_w - w) // 2
    right = new_w - w - left

    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return padded

def process_img(frame):
    is_get_face,face = get_face_single(frame)
    if is_get_face:
        result = get_face_swapper().get(frame, face, source_face, paste_back=True)
    else:
        result=face
    return result

class VideoStreamWidget(object):
    def __init__(self, show_fps=True):
        self.show_fps = show_fps
        # queue
        self.q=Queue()
        # others
        self.threads=[]
        self.capture = cv2.VideoCapture(0)
        self.init = True
        # Start the thread to read frames from the video stream
        i=0
        while i<8:
            thread = Thread(target=self.update)
            thread.daemon = True
            self.threads.append(thread)
            thread.start()
            i+=1

    def update(self):
        # Read the next frame from the stream in a different thread
        global RUN
        while RUN:
        #while True:
            if self.capture.isOpened():
                (status, frame) = self.capture.read()
                if status:
                    result = process_img(frame)
                    self.q.put(result)
            #time.sleep(.01)

    def show_frame(self):
        global RUN
        output = self.q.get()
        if self.init == True:
            self.start_time = time.time()
            self.frame_count = 0
        else:
            # 計算FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time
            
        # Display frames in main program
        
        if self.init == False and self.show_fps == True:
            cv2.putText(output, f"FPS: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frame', output)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            RUN=False
            #v5_play_video.play_video_RUN=False
            #os._exit(0)
        self.init = False
        #return(output)

source_face = ""
RUN=True
def app(source_face_path,show_fps):
    #read source img
    global source_face
    source_face = cv2.imread(source_face_path)
    #mediapipe
    source_face = add_padding(source_face, target_ratio=0.3)
    is_get_source_face,source_face = get_face_single(source_face)
    #is_get_source_face,source_face = get_face_single(source_face)
    if not is_get_source_face:
        print("source no face.")
        os._exit(0)
    #app class
    #v5_play_video.play_video_RUN=True
    global RUN
    video_stream_widget = VideoStreamWidget(show_fps=show_fps)
    while RUN:
        video_stream_widget.show_frame()
    #v5_play_video.play_video_RUN=False
    video_stream_widget.capture.release()
    video_stream_widget.process.terminate()
    cv2.destroyAllWindows()
    video_stream_widget.final()
    #os._exit(0)

f = open('./setting.txt', 'r')
s= f.readlines()
f.close()
# 讀取 source_face_path
source_face_path = s[0].split('=')[1].strip()
#print(f"source_face_path: {source_face_path}")
app(source_face_path,True)