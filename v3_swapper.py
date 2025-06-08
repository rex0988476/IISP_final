
import os
from tqdm import tqdm
import cv2
import insightface
import threading
import v3_globals

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()


def get_face_swapper():
    global FACE_SWAPPER
    # 使用 with 創建一個執行緒鎖區塊，確保多個執行緒之間的安全訪問。
    with THREAD_LOCK:
        # 檢查 FACE_SWAPPER 是否為空。如果是空的，表示尚未創建人臉交換器對象，需要進行初始化。
        if FACE_SWAPPER is None:
            # 轉換 model 路徑，改為絕對路徑
            model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), './inswapper_128.onnx')
            # 創建人臉交換器模型
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=v3_globals.providers)
    # 返回創建的人臉交換器對象。這樣可以確保在需要使用人臉交換器時只創建一次，並且在多個執行緒之間共享相同的模型實例。
    return FACE_SWAPPER

