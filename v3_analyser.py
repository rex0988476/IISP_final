import insightface
import v3_globals

FACE_ANALYSER = None


def get_face_analyser():
    global FACE_ANALYSER
    # 如果是空的，表示尚未創建人臉分析器對象，需要進行初始化。
    if FACE_ANALYSER is None:
        # 創建人臉分析器對象 FACE_ANALYSER
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=v3_globals.providers)
        # 對人臉分析器 FACE_ANALYSER 進行準備操作，使用 prepare 方法。該方法接收一些參數，包括 ctx_id 和 det_size。ctx_id 參數用於指定所使用的計算設備的 ID，這裡設置為 0 表示使用第一個計算設備。det_size 參數則用於指定人臉檢測的輸入大小，這裡設置為 (640, 640)。
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def get_face_single(img_data):
    # 該方法會分析圖像並檢測其中的人臉區域，並將結果存儲在變數 face 中。
    face = get_face_analyser().get(img_data)
        # 將 face 中的人臉區域進行排序，排序的依據是人臉區域的 bbox 屬性的第一個元素。bbox 是一個表示人臉邊界框的屬性，其中包含了人臉區域的左上角和右下角的座標等信息。通過對人臉區域按照左上角的 x 座標進行排序，可以將人臉按照從左到右的順序排列。
        # 最後，從排序後的人臉區域中返回第一個（最左邊的）人臉區域。這個人臉區域是根據左上角 x 座標最小的順序獲得的。
    try:
        return (True,sorted(face, key=lambda x: x.bbox[0])[0])
    except:
        return (False,img_data)
