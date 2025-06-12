from v3_swapper import  get_face_swapper
from v3_analyser import get_face_single
import cv2
src = cv2.imread('captured_frame.jpg')
tar = cv2.imread('./data/target.jpg')
is_get_source_face,tar = get_face_single(tar)
if not is_get_source_face:
    print("source no face.")
    exit(0)
def process_img(frame):
    is_get_face,face = get_face_single(frame)
    if is_get_face:
        result = get_face_swapper().get(frame, face, tar, paste_back=True)
    else:
        result=face
    return result
result = process_img(src)
cv2.imwrite('result.jpg', result)
