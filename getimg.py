import cv2
capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    if not ret:
        break
    cv2.imshow('Video Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('captured_frame.jpg', frame)
        print("Frame saved as 'captured_frame.jpg'")