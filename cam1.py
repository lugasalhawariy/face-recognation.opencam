import cv2
cam = cv2.VideoCapture(0)
cam.set(3, 640) #ubah lebar camera
cam.set(4, 480) #ubah tinggi camera
faceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    retV, frame = cam.read()
    changeGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(changeGray, 1.3, 5)

    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
    
    cv2.imshow("Webku", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()