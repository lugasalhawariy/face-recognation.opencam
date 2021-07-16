import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 720)  # lebar
cam.set(4, 480)  # tinggi

faceDetector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeDetector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

facedata = 'facedata'  # folder untuk menyimpan data yang mau direkam

# PRIMARY KEY
faceID = input("Masukan Face ID : ")
print("Tatap wajah anda ke dalam webcam! Sedang proses pengambilan gambar...")

# ambil data dari 1
getData = 1

# proses pengulangan
while True:
    retV, frame = cam.read()  # read camera lalu dimasukan ke variabel frame
    gantikeGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detectMultiScale(framenya, scale faktor, min neighbors)
    faces = faceDetector.detectMultiScale(gantikeGray, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        namaFile = "face."+str(faceID)+"."+str(getData)+".jpg"
        cv2.imwrite(facedata+'/'+namaFile, frame)

        # ambil data sebanyak ....
        getData += 1
        roiAbu = gantikeGray[y:y+h, x:x+w]
        roiWarna = frame[y:y+h, x:x+w]
        eyes = eyeDetector.detectMultiScale(roiAbu)

        for (xe, ye, we, he) in eyes:
            cv2.rectangle(roiWarna, (xe, ye), (xe+we, ye+he), (0, 0, 255), 1)

    cv2.imshow('Cameraku', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
    elif getData >= 30:
        break

print('Pengambilan data finish!')
cam.release()
cv2.destroyAllWindows()
