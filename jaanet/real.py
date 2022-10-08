import cv2
from feat import Detector
from feat.utils import read_pictures

webcam = cv2.VideoCapture(0)
currentframe = 0

detector = Detector(au_model = "jaanet")

while True:
    success, frame = webcam.read()
    cv2.imshow("Output", frame)
    cv2.imwrite('frame' + str(currentframe) + '.jpg', frame)
    fileImm = "frame" + str(currentframe)
    immagine = fileImm + ".jpg"
    testo = fileImm + ".txt"



    imm = read_pictures([immagine])
    
    detected_faces = detector.detect_faces(imm)

    detected_landmarks = detector.detect_landmarks(imm, detected_faces)

    detected_aus = detector.detect_aus(frame, detected_landmarks).flatten()

    
    row = []
    for element in detected_aus:
        elemento = str(element.item())
        row.append(elemento)

    f = open(testo, "w+")


    for elemento in row:
        f.write(elemento + "\n")

    f.close()


    currentframe += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
