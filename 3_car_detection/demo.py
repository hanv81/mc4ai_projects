import cv2, pygame, pickle
from model import PretrainModel
from PIL import Image

def main():
    left, top, right, bottom = 170, 30, 660, 410   # ROI
    pygame.mixer.init()
    pygame.mixer.music.load("warning.mp3")
    pretrained = PretrainModel()
    with open('models/model.pickle', 'rb') as f:
        model = pickle.load(f)

    count = 0
    cap = cv2.VideoCapture("video.mp4")
    while True:
        ret, frame = cap.read()
        if not ret or cv2.waitKey(1) == ord('q'):
            break
        roi_image = frame[top:bottom, left:right]
        pil_frame = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
        feature = pretrained.get_feature(pil_frame)
        prediction = model.predict([feature])[0]
        if prediction == 1:
            frame = cv2.rectangle(frame, (left,top), (right,bottom), (0,0,255), 1)
            count += 1
            if count >= 20:
                pygame.mixer.music.play()
        else:
            frame = cv2.rectangle(frame, (left,top), (right,bottom), (0,255,0), 1)
            count = 0

        cv2.imshow('Camera', frame)

    cap.release()
    cv2.destroyAllWindows()

main()