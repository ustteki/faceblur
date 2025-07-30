import cv2
import mediapipe as mp
import numpy as np

class FaceMeshBlurrer:
    def __init__(self):
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.last_face_bbox = None

    def get_face_bbox(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        self.last_face_bbox = None
        if results.multi_face_landmarks:
            h, w, c = img.shape
            for faceLms in results.multi_face_landmarks:
                x_min = w
                y_min = h
                x_max = 0
                y_max = 0
                for lm in faceLms.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, cx)
                    y_min = min(y_min, cy)
                    x_max = max(x_max, cx)
                    y_max = max(y_max, cy)
                box_w = x_max - x_min
                box_h = y_max - y_min
                expand_w = int(box_w * 0.2)
                expand_h = int(box_h * 0.2)
                x_min = max(0, x_min - expand_w)
                y_min = max(0, y_min - expand_h)
                x_max = min(w, x_max + expand_w)
                y_max = min(h, y_max + expand_h)
                self.last_face_bbox = (x_min, y_min, x_max, y_max)
                break  # Only use the first face
        return self.last_face_bbox

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    blurrer = FaceMeshBlurrer()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        bbox = blurrer.get_face_bbox(frame)
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox # it just returns the bbox
            h, w, _ = frame.shape
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            if (x_max - x_min) > 10 and (y_max - y_min) > 10:
                face_roi = frame[y_min:y_max, x_min:x_max]
                blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                frame[y_min:y_max, x_min:x_max] = blurred_face
        else:# this is a falllback if no face is deeteected so u nobody dont gget urself exposed :3
            frame = cv2.GaussianBlur(frame, (99, 99), 30)
            h, w, _ = frame.shape
            text = 'No Face Detected'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.5
            thickness = 6
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0,0,255), thickness, cv2.LINE_AA)
        cv2.imshow('Face Blur (press Q to quit)', frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
