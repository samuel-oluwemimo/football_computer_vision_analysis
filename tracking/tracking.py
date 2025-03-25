import supervision as sv
from ultralytics import YOLO
import cv2 as cv
import pickle
import os
from utils import center

class Tracking():
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames):
        detections = []
        batch_size = 20

        for i in range(0, len(frames), batch_size):
            detection = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections+=detection
        return detections
    
    def read_obj_in_frame(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks


        detections = self.detect_frames(frames)
        tracks = {'ball': [], 'player': [], 'referee': []}
        for idx, detection in enumerate(detections):
            class_names = detection.names
            class_inv = {v:k for k, v in class_names.items()}
            supervision_detection = sv.Detections.from_ultralytics(detection)

            for obj_idx, class_idx in enumerate(supervision_detection.class_id):
                if class_names[class_idx] == 'goalkeeper':
                    supervision_detection.class_id[obj_idx] = class_inv['player']
            
            detection_with_trackers = self.tracker.update_with_detections(supervision_detection)

            tracks['ball'].append({})
            tracks['player'].append({})
            tracks['referee'].append({})

            for frame_detection in detection_with_trackers:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_inv['player']:
                    tracks['player'][idx][track_id] = {'bbox':bbox}

                if class_id == class_inv['referee']:
                    tracks['referee'][idx][track_id] = {'bbox':bbox}
            
            for frame_detection in detection_with_trackers:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == class_inv['ball']:
                    tracks['ball'][idx][1] = {'bbox':bbox}

        if stub_path is not None: 
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)    
        return tracks
    
    def ellipse(self, frames, bbox):
        y2 = bbox[3]
        center = center(bbox)


# def read_frames(url):
#     cap = cv.VideoCapture(url)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#     return frames

# tracker = Tracking('../models/best.pt')
# frames = read_frames('../video_data/08fd33_4.mp4')
# tracker.read_obj_in_frame(frames)

