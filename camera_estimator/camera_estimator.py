import pickle
import cv2 as cv
import numpy as np
import sys
import os
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class CameraEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5
        self.lk_params = dict(
            winSize = (15, 15),
            maxLevel = 2,
            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
        )


        first_frame_grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        maskfeatures = np.zeros_like(first_frame_grayscale)
        maskfeatures[:, 0:20] = 1
        maskfeatures[:, 900:1050] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3, 
            minDistance = 3,
            blockSize = 7,
            mask = maskfeatures,
        )

    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_positions(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                camera_positions = pickle.load(f)
            return camera_positions

        camera_movement = [[0,0]] * len(frames)

        gray = cv.cvtColor(frames[0], cv.COLOR_BGR2GRAY)
        old_features = cv.goodFeaturesToTrack(gray, **self.features)

        for frame_nums in range(1, len(frames)):
            frame_gray = cv.cvtColor(frames[frame_nums], cv.COLOR_BGR2GRAY)
            new_features, _, _ = cv.calcOpticalFlowPyrLK(gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            for i, (old, new) in enumerate(zip(old_features, new_features)):
                old_feature_point = old.ravel()
                new_feature_point = new.ravel()


                distance = measure_distance(old_feature_point, new_feature_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y= measure_xy_distance(old_feature_point, new_feature_point)
            if max_distance > self.minimum_distance:
                camera_movement[frame_nums] = [camera_movement_x, camera_movement_y]
                old_features = cv.goodFeaturesToTrack(frame_gray, **self.features)  
            gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement,f)
        
        return camera_movement
        
    def draw_camera_movement(self,frames, camera_movement_per_frame):
        output_frames=[]

        for frame_num, frame in enumerate(frames):
            frame= frame.copy()

            overlay = frame.copy()
            cv.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha =0.6
            cv.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
            frame = cv.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

            output_frames.append(frame) 

        return output_frames