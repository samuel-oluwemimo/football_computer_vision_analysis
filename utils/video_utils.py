import cv2 as cv

def read_video(video_path):
    cap = cv.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frame, output_video_path):
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_video_path, fourcc, 25.0, (output_video_frame[0].shape[1], output_video_frame[0].shape[0]))
    for frame in output_video_frame:
        out.write(frame)
    out.release()