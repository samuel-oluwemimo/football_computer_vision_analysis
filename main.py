from utils import read_video, save_video
import numpy as np
from tracking import Trackers
from assigning_team import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_estimator import CameraEstimator
from view_transformer import ViewTransformer
from speed_distance_estimator import SpeedDistanceEstimator


def main():
    video_frames = read_video('video_data/test.mp4') # reads our video

    tracker =  Trackers('models/best.pt')

    tracks = tracker.get_obj_tracks(video_frames, 
                                    read_from_stub=True,
                                    stub_path='stubs/track_stubs.pkl')
    
    tracker.add_position_to_tracks(tracks)
    
    # camera movement estimation
    camera_movement_estimator = CameraEstimator(video_frames[0])
    camera_movement_per_frame_estimator = camera_movement_estimator.get_camera_positions(video_frames,
                                                                                         read_from_stub=True,
                                                                                         stub_path='stubs/camera_movement_estimator_stubs.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame_estimator)

    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # interpolate ball position
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    speed_distance_estimator = SpeedDistanceEstimator()
    speed_distance_estimator.calc_speed_distance_per_track(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    # Assign Ball to Closest Player
    player_assigner = PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assigner(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)



    
    # tracks = tracker.read_obj_in_frame(video_frames, stub_path='stubs/track_stubs1.pk1', read_from_stub=True)
    output_video_frame = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw Camera movement
    output_video_frame = camera_movement_estimator.draw_camera_movement(output_video_frame,camera_movement_per_frame_estimator)
    
    speed_distance_estimator.draw_speed_distance(output_video_frame, tracks)

    save_video(output_video_frame,'output_video/output_video.avi') # saves our video

if __name__ == '__main__':
    main()