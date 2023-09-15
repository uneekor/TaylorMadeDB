import argparse
import os
import cv2
import json
from tqdm import tqdm
from models.rtmpose_aihub_v2 import RTMPose
import pandas as pd
import shutil
import numpy as np

def get_joint_angle(a, b, c):
    ab = a - b
    cb = c - b

    # 벡터 각도 계산 (라디안)
    dot_product = np.dot(ab, cb)
    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(cb)
    cosine_similarity = dot_product / ((norm_ab * norm_bc)+0.0001)
    angle_in_radians = np.arccos(np.clip(cosine_similarity, -1.0, 1.0))

    # 결과 출력 (라디안에서도 출력)
    # print("abc 관절 각도 (라디안):", angle_in_radians)

    # 라디안을 도(degree)로 변환하여 출력
    # angle_in_degrees = np.degrees(angle_in_radians)
    # print("abc 관절 각도 (도):", angle_in_degrees)
    return angle_in_radians

def get_joint_angles(keypoints):
    triplets = [[3, 5, 7],
                [10, 12, 14],
                [3, 10, 12],
                [4, 11, 13],
                [11, 13, 15],
                [4, 6, 8],
                [3, 4, 11],
                [1, 4, 11],
                [1, 3, 10],
                [4, 3, 10],
                [5, 7, 16],
                [6, 8, 16],
                ]
    # keypoints = np.split(keypoints, 17)
    angle_feature = []
    for triplet in triplets:
        angle = get_joint_angle(keypoints[triplet[0]], keypoints[triplet[1]], keypoints[triplet[2]])
        angle_feature.append(angle)
    return list(np.array(angle_feature))

def find_farthest_point(wrist, club_bbox):
    wrist = np.array(wrist)
    bbox_coordinates = list(np.array(club_bbox[0:4]).reshape(2, 2))
    bbox_coordinates.append(np.array([bbox_coordinates[0][0], bbox_coordinates[1][1]]))
    bbox_coordinates.append(np.array([bbox_coordinates[1][0], bbox_coordinates[0][1]]))
    bbox_coordinates = np.array(bbox_coordinates)

    # wrist에서의 거리 계산
    distances = np.linalg.norm(bbox_coordinates - wrist, axis=1)

    # distances 중에서 가장 먼 좌표의 인덱스를 찾습니다.
    farthest_point_index = np.argmax(distances)

    # 가장 먼 좌표 반환
    farthest_point = bbox_coordinates[farthest_point_index]
    return list(farthest_point)

def read_inference_write(video_full_path, hpe_model, meta):
    filename = os.path.splitext(os.path.basename(video_full_path))[0]
    youtube_id = os.path.splitext(os.path.basename(video_full_path))[0].split('_')[0]
    view_type = os.path.splitext(os.path.basename(video_full_path))[0].split('_')[1]
    cap = cv2.VideoCapture(video_full_path)

    shot_meta = {}
    poses = []

    shot = meta[meta['youtube_id'] == youtube_id]
    shot_meta['name'] = str(shot.name.values[0]).replace(" ", "_")
    shot_meta['sex'] = str(shot.sex.values[0])
    shot_meta['view'] = view_type

    shot_meta['club'] = str(shot.club.values[0])
    shot_meta['poses'] = poses
    shot_meta['events'] = [int(e) for e in list(shot.events.values[0].split(' '))]
    shot_meta['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_id in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        # 비디오에서 프레임 읽기
        _, frame = cap.read()
        try:
            pose_results, _, bbox_club = hpe_model.pose_estimate(frame)
            # print(pose_results)
            if len(bbox_club) > 0:
                # 클럽 박스 구석 4개중 손목가 가장 거리가 먼점을 리스트로 만든다
                club_head = find_farthest_point(pose_results.keypoints[0].tolist()[7], bbox_club[0])
            else:
                club_head = []
            pose_result = {
                'person_bbox': pose_results.bboxes[0].tolist(),
                'club_bbox': pose_results.bboxes[0].tolist(),
                'club_head': [str(item) for item in club_head],
                'keypoints': pose_results.keypoints[0].tolist(),
                'joint_angles': [str(item) for item in get_joint_angles(np.array(pose_results.keypoints[0].tolist()))]
            }
            poses.append(pose_result)

        except Exception as e:
            print(f'failed shot id: {youtube_id},{frame_id}', e)
            poses.append([])
            # return
    # print(shot_meta)
    with open(os.path.join(args.in_dir, f"{filename}.json"), "w") as json_file:
        json.dump(shot_meta, json_file)


def main(args):
    meta = pd.read_csv(os.path.join(args.in_dir, 'tmDB.csv'))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    cuda_device = 'cuda:' + os.environ['CUDA_VISIBLE_DEVICES']
    args.cuda_device = cuda_device
    rtmpose_model = RTMPose(args)

    # 입력된 동영상 파일 경로에서, 동영상 파일의 이름들을 탐색해 리스트로 만들기
    # 동영상 파일을 읽어서, 좌표 뽑고, json으로 저장하기
    # print(os.listdir(args.in_dir))
    dir_list = os.listdir(args.in_dir)
    dir_list = [item for item in dir_list if item.endswith(".mp4")]
    # dir_list = sorted(dir_list, key=lambda x: int(os.path.splitext(x)[0]))

    for file_name in tqdm(dir_list):
        full_path = os.path.join(args.in_dir, file_name)
        read_inference_write(full_path, rtmpose_model, meta)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract hpe")
    parser.add_argument("--in_dir", type=str, default="",
                        help="Path containing the video to analyze for pose estimation")
    parser.add_argument("--device", type=str, default="0", help="GPU to use")
    parser.add_argument("--det_config", type=str, default="0", help="GPU to use")
    parser.add_argument("--det_checkpoint", type=str, default="0", help="GPU to use")
    parser.add_argument("--pose_config", type=str, default="0", help="GPU to use")
    parser.add_argument("--pose_checkpoint", type=str, default="0", help="GPU to use")
    parser.add_argument("--golfdb", type=str, default="./golfDB.pkl", help="")
    parser.add_argument("--dtl_fo_pair", type=str, default="./golfDB.pkl", help="")
    parser.add_argument("--out_dir", type=str, default="0", help="Directory where video analysis results are stored")
    args = parser.parse_args()
    main(args)