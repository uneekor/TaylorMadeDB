
import argparse
import pandas as pd
from pytube import YouTube
import os
from tqdm import tqdm
import cv2

# 명령행 인수 구문 분석
parser = argparse.ArgumentParser(description='유튜브 동영상 다운로드 스크립트')
parser.add_argument('--input_path', help='데이터프레임 파일 경로')
parser.add_argument('--crop_ratio', help='')
parser.add_argument('--crop_direction', help='')
parser.add_argument('--output_path', help='동영상 저장 경로')
args = parser.parse_args()


if __name__ == '__main__':
    # 입력 동영상 파일 경로
    input_video_path = args.input_path


    # 출력 동영상 파일 경로
    output_video_path = args.output_path

    # 디렉토리 생성 (존재하지 않으면)
    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path)

    # 입력 동영상 파일 열기
    dir_list = os.listdir(args.input_path)
    dir_list = [item for item in dir_list if item.endswith(".mp4")]
    for filename in dir_list:
        input_video_path = os.path.join(args.input_path, filename)
        youtube_id = os.path.splitext(os.path.basename(input_video_path))[0]
        for view_type in ['dtl', 'fo']:
            cap = cv2.VideoCapture(input_video_path)

            # 동영상 정보 가져오기
            frame_width = int(cap.get(3))  # 동영상 프레임 너비
            frame_height = int(cap.get(4))  # 동영상 프레임 높이
            if view_type == 'fo':
                x1, y1 = 0, 0
                # x2, y2 = int(frame_width*float(args.crop_ratio)), frame_height
                x2, y2 = 720, frame_height
            elif view_type == 'dtl':
                # x1, y1 = int(frame_width*float(args.crop_ratio)), 0
                x1, y1 = 820, 0
                x2, y2 = frame_width, frame_height
            fps = int(cap.get(5))  # 동영상 프레임 속도
            total_frames = int(cap.get(7))  # 동영상 총 프레임 수
            print('total_frames', total_frames)
            # 영상의 뒷부분 10%를 저장하지 않도록 설정
            skip_percentage = 80
            skip_frames = int(total_frames * skip_percentage / 100)
            print('skip_frames', skip_frames)

            # 출력 동영상 파일 설정
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(os.path.join(output_video_path, f"{youtube_id}_{view_type}.mp4"), fourcc, fps, (x2 - x1, y2 - y1))

            # 동영상 잘라내기 및 저장
            frame_counter = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 동영상에서 특정 영역을 잘라냄
                cropped_frame = frame[y1:y2, x1:x2]

                # 글자를 이미지 우측 상단에 추가
                font = cv2.FONT_HERSHEY_SIMPLEX  # 글꼴
                font_scale = 1  # 글꼴 크기
                font_color = (128, 255, 255)  # 글자 색상 (BGR)
                font_thickness = 2  # 글자 두께
                text = str(frame_counter)
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                x = 10   # 우측으로부터 10 픽셀 여백
                y = text_size[1] + 10   # 상단으로부터 10 픽셀 여백
                cv2.putText(cropped_frame, text, (x, y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

                # 잘라낸 프레임을 출력 동영상에 추가
                # 처음부터 영상의 뒷부분 10%까지만 저장
                if frame_counter < skip_frames:
                    out.write(cropped_frame)

                frame_counter += 1

            # 파일 닫기
            cap.release()
            out.release()
            cv2.destroyAllWindows()