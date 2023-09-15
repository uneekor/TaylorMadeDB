import argparse
import pandas as pd
from pytube import YouTube
import os
from tqdm import tqdm

# 명령행 인수 구문 분석
parser = argparse.ArgumentParser(description='유튜브 동영상 다운로드 스크립트')
parser.add_argument('--input_path', help='데이터프레임 파일 경로')
parser.add_argument('--output_path', help='동영상 저장 경로')
args = parser.parse_args()

# 데이터 프레임에서 URL 가져오기
def get_video_urls(dataframe):
    ids = dataframe['youtube_id'].tolist()
    urls = []
    for id in ids:
        url = f'https://www.youtube.com/watch?v={id}'
        urls.append(url)
    return urls


# 이미 다운로드된 동영상인지 확인
def is_video_downloaded(url, output_path):
    video_id = YouTube(url).video_id
    file_name = video_id + ".mp4"
    file_path = os.path.join(output_path, file_name)
    return os.path.exists(file_path)


# 유튜브 동영상 다운로드 함수
def download_video(url, output_path):
    try:
        yt = YouTube(url)
        video = yt.streams.get_highest_resolution()
        video.download(output_path, YouTube(url).video_id + '.mp4')
        # print(f"다운로드 완료: {url}")
    except Exception as e:
        print(f"다운로드 실패: {url}\n에러 메시지: {str(e)}")


if __name__ == '__main__':
    # 출력 디렉토리 생성
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        print(f"디렉토리 생성: {args.output_path}")

    # 데이터프레임 읽기
    # df = pd.read_pickle(args.input_path)
    df = pd.read_csv(args.input_path)
    # URL 목록 가져오기
    urls = get_video_urls(df)

    # 동영상 다운로드
    for url in tqdm(urls):
        if is_video_downloaded(url, args.output_path):
            # print(f"이미 다운로드된 동영상 입니다. 건너뛰기: {url}")
            continue
        download_video(url, args.output_path)
