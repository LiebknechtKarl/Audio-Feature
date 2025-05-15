

import os
import cv2

def video_format(folder_path) :
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        # 检查文件是否为 .mp4 视频文件
        if file_name.endswith(".mp4"):
            # 构建视频文件的完整路径
            video_path = os.path.join(folder_path, file_name)
            
            # 加载视频
            cap = cv2.VideoCapture(video_path)
            
            # 检查是否成功加载视频
            if not cap.isOpened():
                print(f"无法加载视频: {video_path}")
                continue
            
            # 获取视频帧率
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 输出视频帧率
            print(f"视频文件: {file_name}, 帧率: {fps} FPS")
            
            # 释放视频资源
            cap.release()
############### ---------------------------------------------------------------


import os
import torchaudio
import torchaudio.transforms as T



def audio_normal(input_dir, output_dir) : 
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 目标采样率
    target_sr = 16000

    # 遍历所有 WAV 文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # 加载音频
            waveform, sample_rate = torchaudio.load(input_path)

            # 转为单声道（取平均或取第一个通道）
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # (1, T)

            # 重采样（如果需要）
            if sample_rate != target_sr:
                resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
                waveform = resampler(waveform)

            # 保存处理后的音频
            torchaudio.save(output_path, waveform, target_sr)

            print(f"Processed: {filename}")


if __name__ == "__main__":
    # 文件夹路径
    folder_path = "./CMLRdataset/video"  # 根据你的实际情况修改这个路径
    # folder_path = "/home/fu/Desktop/ubuntu_data/nlp/demo_55_audio_future/3_feature/CMLRdataset/video"
    video_format(folder_path)



    # 输入和输出目录
    input_audio_dir = './CMLRdataset/audio'
    output_audio_dir = './CMLRdataset/audio_normal'

    audio_normal(input_audio_dir,output_audio_dir)