

# # 本文从数据集读取文件进行预处理   划分数据集






# # def # 读取数据  预处理  创建  数据集

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchvision.transforms as transforms
import av  # 用于读取视频
from PIL import Image
from torch.utils.data import DataLoader, random_split
class AudioVideoDataset(Dataset):
    def __init__(self, audio_dir, video_dir, transform=None, audio_transform=None, max_frames=None):
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.transform = transform
        self.audio_transform = audio_transform
        self.max_frames = max_frames

        # 获取所有.wav文件对应的样本名（不带扩展名）
        self.sample_ids = sorted([
            f.split(".")[0] for f in os.listdir(audio_dir) if f.endswith(".wav")
        ])

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        # 加载语音
        audio_path = os.path.join(self.audio_dir, f"{sample_id}.wav")
        waveform, sample_rate = torchaudio.load(audio_path)
        if self.audio_transform:
            waveform = self.audio_transform(waveform)

        # 加载视频帧
        video_path = os.path.join(self.video_dir, f"{sample_id}.mp4")
        frames = self._load_video_frames(video_path)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        frames_tensor = torch.stack(frames, dim=0)  # shape: (T, C, H, W)

        return waveform, frames_tensor, sample_id

    def _load_video_frames(self, video_path):
        container = av.open(video_path)
        frames = []

        for frame in container.decode(video=0):
            img = frame.to_image()
            frames.append(img)
            if self.max_frames and len(frames) >= self.max_frames:
                break

        return frames
import torch.nn.functional as F

def custom_collate_fn(batch):
    audios, videos, names = zip(*batch)  # 解包

    # 1. Pad 音频
    max_audio_len = max(audio.shape[1] for audio in audios)
    padded_audios = [
        F.pad(audio, (0, max_audio_len - audio.shape[1])) for audio in audios
    ]
    audio_batch = torch.stack(padded_audios)  # shape: [B, C, T]

    # 2. Pad 视频帧
    max_video_len = max(video.shape[0] for video in videos)
    video_batch = []
    for video in videos:
        pad_len = max_video_len - video.shape[0]
        if pad_len > 0:
            pad = video[-1:].repeat(pad_len, 1, 1, 1)  # 重复最后一帧
            video = torch.cat([video, pad], dim=0)
        video_batch.append(video)
    video_batch = torch.stack(video_batch)  # shape: [B, T, C, H, W]

    return audio_batch, video_batch, names

if __name__ == "__main__":

    audio_transform = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)

    video_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # dataset = AudioVideoDataset(
    #     audio_dir='CMLRdataset/audio',
    #     video_dir='CMLRdataset/video',
    #     transform=video_transform,
    #     audio_transform=audio_transform,
    #     max_frames=75  # 可选：限制视频帧数
    # )


    dataset = AudioVideoDataset(
        audio_dir='CMLRdataset/audio_normal',
        video_dir='CMLRdataset/video',
        transform=video_transform,
        audio_transform=audio_transform,
        max_frames=75  # 可选：限制视频帧数
    )


    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn
    )

    # 计算拆分长度   15% 作为测试集
    total_size = len(dataset)
    test_size = int(0.15 * total_size)
    train_size = total_size - test_size

    # 拆分   
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # 拆分随机种子
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size],
                                               generator=torch.Generator().manual_seed(42))


    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn
    )



    for audio, video, name in train_loader:
        print("Audio shape:", audio.shape)    # (B, 1, T)
        print("Video shape:", video.shape)    # (B, T, C, H, W)
        print("Sample names:", name)
        # break
