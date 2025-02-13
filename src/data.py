import torch
import torchaudio
import os
from torch.utils.data import Dataset

class LoFiDataset(Dataset):
    def __init__(self, data_folder: str, samplerate: int = 3000) -> None:
        self.file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]
        self.samplerate = samplerate
    

    def __len__(self) -> int:
        return len(self.file_paths)
    

    def __getitem__(self, idx: int) -> torch.tensor:
        item_path = self.file_paths[idx]
        waveform, samplerate = torchaudio.load(item_path)
        
        resampler = torchaudio.transforms.Resample(
            orig_freq=samplerate,
            new_freq=self.samplerate
        )

        downsampled_waveform = resampler(waveform)
        return downsampled_waveform.mean(axis=0)[:-1]

