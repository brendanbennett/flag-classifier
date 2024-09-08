import torch

from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image

from utils import get_device

class FlagDataset(Dataset):
    def __init__(self, image_dir: str | Path, device: str, flags_to_omit: None | list[str] = None) -> None:
        self.flags_omitted = flags_to_omit or []
        self.device = torch.device(device)
        self.transform = transforms.Compose([
            transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32)
        ])
        self._image_dir = Path(image_dir)
        self._image_paths = [p for p in self._image_dir.iterdir() if p.stem not in self.flags_omitted]
        self._country_names = [path.stem for path in self._image_paths]
        self._images = [self._load_image(path) for path in self._image_paths]
                        
    def _load_image(self, path: Path) -> Image:
        im = Image.open(path).convert("RGBA")
        im = self.transform(im)
        im = im.to(device=self.device)
        return im
    
    def image_size(self) -> tuple[int, int, int]:
        return self._images[0].shape
    
    def __len__(self) -> int:
        return len(self._images)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._images[idx], self._country_names[idx]