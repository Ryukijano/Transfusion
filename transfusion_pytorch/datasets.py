from torch.utils.data import Dataset
from torchvision import transforms as T
from datasets import load_dataset

class FlowersDataset(Dataset):
    def __init__(self, image_size):
        self.ds = load_dataset("nelorth/oxford-flowers")['train']

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.PILToTensor(),
            T.Lambda(lambda t: t / 255.)
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        pil = sample['image']
        tensor = self.transform(pil)
        return tensor