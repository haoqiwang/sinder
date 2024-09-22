from glob import glob

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def make_transform(
    smaller_edge_size: int, patch_size, center_crop=False, max_edge_size=812
) -> transforms.Compose:
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    interpolation_mode = transforms.InterpolationMode.BICUBIC
    assert smaller_edge_size > 0

    if center_crop:
        return transforms.Compose(
            [
                transforms.Resize(
                    size=smaller_edge_size,
                    interpolation=interpolation_mode,
                    antialias=True,
                ),
                transforms.CenterCrop(smaller_edge_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
                transforms.Lambda(
                    lambda img: img[
                        :,
                        : min(
                            max_edge_size,
                            (img.shape[1] - img.shape[1] % patch_size),
                        ),
                        : min(
                            max_edge_size,
                            (img.shape[2] - img.shape[2] % patch_size),
                        ),
                    ]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(
                    size=smaller_edge_size,
                    interpolation=interpolation_mode,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
                transforms.Lambda(
                    lambda img: img[
                        :,
                        : min(
                            max_edge_size,
                            (img.shape[1] - img.shape[1] % patch_size),
                        ),
                        : min(
                            max_edge_size,
                            (img.shape[2] - img.shape[2] % patch_size),
                        ),
                    ]
                ),
            ]
        )


class VisualDataset(Dataset):
    def __init__(self, transform, imgs=None):
        self.transform = transform
        if imgs is None:
            self.files = [
                'resources/example.jpg',
                'resources/villa.png',
                'resources/000000037740.jpg',
                'resources/000000064359.jpg',
                'resources/000000066635.jpg',
                'resources/000000078420.jpg',
            ]
        else:
            self.files = imgs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = self.files[index]
        img = Image.open(img).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


class ImageNetDataset(Dataset):
    def __init__(self, transform, num_train_max=1000000):
        self.transform = transform
        self.files = glob('data/imagenet/train/*/*.JPEG')
        step = len(self.files) // num_train_max
        self.files = self.files[::step]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert('RGB')
        img = self.transform(img)
        return img


def load_data(args, model):
    transform = make_transform(
        args.resolution, model.patch_size, center_crop=True
    )
    dataset = ImageNetDataset(
        transform=transform, num_train_max=args.num_train_max
    )
    return dataset


def load_visual_data(args, model):
    transform = make_transform(
        args.visual_size, model.patch_size, max_edge_size=1792
    )
    dataset = VisualDataset(transform=transform, imgs=vars(args).get('imgs'))
    return dataset
