from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from ds_project.inference_dataset import InferenceDataset


def new_data_loader(
    data_path,
    size_h,
    size_w,
    inference=False,
    batch_size=256,
    shuffle=False,
    num_workers=2,
    image_mean=None,
    image_std=None,
):
    if image_std is None:
        image_std = [0.229, 0.224, 0.225]
    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406]

    transformer = transforms.Compose(
        [
            transforms.Resize((size_h, size_w)),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ]
    )
    if inference:
        dataset = InferenceDataset(data_path, transform=transformer)
    else:
        dataset = ImageFolder(data_path, transform=transformer)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
