from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from ds_project.inference_dataset import InferenceDataset


def new_data_loader(
    data_path,
    size_h,
    size_w,
    image_mean,
    image_std,
    inference=False,
    batch_size=256,
    shuffle=False,
    num_workers=2,
):
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
