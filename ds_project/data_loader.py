import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


def new_data_loader(data_path, size_h, size_w,
                    batch_size=256, shuffle=False, num_workers=2,
                    image_mean=None, image_std=None):
    if image_std is None:
        image_std = [0.229, 0.224, 0.225]
    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406]

    transformer = transforms.Compose([
        transforms.Resize((size_h, size_w)),  # scaling images to fixed size
        transforms.ToTensor(),  # converting to tensors
        transforms.Normalize(image_mean, image_std)  # normalize image data per-channel
    ])
    dataset = torchvision.datasets.ImageFolder(data_path, transform=transformer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
