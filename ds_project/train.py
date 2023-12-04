import torch
import fire
import os

from torch.nn import CrossEntropyLoss
from ds_project.data_loader import new_data_loader
from ds_project.fcnn import new_fcnn
from ds_project.utils import train_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DATA_PATH = r"data"  # PATH TO THE DATASET
EPOCH_NUM = 2


def train(size_h=96, size_w=96, embedding_size=128, num_classes=2):
    model = new_fcnn(size_h, size_w, embedding_size, num_classes)
    model = model.to(device)
    CKPT_NAME = 'model_base.ckpt'
    LOSS_FN = CrossEntropyLoss()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()

    train_batch_gen = new_data_loader(os.path.join(DATA_PATH, 'train_11k'), size_h, size_w,
                                    batch_size=256, shuffle=True, num_workers=2)
    val_batch_gen = new_data_loader(os.path.join(DATA_PATH, 'val'), size_h, size_w,
                                    batch_size=256, shuffle=False, num_workers=2)
    model, opt = train_model(model, train_batch_gen, val_batch_gen, opt, LOSS_FN, n_epochs=EPOCH_NUM, ckpt_name=CKPT_NAME)




if __name__ == '__main__':
    fire.Fire(train)
