import hydra
import torch
from conf_hydra.hydra_conf import TrainConfig
from hydra.core.config_store import ConfigStore
from torch.nn import CrossEntropyLoss

from ds_project.data_loader import new_data_loader
from ds_project.fcnn import new_fcnn
from ds_project.utils import train_model


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)


@hydra.main(version_base=None, config_path="../conf_hydra", config_name="config_train")
def train(cfg: TrainConfig) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = new_fcnn(
        cfg.model.size_h,
        cfg.model.size_w,
        cfg.model.dense_size,
        cfg.model.embedding_size,
        cfg.model.num_classes,
    )
    model = model.to(device)
    loss_fn = CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    opt.zero_grad()

    train_batch_gen = new_data_loader(
        cfg.data.data_train,
        cfg.model.size_h,
        cfg.model.size_w,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_batch_gen = new_data_loader(
        cfg.data.data_val,
        cfg.model.size_h,
        cfg.model.size_w,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=2,
    )
    ckpt = f"{cfg.train.ckpt_path}/{cfg.train.ckpt_name}.ckpt"
    model, opt = train_model(
        model,
        train_batch_gen,
        val_batch_gen,
        opt,
        loss_fn,
        n_epochs=cfg.train.epoch_num,
        ckpt_name=ckpt,
    )
    print(f"Model ckpt saved in {ckpt}")


if __name__ == "__main__":
    train()
