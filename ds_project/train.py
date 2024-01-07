import os
import subprocess

import hydra
import mlflow
import torch
from conf_hydra.hydra_conf import TrainConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.nn import CrossEntropyLoss

from ds_project.data_loader import new_data_loader
from ds_project.fcnn import new_fcnn
from ds_project.train_test_utils import train_model


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)


@hydra.main(version_base=None, config_path="../conf_hydra", config_name="config_train")
def train(cfg: TrainConfig) -> None:
    subprocess.run(["dvc", "pull"])

    device = torch.device("cpu")
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
        cfg.img_transforms.image_mean,
        cfg.img_transforms.image_std,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_batch_gen = new_data_loader(
        cfg.data.data_val,
        cfg.model.size_h,
        cfg.model.size_w,
        cfg.img_transforms.image_mean,
        cfg.img_transforms.image_std,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=2,
    )
    if not os.path.exists(cfg.train.ckpt_path):
        os.mkdir(cfg.train.ckpt_path)
    ckpt = f"{cfg.train.ckpt_path}/{cfg.train.ckpt_name}.ckpt"

    mlflow.set_tracking_uri(uri=cfg.mlflow.tracking_uri)
    mlflow.set_experiment(f"{cfg.train.ckpt_name} №1")

    with mlflow.start_run(run_name="Run №1"):
        mlflow.log_params(
            {
                "model_name": cfg.train.ckpt_name,
                "model_params": OmegaConf.to_container(cfg.model, resolve=True),
                "epoch_num": cfg.train.epoch_num,
                "lr": cfg.train.lr,
                "data_train": cfg.data.data_train,
                "data_val": cfg.data.data_val,
                "git_commit_id": "write git commit here",
            }
        )

        model, opt = train_model(
            model,
            train_batch_gen,
            val_batch_gen,
            opt,
            loss_fn,
            n_epochs=cfg.train.epoch_num,
            ckpt_name=ckpt,
            log_to_mlflow=True,
        )

        mlflow.set_tag(
            "Training Info",
            f"{cfg.train.ckpt_name} model trained on data {cfg.data.data_train}",
        )

    print(f"Model ckpt saved in {ckpt}")


if __name__ == "__main__":
    train()
