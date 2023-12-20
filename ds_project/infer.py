import subprocess

import hydra
import numpy as np
import pandas as pd
import torch
from conf_hydra.hydra_conf import InferConfig
from hydra.core.config_store import ConfigStore

from ds_project.data_loader import new_data_loader


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
cs = ConfigStore.instance()
cs.store(name="infer_config", node=InferConfig)


@torch.no_grad()
def predict(model, batch_generator):
    model.train(False)

    result = np.empty(shape=(0,))
    files = []

    for X_batch, _, path_batch in batch_generator:
        logits = model(X_batch.to(device))
        predicted = torch.nn.functional.softmax(logits, dim=1)
        predicted = torch.argmax(predicted, dim=1).numpy()
        result = np.concatenate((result, predicted))
        files.extend(path_batch)

    return result, files


@hydra.main(version_base=None, config_path="../conf_hydra", config_name="config_infer")
def inference(cfg: InferConfig, data_infer=None):
    subprocess.run(["dvc", "pull"])
    model = torch.load(
        f"{cfg.infer.ckpt_path}/{cfg.infer.ckpt_name}.ckpt",
        map_location=torch.device(device),
    )
    if data_infer is None:
        data_infer = cfg.data.data_infer

    inf_batch_gen = new_data_loader(
        data_infer,
        cfg.model.size_h,
        cfg.model.size_w,
        inference=True,
        batch_size=cfg.infer.batch_size,
        shuffle=False,
        num_workers=2,
    )

    res, files = predict(model, inf_batch_gen)
    df = pd.DataFrame({"img path": files, "infer": res})
    result_path = f"{data_infer}_{cfg.infer.ckpt_name}.csv"
    df.to_csv(result_path, index=False)

    print(f"Inference saved in file: {result_path}")


if __name__ == "__main__":
    inference()
