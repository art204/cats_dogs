import fire
import numpy as np
import pandas as pd
import torch

from ds_project.data_loader import new_data_loader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def predict(model, batch_generator):
    model.train(False)

    result = np.empty(shape=(0,))
    files = []

    for X_batch, y_batch, path_batch in batch_generator:
        logits = model(X_batch.to(device))
        predicted = torch.nn.functional.softmax(logits, dim=1)
        predicted = torch.argmax(predicted, dim=1).numpy()
        result = np.concatenate((result, predicted))
        files.extend(path_batch)

    return result, files


def inference(
    model_path="model_base.ckpt", data_path="data/inference", size_h=96, size_w=96
):
    model = torch.load(model_path, map_location=torch.device(device))

    inf_batch_gen = new_data_loader(
        data_path,
        size_h,
        size_w,
        inference=True,
        batch_size=256,
        shuffle=False,
        num_workers=2,
    )

    res, files = predict(model, inf_batch_gen)
    df = pd.DataFrame({"img path": files, "infer": res})
    result_path = data_path + ".csv"
    df.to_csv(result_path, index=False)

    return result_path


if __name__ == "__main__":
    fire.Fire(inference())
