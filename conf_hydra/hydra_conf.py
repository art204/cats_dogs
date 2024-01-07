from dataclasses import dataclass


@dataclass
class Data:
    data_train: str
    data_val: str
    data_test: str
    data_infer: str


@dataclass
class TrainParams:
    epoch_num: int
    lr: float
    batch_size: int
    ckpt_path: str
    ckpt_name: str


@dataclass
class InferParams:
    ckpt_path: str
    ckpt_name: str
    batch_size: int


@dataclass
class ModelParams:
    size_h: int
    size_w: int
    dense_size: int
    embedding_size: int
    num_classes: int


@dataclass
class MlflowParams:
    tracking_uri: str


@dataclass
class TransformsConfig:
    image_mean: list
    image_std: list


@dataclass
class TrainConfig:
    data: Data
    train: TrainParams
    model: ModelParams
    mlflow: MlflowParams
    img_transforms: TransformsConfig


@dataclass
class InferConfig:
    data: Data
    infer: InferParams
    model: ModelParams
    mlflow: MlflowParams
    img_transforms: TransformsConfig
