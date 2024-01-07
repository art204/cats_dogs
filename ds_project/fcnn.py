from torch import nn


def new_fcnn(size_h, size_w, dense_size, embedding_size, num_classes):
    model = nn.Sequential()

    model.add_module("flatten", nn.Flatten(start_dim=1))

    model.add_module("dense1", nn.Linear(3 * size_h * size_w, dense_size))
    model.add_module("dense1_relu", nn.ReLU())
    model.add_module("dropout1", nn.Dropout(0.1))
    model.add_module("dense2", nn.Linear(dense_size, embedding_size))
    model.add_module("dense2_relu", nn.ReLU())
    model.add_module("dropout2", nn.Dropout(0.1))

    model.add_module("dense3_logits", nn.Linear(embedding_size, num_classes))
    return model
