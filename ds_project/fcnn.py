from torch import nn

from ds_project.flatten import Flatten


def new_fcnn(size_h, size_w, embedding_size, num_classes):
    model = nn.Sequential()
    # reshape from "images" to flat vectors
    model.add_module("flatten", Flatten())
    # dense "head"
    model.add_module("dense1", nn.Linear(3 * size_h * size_w, 256))
    model.add_module("dense1_relu", nn.ReLU())
    model.add_module("dropout1", nn.Dropout(0.1))
    model.add_module("dense3", nn.Linear(256, embedding_size))
    model.add_module("dense3_relu", nn.ReLU())
    model.add_module("dropout3", nn.Dropout(0.1))
    # logits for NUM_CLASSES=2: cats and dogs
    model.add_module("dense4_logits", nn.Linear(embedding_size, num_classes))
    return model
