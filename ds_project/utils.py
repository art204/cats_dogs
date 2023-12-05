import time

import numpy as np
import torch
from IPython.display import clear_output
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from tqdm.auto import tqdm


def compute_loss(model, data_batch, loss_function):
    """Compute the loss using loss_function for the batch of data and return mean loss value for this batch."""
    # load the data
    img_batch = data_batch["img"]
    label_batch = data_batch["label"]

    # forward pass
    logits = model(img_batch)

    # loss computation
    loss = loss_function(logits, label_batch)

    return loss, model


def accuracy(scores, labels, threshold=0.5):
    assert type(scores) is np.ndarray and type(labels) is np.ndarray
    predicted = np.array(scores > threshold).astype(np.int32)
    return np.mean(predicted == labels)


def f1(scores, labels, threshold=0.5):
    assert type(scores) is np.ndarray and type(labels) is np.ndarray
    predicted = np.array(scores > threshold).astype(np.int32)
    return f1_score(labels, predicted)


def calculate_metrics(scores, labels, tracked_metrics=None, print_log=False):
    """Compute all the metrics from tracked_metrics dict using scores and labels."""

    # you may add other metrics if you wish
    if tracked_metrics is None:
        tracked_metrics = {"accuracy": accuracy, "f1-score": f1}

    assert len(labels) == len(scores), print(
        "Label and score lists are of different size"
    )

    scores_array = np.array(scores).astype(np.float32)
    labels_array = np.array(labels)

    metric_results = {}
    for k, v in tracked_metrics.items():
        metric_value = v(scores_array, labels_array)
        metric_results[k] = metric_value

    if print_log:
        print(" | ".join(["{}: {:.4f}".format(k, v) for k, v in metric_results.items()]))

    return metric_results


def get_score_distributions(epoch_result_dict):
    """Return per-class score arrays."""
    scores = epoch_result_dict["scores"]
    labels = epoch_result_dict["labels"]

    # save per-class scores
    for class_id in [0, 1]:
        epoch_result_dict["scores_" + str(class_id)] = np.array(scores)[
            np.array(labels) == class_id
        ]

    return epoch_result_dict


@torch.no_grad()  # we do not need to save gradients on evaluation
def test_model(model, batch_generator, loss_function, subset_name="test", print_log=True):
    """Evaluate the model using data from batch_generator and metrics defined above."""

    # disable dropout / use averages for batch_norm
    model.train(False)

    # save scores, labels and loss values for performance logging
    score_list = []
    label_list = []
    loss_list = []

    device = next(model.parameters()).device

    for X_batch, y_batch in batch_generator:
        # do the forward pass
        logits = model(X_batch.to(device))
        scores = torch.softmax(logits, -1)[:, 1]
        labels = y_batch.numpy().tolist()

        # compute loss value
        loss = loss_function(logits, y_batch.to(device))

        # save the necessary data
        loss_list.append(loss.detach().cpu().numpy().tolist())
        score_list.extend(scores)
        label_list.extend(labels)

    if print_log:
        print("Results on {} set | ".format(subset_name), end="")

    metric_results = calculate_metrics(score_list, label_list, print_log=print_log)
    metric_results["scores"] = score_list
    metric_results["labels"] = label_list
    metric_results["loss"] = loss_list

    return metric_results


def train_model(
    model,
    train_batch_generator,
    val_batch_generator,
    opt,
    loss_fn,
    n_epochs,
    ckpt_name=None,
    visualize=False,
):
    """
    Run training: forward/backward pass using train_batch_generator and evaluation using val_batch_generator.
    Log performance using loss monitoring and score distribution plots for validation set.
    """

    train_loss, val_loss = [], [1]
    val_loss_idx = [0]
    best_model = None
    top_val_accuracy = 0

    for epoch in range(n_epochs):
        start_time = time.time()
        device = next(model.parameters()).device

        # Train phase
        model.train(True)  # enable dropout / batch_norm training behavior
        for X_batch, y_batch in tqdm(train_batch_generator, desc="Training", leave=False):
            # move data to target device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            data_batch = {"img": X_batch, "label": y_batch}

            # train on batch: compute loss, calc grads, perform optimizer step and zero the grads
            loss, model = compute_loss(model, data_batch, loss_fn)

            # compute backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            # log train loss
            train_loss.append(loss.detach().cpu().numpy())

        # Evaluation phase
        metric_results = test_model(
            model, val_batch_generator, loss_fn, subset_name="val"
        )
        metric_results = get_score_distributions(metric_results)

        if visualize:
            clear_output()

        # Logging
        val_loss_value = np.mean(metric_results["loss"])
        val_loss_idx.append(len(train_loss))
        val_loss.append(val_loss_value)

        if visualize:
            # tensorboard for the poor
            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            ax1.plot(train_loss, color="b", label="train")
            ax1.plot(val_loss_idx, val_loss, color="c", label="val")
            ax1.legend()
            ax1.set_title("Train/val loss.")

            ax2.hist(
                metric_results["scores_0"],
                bins=50,
                range=[0, 1.01],
                color="r",
                alpha=0.7,
                label="cats",
            )
            ax2.hist(
                metric_results["scores_1"],
                bins=50,
                range=[0, 1.01],
                color="g",
                alpha=0.7,
                label="dogs",
            )
            ax2.legend()
            ax2.set_title("Validation set score distribution.")

            plt.show()

        print(
            "Epoch {} of {} took {:.3f}s".format(
                epoch + 1, n_epochs, time.time() - start_time
            )
        )
        val_accuracy_value = metric_results["accuracy"]
        if val_accuracy_value > top_val_accuracy and ckpt_name is not None:
            top_val_accuracy = val_accuracy_value

            # save checkpoint of the best model to disk
            with open(ckpt_name, "wb") as f:
                torch.save(model, f)

    return model, opt
