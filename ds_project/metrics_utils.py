import numpy as np
from sklearn.metrics import f1_score


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

    if tracked_metrics is None:
        tracked_metrics = {"accuracy": accuracy, "f1-score": f1}

    assert len(labels) == len(scores), print(
        "Label and score lists are of different size"
    )

    scores_array = np.array(scores).astype(np.float32)
    labels_array = np.array(labels)

    metric_results = {}
    metric_results["metrics"] = {}
    for k, v in tracked_metrics.items():
        metric_value = v(scores_array, labels_array)
        metric_results["metrics"][k] = metric_value

    if print_log:
        print(
            " | ".join(
                ["{}: {:.4f}".format(k, v) for k, v in metric_results["metrics"].items()]
            )
        )

    return metric_results


def get_score_distributions(epoch_result_dict):
    """Return per-class score arrays."""
    scores = epoch_result_dict["scores"]
    labels = epoch_result_dict["labels"]

    for class_id in [0, 1]:
        epoch_result_dict["scores_" + str(class_id)] = np.array(scores)[
            np.array(labels) == class_id
        ]

    return epoch_result_dict
