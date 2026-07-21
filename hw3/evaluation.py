import numpy as np
import matplotlib.pyplot as plt


def f1_score(y_true, y_pred):
    """ returns f1_score of binary classification task with true labels y_true and predicted labels y_pred"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    recall_ = tp / (tp + fn)
    precision_ = tp / (tp + fp)
    if recall_ == 0 and precision_ == 0:
        return 0
    return 2 * precision_ * recall_ / (precision_ + recall_)


def rmse (y_true, y_pred):
    """returns RMSE of regression task with true labels y_true and predicted labels y_pred"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def visualize_results(k_list, scores, metric, title, path):
    """ plot a results graph of cross validation scores """
    plt.figure()
    plt.plot(k_list, scores)
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel(metric)
    plt.savefig(path)
    plt.close('all')
    