from data import *
from cross_validation import *
from knn import *
from evaluation import *
import numpy as np

if __name__ == "__main__":
    """pass"""
    data_path = "london_sample_2500.csv"
    df = load_data(data_path)
    folds = get_folds()
    k_values = [3,5,11,25,51,75,101]

    print("Part1 - Classification")
    feature_names = ["t1", "t2", "wind_speed", "hum"]
    X_train = add_noise(df[feature_names]).values
    y_train = adjust_labels(df["season"])
    classification_means = []
    for k in k_values:
        scores = cross_validation_scores(ClassificationKNN(k), X_train, y_train, folds, f1_score)
        print(f"k={k}, mean score: {np.mean(scores):.4f}, std of scores: {np.std(scores, ddof=1):.4f}")
        classification_means.append(np.mean(scores))
    visualize_results(k_values, classification_means, "f1 score", "Classification", "Classification.pdf")

    print()

    print("Part2 - Regression")
    feature_names = ["t1", "t2", "wind_speed"]
    X_train = add_noise(df[feature_names]).values
    y_train = np.array(df["hum"])
    regression_means = []
    for k in k_values:
        scores = cross_validation_scores(RegressionKNN(k), X_train, y_train, folds, rmse)
        print(f"k={k}, mean score: {np.mean(scores):.4f}, std of scores: {np.std(scores, ddof=1):.4f}")
        regression_means.append(np.mean(scores))
    visualize_results(k_values, regression_means, "RMSE", "Regression", "Regression.pdf")
