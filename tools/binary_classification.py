# Treat each frame a binary classification problem based on the background colour
import numpy as np
import cv2
from PIL import Image
import torch
from sklearn.metrics import accuracy_score

# def accuracy_score(y_true, y_pred):
#     '''Accuracy score for binary classification'''
#     return np.mean(y_true == y_pred)


def round_to_base(x, base=5):
    if isinstance(x, int):
        return base * round(x / base)
    if isinstance(x, torch.Tensor):
        return base * torch.round(x / base)


def calculate_metric(pred_imgs, true_imgs):
    '''Round background colour and compare'''
    # batch x imgs x H x W
    y_preds = []
    y_trues = []
    for i in range(true_imgs.shape[0]):
        avg1 = pred_imgs[i].mean()
        avg2 = true_imgs[i].mean()
        avg1 = torch.round(avg1)
        avg2 = torch.round(avg2)

        y_preds.append(avg1.cpu())
        y_trues.append(avg2.cpu())

    return accuracy_score(y_trues, y_preds)


if __name__ == "__main__":
    print(
        calculate_metric(
            "../.results/review_5-1_k-3_d-3_lr-1e-2_sl1-mean/test_1-clip_01836.gif",
            "../.results/review_5-1_k-3_d-3_lr-1e-2_sl1-mean/test_1-clip_01836.gif"
        ))
