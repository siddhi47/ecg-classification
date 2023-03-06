from models import models
from data_preparation import ECGLoader
import os
import torch
from argparse import ArgumentParser
import numpy as np
from sklearn.metrics import confusion_matrix

def parse_args():
    args = ArgumentParser(description="Test the model for ECG classification.")
    args.add_argument("--data", type=str, default="dataset", help="Path to the data folder.")
    args.add_argument("--model", type=str, default="resnet", help="Model to use.")
    args.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    args.add_argument("--num-class", type=int, default=4, help="Number of classes.")
    args.add_argument("--save", type=str, default="models", help="Path to save the model to.")

    return args.parse_args()

def main(args):
    
    # Create the model
    model = models[args.model](num_classes=args.num_class)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.cuda()
    else:
        device = torch.device("cpu")

    # Create the dataset
    dataloader = ECGLoader(os.path.join(args.data, "val"), args.batch_size)
    if len(dataloader) == 0:
        print(f"Could not find any data in {args.data}")
        return

    # Load the latest model
    if any([args.model in file for file in os.listdir(args.save)]):
        latest_epoch = max(
            [
                int(file.split("_")[1].split(".")[0])
                for file in os.listdir(args.save)
                if args.model in file
            ]
        )
        model.load_state_dict(torch.load(os.path.join(args.save, f"model{args.model}_{latest_epoch}.pth")))
        print(f"Loaded model from epoch {latest_epoch}.")
    else:
        print(f"No saved model found for {args.model}.")
        return

    model.eval()
    correct = 0
    total = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            _, predicted = torch.max(y_pred.data, 1)

            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(predicted.cpu().numpy())
            total += y.size(0)
            correct += (predicted == y).sum().item()
            false_positive += (predicted == 1) & (y == 0)
            false_negative += (predicted == 0) & (y == 1)
            true_positive += (predicted == 1) & (y == 1)
            true_negative += (predicted == 0) & (y == 0)


    print(f"True positive: {true_positive}")
    print(f"True negative: {true_negative}")
    print(f"False positive: {false_positive}")
    print(f"False negative: {false_negative}")
    print(f"Precision: {true_positive / (true_positive + false_positive)}")
    print(f"Recall: {true_positive / (true_positive + false_negative)}")
    print(f"F1 score: {2 * true_positive / (2 * true_positive + false_positive + false_negative)}")
    print(f"Specificity: {true_negative / (true_negative + false_positive)}")
    print(f"Accuracy: {100 * correct / total}%")

    cm = confusion_matrix(y_true_list, y_pred_list)

    f1_list = []
    for idx,(i,j,k) in enumerate(zip(cm.sum(axis=1), cm.sum(axis=0), cm.diagonal())):
        f1 = 2 * k / (i + j)
        print(f"F1 score for class {idx}: {f1}")
        f1_list.append(f1)
    print(f"Mean F1 score: {np.mean(f1_list)}")
if __name__ == "__main__":
    main(parse_args())
