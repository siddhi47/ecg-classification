from models import models
from data_preparation import ECGLoader
import argparse
import os
import torch
from loguru import logger


logger.add("train.log", rotation="1000 MB", )

def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a model for ECG classification."
    )
    parser.add_argument(
        "--data", type=str, default="data", help="Path to the data folder."
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="reference.csv",
        help="Path to the reference file.",
    )
    parser.add_argument("--model", type=str, default="resnet", help="Model to use.")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for."
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum.")
    parser.add_argument(
        "--weight-decay", type=float, default=0.0001, help="Weight decay."
    )
    parser.add_argument(
        "--save", type=str, default="models", help="Path to save the model to."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers to use."
    )
    parser.add_argument(
        "--pin-memory", type=bool, default=True, help="Whether to use pinned memory."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="How often to log the training progress.",
    )
    parser.add_argument(
        "--save-interval", type=int, default=1, help="How often to save the model."
    )
    parser.add_argument(
        "--val-interval", type=int, default=1, help="How often to validate the model."
    )
    parser.add_argument(
        "--debug", type=bool, default=False, help="Whether to run in debug mode."
    )
    return parser.parse_args()


def main():
    """Run the training."""
    args = parse_args()
    torch.manual_seed(args.seed)
    logger.info(f" Starting training for {args.model} model.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models[args.model](num_classes=4)
    model.to(device)
    latest_epoch = 0
    if any([args.model in file for file in os.listdir(args.save)]):
        # get latest model
        print("Saved model found.")
        latest_epoch = max(
            [
                int(file.split("_")[1].split(".")[0])
                for file in os.listdir(args.save)
                if args.model in file
            ]
        )
        model.load_state_dict(
            torch.load(os.path.join(args.save, f"model{args.model}_{latest_epoch}.pth"))
        )
        logger.info(f"Loaded model from epoch {latest_epoch}.")

    # load data
    train_loader = ECGLoader(
        os.path.join(args.data,'train'),
        batch_size=args.batch_size,
        )
    val_loader = ECGLoader(
        os.path.join(args.data,'val'),
        batch_size=args.batch_size,
        )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(latest_epoch, latest_epoch+args.epochs):
        model.train()
        accuracy=[]
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                acc = (output.argmax(dim=1) == target).float().mean()
                logger.info(
                        "\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t accuracy: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                        acc.item(),
                        end="",
                    )
                )
                accuracy.append(acc.item())
        logger.info(f"Training Accuracy = {sum(accuracy)/len(accuracy)}")
        if epoch % args.val_interval == 0:
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            val_loss /= len(val_loader)
            logger.info(
                "Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    val_loss,
                    correct,
                    len(val_loader),
                    100.0 * correct / len(val_loader),
                )
            )
        if epoch % args.save_interval == 0:
            os.makedirs(args.save, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.save, "model{}_{}.pth".format(args.model, epoch)),
            )
    
if __name__ == "__main__":
    main()
