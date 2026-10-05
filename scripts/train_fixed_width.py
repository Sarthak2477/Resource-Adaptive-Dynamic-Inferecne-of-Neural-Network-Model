"""Train a separate supervised FP32 ResNet checkpoint at one fixed width."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import Model, get_dataloaders, set_model_bit_width, set_model_width
from models.config import FLAGS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=float, required=True, choices=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=FLAGS.num_epochs)
    parser.add_argument("--batch-size", type=int, default=FLAGS.batch_size)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1:
        raise ValueError("epochs, batch size, and threads must be positive")

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite fixed-width checkpoint: {output_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    FLAGS.batch_size = args.batch_size

    train_loader, _ = get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(num_classes=FLAGS.num_classes, input_size=FLAGS.image_size).to(device)
    set_model_width(model, args.width)
    set_model_bit_width(model, 32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=FLAGS.lr,
        momentum=FLAGS.momentum,
        weight_decay=FLAGS.weight_decay,
        nesterov=FLAGS.nesterov,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=FLAGS.multistep_lr_milestones,
        gamma=FLAGS.multistep_lr_gamma,
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1
        scheduler.step()
        print(f"epoch={epoch + 1}/{args.epochs} loss={total_loss / max(1, batch_count):.5f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "training_mode": "separately_trained_fixed_width_fp32",
        "architecture": "models.Model",
        "width_mult": args.width,
        "bit_width": 32,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "optimizer": "SGD",
        "learning_rate": FLAGS.lr,
        "weight_decay": FLAGS.weight_decay,
    }
    with output_path.open("xb") as checkpoint_file:
        torch.save(checkpoint, checkpoint_file)
    print(f"Fixed-width FP32 checkpoint saved: {output_path}")


if __name__ == "__main__":
    main()