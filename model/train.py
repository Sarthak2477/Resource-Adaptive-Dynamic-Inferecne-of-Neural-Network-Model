import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from .config import FLAGS
from .ops import ResnetBatchNorm2d
from .resnet import Model, set_model_width, set_model_bit_width

def get_dataloaders():
    """Sets up CIFAR-10 data loaders. Downloads/extracts dataset if missing."""
    # Ensure data directory exists
    os.makedirs("./data", exist_ok=True)
    cifar_dir = "./cifar10"
    
    if not os.path.exists(cifar_dir):
        print("CIFAR-10 dataset not found locally. Downloading and extracting...")
        import urllib.request
        import tarfile
        
        url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
        tar_path = "./cifar10.tgz"
        
        print("Downloading cifar10.tgz...")
        urllib.request.urlretrieve(url, tar_path)
        
        print("Extracting cifar10.tgz...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=".")
        print("Dataset extracted successfully.")

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((32, 32)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = ImageFolder("./cifar10/train", transform=train_transform)
    test_dataset = ImageFolder("./cifar10/test", transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    min_w, max_w = FLAGS.width_mult_range
    min_bw, max_bw = min(FLAGS.bit_width_list), max(FLAGS.bit_width_list)
    kd_loss_fn = nn.KLDivLoss(reduction='batchmean')
    total_loss, n_steps = 0.0, 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # 1) max width, full precision -- real labels, this step's teacher
        set_model_width(model, max_w)
        set_model_bit_width(model, max_bw)
        out_max = model(images)
        loss_max = criterion(out_max, labels)
        loss_max.backward()
        soft_target = nn.functional.softmax(out_max.detach(), dim=1)

        # 2) sandwich rule
        sampled = [(min_w, min_bw), (max_w, min_bw)] + [
            (random.uniform(min_w, max_w), random.choice(FLAGS.bit_width_list))
            for _ in range(FLAGS.num_sample_widths)
        ]
        
        for wm, bw in sampled:
            set_model_width(model, wm)
            set_model_bit_width(model, bw)
            out = model(images)
            log_probs = nn.functional.log_softmax(out, dim=1)
            loss = kd_loss_fn(log_probs, soft_target)
            loss.backward()

        optimizer.step()
        total_loss += loss_max.item()
        n_steps += 1

    return total_loss / n_steps


@torch.no_grad()
def recalibrate_bn(model, loader, width_mult, bit_width, device, num_batches=100):
    core = model.module if hasattr(model, 'module') else model
    core.set_width(width_mult)
    core.set_bit_width(bit_width)

    key = (width_mult, bit_width)
    bn_layers = [m for m in core.modules() if isinstance(m, ResnetBatchNorm2d)]
    for m in bn_layers:
        m.calibrated_running_mean.pop(key, None)
        m.calibrated_running_var.pop(key, None)

    stats = {id(m): {'mean': None, 'var': None, 'n': 0} for m in bn_layers}

    def make_hook(m):
        def hook(module, inp, out):
            x = inp[0]
            dims = [0, 2, 3]
            batch_mean = x.mean(dim=dims)
            batch_var = x.var(dim=dims, unbiased=False)
            s = stats[id(m)]
            if s['mean'] is None:
                s['mean'], s['var'], s['n'] = batch_mean, batch_var, 1
            else:
                s['n'] += 1
                s['mean'] += (batch_mean - s['mean']) / s['n']
                s['var'] += (batch_var - s['var']) / s['n']
        return hook

    handles = [m.register_forward_hook(make_hook(m)) for m in bn_layers]

    try:
        core.train()
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            core(images.to(device))
    finally:
        for h in handles:
            h.remove()

    for m in bn_layers:
        s = stats[id(m)]
        if s['mean'] is not None:
            m.calibrated_running_mean[key] = s['mean']
            m.calibrated_running_var[key] = s['var']

    core.eval()


@torch.no_grad()
def evaluate(model, loader, criterion, device, configs=None):
    core = model.module if hasattr(model, 'module') else model
    configs = configs or FLAGS.deploy_configs
    core.eval()
    results = {}
    
    for wm, bw in configs:
        core.set_width(wm)
        core.set_bit_width(bw)
        correct, total, loss_sum = 0, 0, 0.0
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = core(images)
            loss_sum += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
        results[(wm, bw)] = {'loss': loss_sum / len(loader), 'acc': correct / total}
        
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_dataloaders()

    model = Model(num_classes=FLAGS.num_classes, input_size=FLAGS.image_size)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Initial settings
    set_model_width(model, max(FLAGS.width_mult_range))
    set_model_bit_width(model, max(FLAGS.bit_width_list))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum,
        weight_decay=FLAGS.weight_decay, nesterov=FLAGS.nesterov)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=FLAGS.multistep_lr_milestones, gamma=FLAGS.multistep_lr_gamma)

    checkpoint_path = "us_resnet_checkpoint.pt"
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    for epoch in range(start_epoch, FLAGS.num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )
        scheduler.step()
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, f"us_resnet_epoch{epoch + 1}_checkpoint.pt")

    # Recalibrate BN
    print("Recalibrating Batch Normalization layers...")
    for wm in FLAGS.bn_calibration_widths:
        for bw in FLAGS.bn_calibration_bits:
            recalibrate_bn(model, train_loader, wm, bw, device, num_batches=100)

    # Evaluate
    print("Evaluating model configs...")
    val_results = evaluate(model, test_loader, criterion, device)
    for (wm, bw), r in val_results.items():
        print(f"width={wm} bits={bw}: loss={r['loss']:.4f} acc={r['acc']:.4f}")


if __name__ == "__main__":
    main()
