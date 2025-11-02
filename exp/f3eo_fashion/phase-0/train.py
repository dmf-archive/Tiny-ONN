import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from config import get_config
from data import get_dataloaders
from model import MiniViT

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), data.size(0))
        acc1 = accuracy(output, target, topk=(1,))[0]
        top1.update(acc1.item(), data.size(0))
        pbar.set_postfix({"Loss": f"{losses.avg:.4f}", "Acc": f"{top1.avg:.2f}%"})
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar("Train/Loss", losses.val, global_step)
        writer.add_scalar("Train/Accuracy", top1.val, global_step)
    return losses.avg, top1.avg

def validate(model, val_loader, criterion, device, epoch, writer):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Val {epoch+1}")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            losses.update(loss.item(), data.size(0))
            acc1 = accuracy(output, target, topk=(1,))[0]
            top1.update(acc1.item(), data.size(0))
            pbar.set_postfix({"Loss": f"{losses.avg:.4f}", "Acc": f"{top1.avg:.2f}%"})
    writer.add_scalar("Val/Loss", losses.avg, epoch)
    writer.add_scalar("Val/Accuracy", top1.avg, epoch)
    return losses.avg, top1.avg

def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_path: Path):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses, label='Val Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(epochs, train_accs, label='Train Acc')
    ax2.plot(epochs, val_accs, label='Val Acc')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path / "metrics.png")
    plt.close()

def _save_checkpoint(epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, best_val_acc: float, checkpoint_path: Path):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1}")

def _load_checkpoint(checkpoint_path: Path, model: nn.Module, optimizer: torch.optim.Optimizer):
    if not checkpoint_path.exists():
        print("No checkpoint found. Starting from scratch.")
        return 0, 0.0 # start_epoch, best_val_acc
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['best_val_acc']
    print(f"Checkpoint loaded from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
    return start_epoch, best_val_acc

def main(cfg_override=None):
    cfg = get_config()
    if cfg_override:
        cfg.model.model_type = cfg_override
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(cfg.data.data_dir, cfg.data.batch_size, cfg.data.num_workers)
    
    attention_type = "standard" if cfg.model.model_type == "standard" else "ffn_in_head"
    model = MiniViT(
        img_size=cfg.model.image_size,
        patch_size=cfg.model.patch_size,
        in_chans=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        drop_rate=cfg.model.dropout,
        attention_type=attention_type
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    
    log_dir = Path(cfg.train.log_dir) / cfg.model.model_type
    checkpoint_path = Path(cfg.train.checkpoint_dir) / cfg.model.model_type / "checkpoint.pth"
    writer = SummaryWriter(log_dir=log_dir)
    save_path = log_dir / "plots"
    save_path.mkdir(parents=True, exist_ok=True)

    start_epoch, best_val_acc = 0, 0.0
    if cfg.train.resume:
        start_epoch, best_val_acc = _load_checkpoint(checkpoint_path, model, optimizer)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(start_epoch, cfg.train.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, writer)
        
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), log_dir / "best_model.pth")
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        _save_checkpoint(epoch, model, optimizer, best_val_acc, checkpoint_path)

    writer.close()
    plot_metrics(train_losses, val_losses, train_accs, val_accs, save_path)
    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()