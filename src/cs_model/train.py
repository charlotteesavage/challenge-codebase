import os
import random
import torch
import yaml
import matplotlib.pyplot as plt
from monai.data import DataLoader, PersistentDataset
from tqdm import tqdm

from dataset import get_dataset
from transforms import get_transforms
from unet import build_model

import datetime as dt


torch.backends.cudnn.benchmark = True



def load_config():
    # with open("config.yaml") as f:
    with open("config_all_features.yaml") as f:
        return yaml.safe_load(f)


def select_device():
    if not torch.cuda.is_available():
        return "cpu"
    free_by_idx = [
        torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())
    ]
    return f"cuda:{max(range(len(free_by_idx)), key=free_by_idx.__getitem__)}"


def main():

    cfg = load_config()

    device = select_device()

    print("Using device:", device)

    all_data = get_dataset(cfg["data_dir"])
    random.Random(cfg["seed"]).shuffle(all_data)
    val_data, train_data = all_data[:3], all_data[3:]

    train_transforms = get_transforms(cfg["patch_size"], cfg["train_num_samples"])
    val_transforms = get_transforms(cfg["patch_size"], cfg["val_num_samples"])

    print("Loading train dataset...")
    train_dataset = PersistentDataset(
        data=train_data,
        transform=train_transforms,
        cache_dir=cfg["cache_dir"],
    )
    loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
    )

    print("Loading val dataset...")
    val_dataset = PersistentDataset(
        data=val_data,
        transform=val_transforms,
        cache_dir=cfg["cache_dir"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
    )
    
    model = build_model().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"]
    )

    scaler  = torch.amp.GradScaler("cuda")
    l1_loss = torch.nn.L1Loss()

    out = cfg["output_dir"]

    # not date-stamped so a resumed run (e.g. after an SGE walltime kill) can find its checkpoint
    checkpoints_out = f"{out}/checkpoints"
    os.makedirs(checkpoints_out, exist_ok=True)
    last_checkpoint_path = f"{checkpoints_out}/last_checkpoint.pth"

    start_epoch = 0
    best_val_loss = float("inf")

    train_loss_history = []
    val_loss_history = []

    run_date = dt.datetime.now().strftime("%Y-%m-%d")
    run_starttime = dt.datetime.now().strftime("%H-%M-%S")

    if os.path.exists(last_checkpoint_path):
        print(f"Resuming from {last_checkpoint_path}")
        checkpoint = torch.load(last_checkpoint_path, map_location=device)

        changed = {
            k: (checkpoint["cfg"].get(k), cfg[k])
            for k in cfg
            if checkpoint["cfg"].get(k) != cfg[k]
        }
        if changed:
            print(
                f"WARNING: resuming with a different config than the checkpoint was saved with: {changed}"
            )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        train_loss_history = checkpoint["train_loss_history"]
        val_loss_history = checkpoint["val_loss_history"]
        run_date = checkpoint["run_date"]
        run_starttime = checkpoint["run_starttime"]

    logs_out = f"{out}/logs/{run_date}/{run_starttime}"
    plots_out = f"{out}/plots/{run_date}"
    os.makedirs(logs_out, exist_ok=True)
    os.makedirs(plots_out, exist_ok=True)

    log_path = f"{logs_out}/log.txt"
    plot_path = f"{plots_out}/{run_starttime}_plot.png"

    print("Starting training...")

    for epoch in range(start_epoch, cfg["epochs"]):

        model.train()

        epoch_loss = torch.zeros(1, device=device)

        pbar = tqdm(loader)

        for batch in pbar:

            x    = batch["input"].to(device)
            y    = batch["ct"].to(device)
            mask = batch["prediction_mask"].bool().to(device)
            y[~mask] = 0  # don't bother trying to predict the bed
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):

                pred = model(x)

                loss = l1_loss(pred, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.detach()

            pbar.set_description(f"loss {loss.item():.4f}")

        avg_train_loss = epoch_loss.item() / len(loader)

        scheduler.step()

        # validation
        model.eval()
        val_loss = torch.zeros(1, device=device)
        with torch.no_grad():
            for batch in val_loader:
                x    = batch["input"].to(device)
                y    = batch["ct"].to(device)
                mask = batch["prediction_mask"].bool().to(device)
                y[~mask] = 0  # don't bother trying to predict the bed

                with torch.amp.autocast("cuda"):
                    pred = model(x)
                    loss = l1_loss(pred, y)
                val_loss += loss.detach()
        avg_val_loss = val_loss.item() / len(val_loader)

        print(f"Epoch {epoch}  train={avg_train_loss:.4f}  val={avg_val_loss:.4f}")

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # best checkpoint (by val)
        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss

            torch.save(model.state_dict(), f"{checkpoints_out}/best_model.pth")

        # last checkpoint (every epoch, full state for resuming after a kill)
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "train_loss_history": train_loss_history,
                "val_loss_history": val_loss_history,
                "cfg": cfg,
                "run_date": run_date,
                "run_starttime": run_starttime,
            },
            last_checkpoint_path,
        )

        # log
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_train_loss},{avg_val_loss}\n")

        # plot loss (every 10 epochs)
        if epoch % 10 == 0:
            plt.figure()
            plt.plot(train_loss_history, label="train")
            plt.plot(val_loss_history, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Train / Val Loss")
            plt.legend()
            plt.savefig(plot_path)
            plt.close()


if __name__ == "__main__":
    main()