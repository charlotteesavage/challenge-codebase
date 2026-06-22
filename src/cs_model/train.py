import os
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
    with open("config.yaml") as f:
        return yaml.safe_load(f)



def main():

    cfg = load_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    all_data = get_dataset(cfg["data_dir"])
    val_data, train_data = all_data[:2], all_data[2:]

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

    checkpoints_out = f"{out}/checkpoints/{dt.datetime.now().strftime('%Y-%m-%d')}"
    logs_out = f"{out}/logs/{dt.datetime.now().strftime('%Y-%m-%d')}"
    plots_out = f"{out}/plots/{dt.datetime.now().strftime('%Y-%m-%d')}"

    os.makedirs(f"{checkpoints_out}", exist_ok=True)
    os.makedirs(f"{logs_out}", exist_ok=True)
    os.makedirs(f"{plots_out}", exist_ok=True)

    best_val_loss = float("inf")

    train_loss_history = []
    val_loss_history = []

    print("Starting training...")

    for epoch in range(cfg["epochs"]):

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

        # last checkpoint (every 10 epochs)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{checkpoints_out}/last_model.pth")

        # log
        with open(
            f"{logs_out}/{dt.datetime.now().strftime('%H-%M-%S')}_train_log.txt", "a"
        ) as f:
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
            plt.savefig(f"{plots_out}/loss_curve.png")
            plt.close()


if __name__ == "__main__":
    main()