"""Train the keypoint model."""  
import argparse  
import json  
import numpy as np  
import torch  

import config as C  
from data import KPDataset, load_annotations, split_samples  
from model import KPNet  
from utils import decode_heatmaps, get_device, set_seed  


def validate(model, loader, dev):  # Defines a function to evaluate model performance
    model.eval()  # Switches the model to evaluation mode
    errs = [[] for _ in C.KP_NAMES]  # Creates an empty error list for each keypoint class
    with torch.no_grad():  # Disables gradient calculations during validation
        for x, _, vis, pts in loader:  # Iterates through validation batches containing images, visibility flags, and true coordinates
            p, _ = decode_heatmaps(model(x.to(dev)).cpu())  # Predicts heatmaps, moves them to CPU, and converts them into coordinates
            d = (p - pts).norm(dim=-1)  # Calculates Euclidean distance between predicted and actual keypoint coordinates
            for b in range(d.shape[0]):  # Iterates through each image in the current batch
                for k in range(len(C.KP_NAMES)):  # Iterates through all keypoint classes
                    if vis[b, k]:  # Checks whether the current keypoint is present
                        errs[k].append(d[b, k].item())  # Stores the pixel error for this keypoint
    per_kp = [float(np.mean(e)) if e else float("nan") for e in errs]  # Calculates the average pixel error for each keypoint
    return float(np.nanmean(per_kp)), per_kp  # Returns the overall mean error and individual keypoint errors


def main():  # Defines the main training workflow
    ap = argparse.ArgumentParser()  # Creates a parser for command-line options
    ap.add_argument("--json", default=str(C.JSON_PATH))  # Specifies the annotation JSON path
    ap.add_argument("--images", default=str(C.IMAGES_DIR))  # Specifies the directory containing orchard images
    ap.add_argument("--out", default=str(C.RUNS_DIR))  # Specifies the output directory for training results
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)  # Sets the number of training epochs
    ap.add_argument("--bs", type=int, default=C.BATCH_SIZE)  # Sets the number of images processed per batch
    ap.add_argument("--lr", type=float, default=C.LR)  # Sets the learning rate for optimization
    ap.add_argument("--val_frac", type=float, default=C.VAL_FRAC)  # Sets the fraction of data reserved for validation
    ap.add_argument("--seed", type=int, default=C.SEED)  # Sets the random seed for reproducibility
    ap.add_argument("--new_split", action="store_true", help="ignore saved splits.json")  # Adds an option to regenerate the dataset split
    args = ap.parse_args()  # Reads the command-line arguments provided by the user

    from pathlib import Path  # Imports Path for handling file and directory paths
    out = Path(args.out)  # Converts the output directory string into a Path object
    out.mkdir(parents=True, exist_ok=True)  # Creates the output directory if it does not exist
    set_seed(args.seed)  # Sets random seeds for reproducible experiments
    dev = get_device()  # Selects CUDA GPU if available, otherwise CPU

    samples = load_annotations(Path(args.json), Path(args.images))  # Loads usable images and their keypoint annotations
    split_file = out / "splits.json"  # Defines the path for storing the dataset split
    if args.new_split and split_file.exists():  # Checks whether a new split was requested and a saved split exists
        split_file.unlink()  # Deletes the existing split file to allow a new split
    tr, val = split_samples(samples, args.val_frac, args.seed, split_file)  # Divides the samples into training and validation sets
    print(f"train {len(tr)} | val {len(val)} | device {dev}")  # Displays the dataset sizes and selected computing device

    tl = torch.utils.data.DataLoader(KPDataset(tr, True), batch_size=args.bs, shuffle=True,  # Creates a shuffled training data loader with augmentation enabled
                                     num_workers=2, drop_last=len(tr) > args.bs)  # Uses two workers and drops an incomplete final batch when appropriate
    vl = torch.utils.data.DataLoader(KPDataset(val, False), batch_size=args.bs, num_workers=2)  # Creates a validation data loader without training augmentation

    model = KPNet().to(dev)  # Creates the keypoint detection model and moves it to the selected device
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)  # Initializes the AdamW optimizer with learning rate and weight decay
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,  # Creates a learning-rate scheduler that follows a one-cycle schedule
                                                total_steps=args.epochs * len(tl))  # Sets the total number of scheduler updates
    best, history = 1e9, []  # Initializes the best validation error and an empty training history list

    for ep in range(1, args.epochs + 1):  # Repeats the training process for each epoch
        model.train()  # Switches the model to training mode
        tot = 0.0  # Initializes the total training loss for the current epoch
        for x, hm, vis, _ in tl:  # Iterates through training batches containing images, target heatmaps, and visibility flags
            x, hm, vis = x.to(dev), hm.to(dev), vis.to(dev)  # Moves images, heatmaps, and visibility flags to the selected device
            loss = (((model(x) - hm) ** 2).mean((2, 3)) * vis).sum() / vis.sum().clamp(min=1)  # Calculates mean squared heatmap loss for visible keypoints
            opt.zero_grad()  # Clears gradients from the previous training step
            loss.backward()  # Calculates gradients using backpropagation
            opt.step()  # Updates model parameters using the calculated gradients
            sched.step()  # Updates the learning rate according to the scheduler
            tot += loss.item()  # Adds the current batch loss to the epoch's total loss

        mean_err, per_kp = validate(model, vl, dev)  # Measures validation error overall and for each keypoint
        history.append({"epoch": ep, "loss": tot / len(tl), "val_px_err": mean_err,  # Records the epoch, average training loss, and validation error
                        "per_kp": dict(zip(C.KP_NAMES, per_kp))})  # Stores validation errors associated with each keypoint name
        print(f"ep {ep:3d} loss {tot/len(tl):.5f} val_err {mean_err:6.2f}px | " +  # Prints the epoch number, average loss, and overall validation error
              " ".join(f"{n[:8]}={e:5.1f}" for n, e in zip(C.KP_NAMES, per_kp)))  # Adds each keypoint's abbreviated name and error to the output

        ckpt = {"model": model.state_dict(), "kp_names": C.KP_NAMES, "epoch": ep}  # Creates a checkpoint containing weights, keypoint names, and epoch
        torch.save(ckpt, out / "last.pt")  # Saves the checkpoint from the current epoch
        if mean_err < best:  # Checks whether the validation error is lower than the best recorded error
            best = mean_err  # Updates the best validation error
            torch.save(ckpt, out / "best.pt")  # Saves the model checkpoint with the best validation result
        json.dump(history, open(out / "history.json", "w"), indent=2)  # Saves the training history in a readable JSON file

    print(f"best val mean pixel error {best:.2f} -> {out/'best.pt'}")  # Prints the best validation error and the best checkpoint path


if __name__ == "__main__":  # Checks whether this script is being executed directly
    main()  # Starts the training workflow