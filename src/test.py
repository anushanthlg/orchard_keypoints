"""Evaluate a checkpoint on the validation split (same split used in training).""" 
import argparse  
import json  
from pathlib import Path  

import cv2  
import numpy as np  
import torch  

import config as C  
from data import KPDataset, load_annotations, split_samples  
from model import KPNet  
from utils import decode_heatmaps, draw_keypoints, get_device  

THRESHOLDS = (2, 5, 10, 20)  # Defines PCK evaluation thresholds in pixels for 256x256 images.


def main():  
    ap = argparse.ArgumentParser()  # Creates a parser for command-line arguments.
    ap.add_argument("--weights", default=str(C.RUNS_DIR / "best.pt"))  # Specifies the checkpoint file to evaluate.
    ap.add_argument("--json", default=str(C.JSON_PATH))  # Specifies the JSON annotation file.
    ap.add_argument("--images", default=str(C.IMAGES_DIR))  # Specifies the directory containing the images.
    ap.add_argument("--split", choices=["val", "train"], default="val")  # Allows evaluation on validation or training images.
    ap.add_argument("--save_overlays", action="store_true",  # Adds an optional flag to save images with ground truth and predictions.
                    help="save GT (solid) vs prediction (hollow) overlays")  # Explains what the overlay option does.
    ap.add_argument("--out", default=str(C.RUNS_DIR / "eval"))  # Specifies the directory for evaluation results.
    args = ap.parse_args()  # Reads the command-line arguments supplied by the user.

    dev = get_device()  
    model = KPNet(pretrained=False).to(dev)  # Creates the model without downloading pretrained weights and moves it to the selected device.
    model.load_state_dict(torch.load(args.weights, map_location=dev)["model"])  # Loads the trained model parameters from the checkpoint.
    model.eval()  # Sets the model to evaluation mode.

    samples = load_annotations(Path(args.json), Path(args.images), verbose=False)  # Loads the image annotations without printing detailed loading messages.
    tr, val = split_samples(samples, split_file=Path(args.weights).parent / "splits.json")  # Recreates the saved training and validation split.
    subset = val if args.split == "val" else tr  # Selects the validation or training samples based on the argument.
    ds = KPDataset(subset, train=False)  # Creates a dataset without training-time data augmentation.

    out = Path(args.out)  # Converts the output directory string into a Path object.
    out.mkdir(parents=True, exist_ok=True)  # Creates the output directory, including missing parent directories.
    errs = [[] for _ in C.KP_NAMES]  # Creates one list to store prediction errors for each keypoint.
    rows = {}  # Creates a dictionary to store per-image prediction errors.

    with torch.no_grad():  # Disables gradient calculations because the model is only being evaluated.
        for i, s in enumerate(subset):  # Iterates through each selected image and its annotation.
            x, _, vis, pts = ds[i]  # Gets the image tensor, unused heatmaps, visibility flags, and ground-truth coordinates.
            p, _ = decode_heatmaps(model(x[None].to(dev)).cpu())  # Predicts heatmaps and decodes them into keypoint coordinates on the CPU.
            p = p[0]  # Removes the batch dimension to obtain the keypoints for this image.
            d = (p - pts).norm(dim=-1)  # Calculates the Euclidean pixel error for each keypoint.

            rows[s["name"]] = {n: float(d[k]) if vis[k] else None for k, n in enumerate(C.KP_NAMES)}  # Stores each visible keypoint's error, using None when it is not annotated.

            for k in range(len(C.KP_NAMES)):  # Iterates through all configured keypoint types.
                if vis[k]:  # Checks whether this keypoint has a ground-truth annotation.
                    errs[k].append(float(d[k]))  # Adds its prediction error to the corresponding list.

            if args.save_overlays:  # Checks whether the user requested visualization images.
                bgr = cv2.resize(cv2.imread(str(s["path"])), (C.IMG_SIZE, C.IMG_SIZE))  # Reads the original image and resizes it to the model's input dimensions.
                for k, n in enumerate(C.KP_NAMES):  # Iterates through the keypoint types and their names.
                    if vis[k]:  # Checks whether the ground-truth keypoint is visible.
                        cv2.circle(bgr, tuple(int(round(v)) for v in pts[k]), 4, (255, 255, 255), -1)  # Draws the ground-truth keypoint as a filled white circle.
                draw_keypoints(bgr, p.numpy().tolist())  # Draws the model's predicted keypoints on the image.
                cv2.imwrite(str(out / s["name"]), bgr)  # Saves the visualization using the original image filename.

    print(f"\nEvaluated {len(subset)} {args.split} images")  # Prints how many images were evaluated and which split was used.
    print(f"{'keypoint':24s} {'n':>4s} {'mean px':>8s} {'median':>8s} " +  # Prints the column headings for the evaluation table.
          " ".join(f"PCK@{t:<3d}" for t in THRESHOLDS))  # Adds a column heading for each PCK threshold.

    all_e = []  # Creates a list to combine errors from all keypoint types.
    for n, e in zip(C.KP_NAMES, errs):  # Iterates through each keypoint name and its collected errors.
        if not e:  # Checks whether this keypoint has any visible ground-truth annotations.
            continue  # Skips keypoints that have no errors to evaluate.
        e = np.array(e)  # Converts the list of errors into a NumPy array.
        all_e.append(e)  # Adds this keypoint's errors to the overall error collection.
        pck = " ".join(f"{(e <= t).mean()*100:6.1f}%" for t in THRESHOLDS)  # Calculates PCK percentages at each pixel threshold.
        print(f"{n:24s} {len(e):4d} {e.mean():8.2f} {np.median(e):8.2f} {pck}")  # Prints the count, mean error, median error, and PCK scores.

    all_e = np.concatenate(all_e)  # Combines all keypoint errors into a single array.
    print(f"{'ALL':24s} {len(all_e):4d} {all_e.mean():8.2f} {np.median(all_e):8.2f}")  # Prints the overall number of errors, mean error, and median error.
    json.dump(rows, open(out / "per_image_errors.json", "w"), indent=2)  # Saves the errors for each image in a formatted JSON file.
    print(f"per-image errors -> {out/'per_image_errors.json'}")  # Prints the location of the saved error report.


if __name__ == "__main__":  
    main()  