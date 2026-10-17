"""Sanity-check the annotations: draw ground-truth keypoints on every image.

    python src/verify_labels.py
Writes overlays to runs/label_check/ and prints dataset statistics.
"""

from pathlib import Path  
import cv2 
import numpy as np 
import config as C 
from data import load_annotations 
from utils import draw_keypoints 


def main():  #  main annotation verification workflow
    samples = load_annotations()  # Loads images and their ground-truth keypoint annotations
    out = C.RUNS_DIR / "label_check"  # Sets the output directory for annotated images
    out.mkdir(parents=True, exist_ok=True)  # Creates the output directory if it does not exist

    counts = np.zeros(len(C.KP_NAMES), int)  # Initializes one keypoint counter per class
    for s in samples:  # Processes every usable annotated image
        bgr = cv2.resize(cv2.imread(str(s["path"])), (C.IMG_SIZE, C.IMG_SIZE))  # resizes the image to the configured dimensions
        pts = {n: tuple(s["pts"][k]) for k, n in enumerate(C.KP_NAMES) if s["vis"][k]}  # Creates a dictionary of visible keypoint names and coordinates
        counts += s["vis"].astype(int)  # Adds the visible keypoints from this image to the counters
        cv2.imwrite(str(out / s["name"]), draw_keypoints(bgr, pts))  # Draws ground-truth keypoints and saves the annotated image

    print("\nlabels per keypoint:")  # Prints a heading for keypoint counts
    for n, c in zip(C.KP_NAMES, counts):  # Iterates through each keypoint name and its count
        print(f"  {n:24s} {c}/{len(samples)}")  # Prints the count of images containing each keypoint

    allpts = np.stack([s["pts"] for s in samples])  # Combines coordinates from all images into one array
    vis = np.stack([s["vis"] for s in samples]).astype(bool)  # Combines visibility flags and converts them to Boolean values
    print("\nmean position (x, y) per keypoint:")  # Prints a heading for mean keypoint coordinates
    for k, n in enumerate(C.KP_NAMES):  # Iterates through each keypoint class and its index
        m = allpts[vis[:, k], k].mean(0)  # Calculates the mean coordinates using only images where the keypoint is visible
        print(f"  {n:24s} ({m[0]:.1f}, {m[1]:.1f})")  # Prints the mean x and y coordinates to one decimal place

    print(f"\noverlays -> {out}")  # Prints the directory containing the annotated images


if __name__ == "__main__":  # Checks whether the script is being run directly
    main()  # Executes the annotation verification workflow