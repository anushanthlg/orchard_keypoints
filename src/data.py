"""Annotation parsing, train/val split and the PyTorch Dataset."""  
import json  
import random  
import cv2  
import numpy as np 
import torch  
import config as C  
from utils import make_heatmaps  # function that generates keypoint heatmaps


def load_annotations(json_path=C.JSON_PATH, image_root=C.IMAGES_DIR, verbose=True):  # Loads annotations and locates image files
    """Return list of dicts: {name, path, pts[4,2], vis[4]} for the 4 target keypoints."""  
    data = json.load(open(json_path))  # Reads the annotation JSON file into Python objects
    index = {p.name: p for p in image_root.rglob("*")  # Creates a dictionary mapping image filenames to their paths
             if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}  # only PNG and JPEG images
    samples, not_found, no_kp = [], 0, 0  # Initializes the usable samples list and two counters
    for im in data["images"]:  
        pts = np.zeros((len(C.KP_NAMES), 2), np.float32)  # Creates four empty keypoint coordinate pairs
        vis = np.zeros(len(C.KP_NAMES), np.float32)  # Creates four visibility indicators initialized to zero
        for lab in im["labels"]: 
            if lab["class_name"] in C.KP_NAMES and lab["keypoints"]:  # Checks whether the label is a target class with keypoints
                k = C.KP_NAMES.index(lab["class_name"])  # Finds the index of the current keypoint class
                if vis[k] == 0:  # Checks whether this keypoint class has not already been recorded
                    kp = lab["keypoints"][0]  # Selects the first keypoint from the label
                    pts[k] = (kp["x"], kp["y"])  # Stores the keypoint's x and y coordinates
                    vis[k] = 1  # Marks the keypoint as present
        if vis.sum() == 0:  # Checks whether the image contains no target keypoints
            no_kp += 1  # Increments the counter for images without target keypoints
            continue  
        path = index.get(im["image_name"]) 
        if path is None:  # Checks whether the image file is missing
            not_found += 1  # Increments the missing-image counter
            continue  # Skips the image because its file cannot be found
        pts = np.clip(pts, 0, C.IMG_SIZE - 1)  # Restricts coordinates to the range 0–255 for 256-pixel images
        samples.append({"name": im["image_name"], "path": path, "pts": pts, "vis": vis})  # Stores the image record and its labels
    if verbose:  # Checks whether summary output is enabled
        print(f"{len(samples)} usable images | {no_kp} without target keypoints | "  # Prints the number of usable images and begins the summary
              f"{not_found} not found on disk")  # Completes the summary with the number of missing image files
    return samples  # Returns the list of usable annotated image records


def split_samples(samples, val_frac=C.VAL_FRAC, seed=C.SEED, split_file=None):  # Splits the dataset into training and validation sets
    """Deterministic split. If split_file exists it is reused so train/eval agree."""  # Describes the reproducible splitting behavior
    by_name = {s["name"]: s for s in samples}  # Maps each image filename to its record
    if split_file is not None and split_file.exists():  # Checks whether a saved dataset split exists
        sp = json.load(open(split_file))  # Loads the previously saved split
        return ([by_name[n] for n in sp["train"] if n in by_name],  # Retrieves records assigned to training
                [by_name[n] for n in sp["val"] if n in by_name])  # Retrieves records assigned to validation
    names = sorted(by_name)  # Collects and sorts the image filenames
    random.Random(seed).shuffle(names)  # Shuffles filenames reproducibly using the specified seed
    n_val = max(1, int(len(names) * val_frac))  # Calculates the validation set size with at least one image
    val_names, train_names = names[:n_val], names[n_val:]  # Separates filenames into validation and training groups
    if split_file is not None:  # Checks whether the split should be saved to disk
        split_file.parent.mkdir(parents=True, exist_ok=True)  # Creates the parent directory if necessary
        json.dump({"train": train_names, "val": val_names}, open(split_file, "w"), indent=2)  # Saves the split as formatted JSON
    return [by_name[n] for n in train_names], [by_name[n] for n in val_names]  # Returns the training and validation records


class KPDataset(torch.utils.data.Dataset):  # Defines a custom PyTorch dataset for keypoint detection
    def __init__(self, samples, train):  # Initializes the dataset with samples and a training flag
        self.samples, self.train = samples, train  # Stores the image records and training mode
        self.mean = np.array(C.MEAN, np.float32)  # Converts RGB means into a NumPy array
        self.std = np.array(C.STD, np.float32)  # Converts RGB standard deviations into a NumPy array

    def __len__(self):  # Defines how PyTorch obtains the dataset length
        return len(self.samples)  # Returns the number of image records

    def __getitem__(self, i):  # Loads and prepares the sample at index i
        s = self.samples[i]  # Retrieves the selected image record
        img = cv2.cvtColor(cv2.imread(str(s["path"])), cv2.COLOR_BGR2RGB)  # Reads the image and converts BGR channels to RGB
        img = cv2.resize(img, (C.IMG_SIZE, C.IMG_SIZE))  # Resizes the image to 256 × 256 pixels
        pts, vis = s["pts"].copy(), s["vis"].copy()  # Copies coordinates and visibility labels for augmentation

        if self.train:  # Applies random augmentations only during training
            if random.random() < 0.5:  # Chooses horizontal flipping with a 50% probability
                img = img[:, ::-1]  # Flips the image horizontally
                pts[:, 0] = (C.IMG_SIZE - 1) - pts[:, 0]  # Mirrors the x-coordinate of every keypoint
                pts, vis = pts[C.FLIP_PERM], vis[C.FLIP_PERM]  # Swaps left/right keypoint labels after flipping
            img = img.astype(np.float32) / 255.0  # Converts pixel values to floating-point values between zero and one
            img = img * random.uniform(0.7, 1.3) + random.uniform(-0.1, 0.1)  # Randomly changes image brightness and intensity
            img = (img - img.mean()) * random.uniform(0.8, 1.2) + img.mean()  # Randomly adjusts contrast around the image mean
            img = img * np.random.uniform(0.9, 1.1, size=3).astype(np.float32)  # Applies separate random intensity changes to RGB channels
            if random.random() < 0.3:  # Selects Gaussian blur with a 30% probability
                img = cv2.GaussianBlur(img, (3, 3), 0)  # Applies a small Gaussian blur to the image
            img = np.clip(img + np.random.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)  # Adds random noise and clips pixel values to zero through one
        else:  # Handles validation samples without random training augmentation
            img = img.astype(np.float32) / 255.0  # Converts pixel values to floating-point values between zero and one

        x = torch.from_numpy(((img - self.mean) / self.std).transpose(2, 0, 1).copy())  # Normalizes the image, rearranges channels, and converts it into a PyTorch tensor
        return (x, torch.from_numpy(make_heatmaps(pts, vis)),  # Returns the image tensor and its target keypoint heatmaps
                torch.from_numpy(vis), torch.from_numpy(pts))  # Returns visibility indicators and keypoint coordinates