"""Central settings shared by all scripts."""  
from pathlib import Path  

ROOT = Path(__file__).resolve().parent.parent  # Gets project root directory
DATA_ROOT = ROOT  # Sets the project root as the data directory
JSON_PATH = DATA_ROOT / "orchard_keypoint_json.json"  # Defines the annotation JSON file path
IMAGES_DIR = DATA_ROOT / "orchard_keypoint_images"  # folder containing orchard images
RUNS_DIR = ROOT / "runs"  # folder for training outputs

KP_NAMES = ["vanishing_point", "vanishing_point_right", "left_intercept", "right_intercept"]  #four keypoint classes
FLIP_PERM = [1, 0, 3, 2]  # Swaps left/right keypoint labels during horizontal flipping

IMG_SIZE = 256  # Sets image dimensions to 256 × 256 pixels
STRIDE = 4  # Sets the spatial downsampling factor to 4
HM_SIZE = IMG_SIZE // STRIDE  # heatmap dimensions-64 × 64 pixels
SIGMA = 2.0  # Sets the Gaussian heatmap standard deviation to 2 pixels

MEAN = (0.485, 0.456, 0.406)  # RGB channel means for image normalization
STD = (0.229, 0.224, 0.225)  # RGB channel standard deviations for normalization

# Training defaults
EPOCHS = 120  # maximum number of training epochs
BATCH_SIZE = 8  # number of images processed per batch
LR = 1e-3  # Sets the learning rate to 0.001
VAL_FRAC = 0.2  # 20% of dataset for validation
SEED = 0  # Sets the random seed for reproducibility