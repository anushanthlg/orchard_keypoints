# Orchard keypoint detection

Detects 4 keypoints in orchard-row images (256x256): `vanishing_point`, `vanishing_point_right`,
`left_intercept`, `right_intercept`. The trunk classes in the JSON are ignored.

## Structure

```
orchard_keypoints/
├── src/
│   ├── config.py         paths, keypoint names, sizes, training defaults
│   ├── data.py           JSON parsing, saved train/val split, Dataset + augmentation
│   ├── model.py          KPNet: ResNet-18 + deconv head -> heatmaps
│   ├── utils.py          heatmap encode/decode, preprocessing, drawing
│   ├── verify_labels.py  draw ground truth on every image, print label stats
│   ├── train.py          training -> runs/best.pt, last.pt, history.json, splits.json
│   ├── test.py           mean/median pixel error + PCK per keypoint on the val split
│   └── inference.py      predict on an image/folder -> overlays + predictions.json
├── requirements.txt
├── README.md
├── orchard_keypoint_json.json     (you add: annotations)
├── orchard_keypoint_images/       (you add: images)
└── runs/                          (created automatically)
```

## Usage (run from the project root)

```bash
pip install -r requirements.txt
python src/verify_labels.py           # 1. check the labels look right (runs/label_check/)
python src/train.py                   # 2. train
python src/test.py --save_overlays    # 3. evaluate on the val split (white = GT)
python src/inference.py --images orchard_keypoint_images
```

Default data locations are set in `src/config.py` (`JSON_PATH`, `IMAGES_DIR`) and can be
overridden with `--json` / `--images`.

## Notes

- The split is saved to `runs/splits.json`, so `test.py` uses the same validation images
  as training. Use `python src/train.py --new_split` to re-split.
- Augmentation is photometric + horizontal flip (left/right classes swapped). Geometric
  augmentation is avoided because the intercepts lie on the image border.
- Images without any of the 4 target keypoints are skipped; missing keypoints within an
  image are masked out of the loss.
