# Endterm — CNN Verifier for YOLOv8n Pedestrian Detections

Student: **Bolatov Bekdaulet, SE-2405** · Course: Machine Learning (AMO 3208) · Instructor: Mimenbayeva A.B.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BekdauletBol/ml-endterm-pedestrian-verifier/blob/main/Endterm_Bolatov_Bekdaulet.ipynb)

This is the endterm project of my Machine Learning course. It continues my Assignment 2 and my diploma
project [public.safe](https://github.com/BekdauletBol/public_final). public.safe uses phones as cameras
and the laptop runs **YOLOv8n** to find people in danger zones. YOLOv8n sometimes calls a toy, a statue or
an animal a "person", and every such box inside a zone is a false alarm.

My model looks at the crop of every YOLOv8n "person" box and says **pedestrian**, **other_object** or
**background**. Only *pedestrian* boxes go to the zone rules.

## Results

Test set of PedCrops v1 (3,085 crops, 738 photos that were not used for training):

| Model | Test accuracy | Macro F1 | Pedestrian precision | Pedestrian recall | Test loss | Training time |
|---|---|---|---|---|---|---|
| Config 1: tanh, 64 neurons, batch 1024 | 0.805 | 0.803 | 0.722 | 0.779 | 0.466 | 19.2 min (30 epochs) |
| Config 2: tanh, 32 neurons, batch 128 | 0.805 | 0.803 | 0.719 | 0.776 | 0.482 | 6.6 min (18 epochs, early stop) |
| Config 3: relu, 32 neurons, batch 1024 | 0.808 | 0.804 | 0.705 | 0.811 | 0.463 | 8.6 min (30 epochs) |
| Improved CNN: + BatchNorm, Dropout, augmentation, AdamW, cosine LR | 0.853 | 0.850 | 0.822 | 0.810 | 0.370 | 17.2 min (30 epochs) |
| MobileNetV2, only the new head trained | 0.838 | 0.839 | 0.878 | 0.842 | 0.422 | 4.2 min (3 epochs) |
| **MobileNetV2, full fine-tuning** | **0.928** | **0.928** | **0.925** | **0.941** | **0.214** | 28.0 min (10 epochs) |

All three configurations use Adam (lr 0.001), Xavier-uniform initialization and 1 hidden layer, as in the
task table. Training was done on my MacBook (Apple M1, 8 GB, `mps`).

**Practical test on real YOLOv8n boxes.** I ran YOLOv8n (the same weights as in public.safe) on the
738 test photos. It gave 1,184 "person" boxes: 939 true and 245 false by the COCO labels. With the verifier
(threshold 0.5):

| | YOLOv8n only | YOLOv8n + verifier |
|---|---|---|
| true pedestrians kept | 939 | 903 (−3.8 %) |
| false positives | 245 | 213 (−13 %) |
| precision | 0.793 | 0.809 |
| recall | 0.603 | 0.580 |
| extra time | — | about 2 ms per box (32 ms for 16 boxes, M1) |

The reduction is small, and I checked why. 114 of the 245 "false positives" overlap a real person, and
in my manual check of 40 kept "no person" boxes, 27 showed a real person who simply has no box in COCO.
So the verifier is right about most of the boxes it keeps. See section 8 of the notebook and the report.

## Files

| Path | What it is |
|---|---|
| `Endterm_Bolatov_Bekdaulet.ipynb` | the notebook with all steps and outputs |
| `report/` | the report (PDF and DOCX) |
| `src/data.py` | loading the dataset, GPU augmentation, normalization |
| `src/models.py` | my CNN (`PedCNN`) and MobileNetV2 transfer learning |
| `src/train.py` | training loop: validation, early stopping, gradient accumulation |
| `src/evaluate.py` | metrics, confusion matrix, curves |
| `src/verified_detector.py` | YOLOv8n + verifier with the same `detect()` method as the public.safe detector |
| `scripts/make_yolo_detections.py` | runs YOLOv8n on the test photos and marks every box as TP or FP |
| `data/pedcrops_v1/` | **the dataset**: `train.zip`, `val.zip`, `test.zip`, `metadata.csv`, `stats.json` |
| `data/build_dataset.py` | how the dataset was built from COCO (Assignment 2) |
| `data/yolo_test_detections/` | the 1,184 YOLOv8n boxes on the test photos (crops, CSV, demo photos) |
| `models/` | trained weights of all 6 models and `yolov8n.pt` |
| `results/` | training histories, result tables and all figures |

## Dataset: PedCrops v1

* Source: COCO val2017 (4,952 photos, 36,781 boxes) from the Hugging Face mirror `detection-datasets/coco`.
* 21,669 RGB crops of 128×128 pixels, 3 balanced classes (7,223 each): pedestrian, other_object, background.
* Split 70/15/15 **by photo**, so no photo is in two splits: train 15,260 · val 3,324 · test 3,085.
* COCO annotations are under CC BY 4.0; the photos come from Flickr under their own licenses
  (see [cocodataset.org](https://cocodataset.org/#termsofuse)).

## How to run

Google Colab: click the badge above and run all cells. The first cell clones this repository.

On a computer:

```bash
git clone https://github.com/BekdauletBol/ml-endterm-pedestrian-verifier.git
cd ml-endterm-pedestrian-verifier
pip install -r requirements.txt
jupyter notebook Endterm_Bolatov_Bekdaulet.ipynb
```

By default `TRAIN = False`, so the notebook loads my saved weights and runs in a few minutes.
Set `TRAIN = True` in the first cell to train all models again (about 1.5 hours on my laptop).

Verifier demo on any photo (green = kept, red = rejected):

```bash
python -m src.verified_detector path/to/photo.jpg --out result.jpg
```
