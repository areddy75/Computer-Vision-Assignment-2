# Assignment 2: Self-Supervised Visual Representation Learning

ITCS 6169/8169 Computer Vision. Sixteen-class image classification from **2,400 training images**, using
self-supervised learning (SSL). **No external pretraining**: every network is randomly initialised and trained only
on the provided training images.

**Best test accuracy: 79.75% (319/400)**. The model is a small ResNet pretrained with SimCLR on the training
images, then fine-tuned with a joint cross-entropy + contrastive loss, with horizontal-flip TTA.

## Results (single seed, from `runs/experiment_summary.csv`)

| Method (same 1.2 M-param backbone) | Val % | Test % |
|---|---|---|
| E0 Random-init backbone, frozen + linear probe | 34.0 | 28.5 |
| E1 Supervised baseline (cross-entropy only, 200 ep) | 79.4 | 74.0 |
| E2a SimCLR (500 ep) → frozen linear probe | 76.5 | 74.3 |
| E2b SimCLR → full fine-tune (100 ep) | 82.3 | 78.3 |
| E2c Control: random init, same fine-tune recipe | 78.3 | 75.0 |
| E3a Rotation prediction → linear probe | 61.3 | 62.3 |
| E3b Rotation prediction → fine-tune | 79.6 | 76.0 |
| E4a Joint CE + SimCLR from scratch (λ = 0.5) | 81.5 | 77.0 |
| E4b Control: same two strong views, λ = 0 | 82.1 | 75.3 |
| **E7 Best: SimCLR init → joint fine-tune + flip TTA** | **82.3** | **79.8** |

The notebook also contains an augmentation ablation for SimCLR (E5), a label-efficiency study with 10/25/50/100%
of the labels (E6), train/val curves, and the test confusion matrix. The 2-page report explains the findings.

## Repository layout

```
Assignment2_SSL.ipynb     all training + evaluation code, executed with outputs (E0-E7)
models.py                 model definition and preprocessing (same as notebook section 3)
predict.py                inference / evaluation with the saved checkpoint
requirements.txt
runs/
  best_model.pt           best checkpoint (E7), with class names and normalisation stats
  config.json             epochs, augmentation settings, seed, image size
  experiment_summary.csv  val/test accuracy of every experiment
  results_summary.json    every number used in the report
  E*.json                 per-experiment training logs (loss/accuracy per epoch)
  test2_predictions.csv   predictions for the unlabelled test2 images
Assignment2_Report.pdf    2-page report
AI_USAGE.md               how AI coding tools were used
```

## Data

The dataset is not included. Place it next to the notebook like this:

```
data/train/<ClassName>/*.jpg   (16 classes x 150 images)
data/test/<ClassName>/*.jpg    (16 classes x 25 images)
data/test2/*.jpg               (400 unlabelled images)
```

All images are converted to grayscale, because every *Flower* image is RGB while every other class is grayscale.
Without this, colour alone would reveal the Flower class.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3, PyTorch 2.15 on an Apple-silicon Mac (MPS backend). The code picks CUDA, then MPS, then CPU.

## Evaluate the trained model (inference)

```bash
python predict.py --data data/test                     # prints accuracy: 79.75% (319/400)
python predict.py --data data/test2 --out test2.csv    # predictions for unlabelled images
```

`predict.py` loads `runs/best_model.pt`, applies the same preprocessing as the notebook (grayscale, 72x72, resized to
64x64, normalised) and averages the predictions of the image and its horizontal flip. Use `--no-tta` to turn that off.

## Reproduce all experiments

```bash
BUDGET=full RUN_DIR=runs jupyter nbconvert --to notebook --execute Assignment2_SSL.ipynb \
    --output Assignment2_SSL_rerun.ipynb --ExecutePreprocessor.timeout=-1
```

* `BUDGET=full` uses the epochs reported above (supervised 200, SimCLR 500, rotation 200, fine-tune 100, joint 200,
  ablation 100, best recipe 200). The run took **55 minutes on an Apple M-series GPU (MPS)**. `BUDGET=reduced` uses
  about half the epochs for slower machines. Without the variable, CUDA defaults to `full` and other devices to `reduced`.
* Every experiment saves `runs/<name>.pt` and `runs/<name>.json` and is **skipped when those files exist**. To retrain
  from scratch, point `RUN_DIR` at an empty folder or set `FORCE_RERUN = True` in the notebook.
* For a quick check that everything runs, use `EPOCH_SCALE=0.02` (about 2 minutes).
* On Google Colab, upload this folder (with `data/`) to the top level of *My Drive*, choose a T4 GPU and *Run all*.
  The first cell mounts Drive.
* Seed 0 throughout, with a stratified 80/20 train/val split of the training set. The test set is evaluated once, at the end.
  GPU kernels are not bit-exact across devices, so a rerun can differ slightly (typically < 1 point).

## Method summary

* **Backbone:** small ResNet (stride-2 3x3 stem, four residual stages with 32/64/128/256 channels, global average
  pooling, 256-d features, 1.2 M parameters). The same backbone is used in every experiment.
* **SimCLR:** two augmented views of each image, a projection MLP (256→512→128) and the NT-Xent loss with τ = 0.2 and
  batch 256. Augmentations: random-resized crop (0.2-1), flip, brightness/contrast jitter, Gaussian blur and solarise
  (grayscale only). Pretraining uses the 1,920 training-split images without labels.
* **Best recipe (E7):** start from the SimCLR backbone and projector, then fine-tune for 200 epochs with
  `CE(two strong views) + 0.5 · NT-Xent`, AdamW (head lr 1e-3, backbone lr 5e-4), cosine schedule, label smoothing 0.1,
  and horizontal-flip TTA at test time.
