# weights/

`model.pt` is the trained checkpoint, committed here (9.5 MB, well under the
100 MB limit in the brief), so `predict.py` works straight from a clone with no
download step.

It holds the **full** model state — frozen MobileNetV2 encoder plus the trained
decoder — along with the metadata needed to reproduce the run:

| field | value |
|---|---|
| `val_miou` | 0.5722 (best of 30 epochs, 4 training classes) |
| `size` | 256 (training/inference resolution) |
| `seed` | 42 |

Produced by the commands in the repo README, "How to reproduce training". The
full training log is in `garmentimage-training.ipynb`.
