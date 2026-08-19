# weights/

`train.py` saves the best checkpoint here as `model.pt` (full model state:
frozen encoder + trained decoder, a few MB). It is committed once trained, so
`predict.py` needs no download at inference time.

This folder ships empty because the model must be trained on Fashionpedia first
(see the repo README, "How to reproduce training"). After training, `model.pt`
appears here automatically.
