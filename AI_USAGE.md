# Use of AI coding tools

Claude Code (Anthropic) was used as a pair programmer for this assignment:

* implementing the training pipeline in `Assignment2_SSL.ipynb`: batched grayscale augmentations, the small ResNet,
  the NT-Xent (SimCLR) loss, rotation prediction, the joint objective, kNN and linear-probe evaluation, all as per instruction;
* setting up the run on an Apple-silicon GPU, plus a smoke test with tiny epoch counts before the full run;

All reported numbers come from the executed notebook (`runs/experiment_summary.csv`, `runs/results_summary.json`).
`predict.py` was checked against the notebook: it gives the same 79.75% test accuracy and identical test2 predictions.
I reviewed the code, the experiment design (matched controls E2c and E4b, augmentation ablation, label-efficiency
study) and the interpretation, and I can explain each component.
