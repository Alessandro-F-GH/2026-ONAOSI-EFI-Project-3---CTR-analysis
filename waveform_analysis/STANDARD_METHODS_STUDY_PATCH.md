# LED and CFD study-result patch

- Enables `standard_methods: [led, cfd]` by default.
- Evaluates both on the exact CV validation and blind masks used by ML.
- Stores target-specific energy and timing CFD timestamps during preprocessing.
- Adds LED/CFD rows to `_state/all_results.csv` and `model_loss_results.csv`.
- Allows either standard method to win `summary_results.csv` by the configured
  CV selection metric.
- Creates no standard-method checkpoints.
- Keeps standard methods out of the window-size plot because they have no window.
