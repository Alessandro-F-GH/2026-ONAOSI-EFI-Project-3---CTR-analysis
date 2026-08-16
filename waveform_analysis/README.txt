Replace these files in the repository root:

  ml_pipeline/torch_data.py
  ml_pipeline/training_utils.py
  ml_pipeline/study.py

Target definition:
  delta_j = t_LED,j - t_anchor,j
  Delta_delta = delta_1 - delta_2
  y_ML = Delta_t_LED - Delta_delta - true_TOF

Evaluation:
  total_led_correction = model_prediction
  corrected = Delta_t_LED - model_prediction
  NO anchor/alignment term is added analytically.

The study target protocol version is bumped to native_alignment_residual_subtracted_v2,
so start once with --restart. Preprocessing does not need to be rebuilt.

Then future interrupted runs can use --resume.
