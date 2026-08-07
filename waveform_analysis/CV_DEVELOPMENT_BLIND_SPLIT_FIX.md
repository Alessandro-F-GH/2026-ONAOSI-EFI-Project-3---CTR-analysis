# Cross-validation development/blind split fix

CV studies now preprocess each ROOT file directly into:

```text
development | one guard gap | blind
```

The full development set is passed to K-fold cross-validation. No preliminary
validation subset is created or concatenated afterward.

Implementation details:

- generated CV preprocessing uses `development_blind: true`;
- `train_fraction = 1 - blind_fraction`;
- `validation_fraction = 0`;
- `test_fraction = blind_fraction`;
- contiguous splitting removes one guard gap instead of two;
- prepared development datasets store all development indices in `train` and
  an empty `validation` array for compatibility with the canonical format;
- `_fold_masks` rejects any unexpected preliminary validation indices;
- legacy `initial_validation_fraction` keys are removed while loading study
  configurations;
- `--resume` refuses old three-way-split studies and asks for `--restart`, since
  their checkpoints were trained on a different event set.

Regression coverage includes direct contiguous event accounting, generated
preprocessing fractions, and resume compatibility checks.
