# Compact CSV results patch

The experiment result export is now numeric and non-redundant.

## `all_results.csv`

The CSV contains only numeric cells. Repeated strings and paths were removed:

- no absolute ROOT path per row;
- no experiment name per row;
- no repeated model/loss type strings;
- no repeated window label;
- no JSON hyperparameter payload;
- no free-text error message;
- categorical values use integer codes;
- the 96-bit internal row key is represented by two exact 48-bit integers.

## `results_metadata.json`

A sidecar file stores information once:

- experiment name;
- ROOT identifiers and paths;
- categorical codebooks;
- model and loss labels/types;
- window labels;
- one parameter dictionary per trial;
- failure messages keyed by row identifier.

Resume and plotting still work because the study runner reconstructs the original
internal rows when reading the numeric CSV. Existing legacy CSVs are accepted and
converted on their next write.
