# Judge validation

## Known gap

A genuine Cohen's kappa report requires human labelling of
`results/labels/judge_labels.jsonl`. That labelling has not happened yet, so no
agreement or accuracy claim is reported here.

The `python -m bench.validate_judge --report` code path is implemented and
unit-tested with synthetic hand-labelled fixtures for exact agreement, kappa,
confusion-matrix, and disagreement calculations. It has intentionally not been
run against the unlabelled records.
