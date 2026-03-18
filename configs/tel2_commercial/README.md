# TEL2 Commercial Reconstruction Notes

## Input scaling finding

The initial TEL2 reconstruction pipeline reused `LogAmplitude` with the default
`[0.02, 40]` clamp that works for the San Francisco ALOS-2 reconstruction task.
That turned out to be a bad fit for the TEL2 commercial scene.

A stats report generated with `scripts/report_tel2_dataset_stats.py` showed:

- TEL2 raw amplitude p95 values were on the order of `1e6`
- `100%` of TEL2 pixels in the training-view channels were above the
  `LogAmplitude.max_value=40` clamp
- after `LogAmplitude`, every TEL2 channel collapsed to `1.0` everywhere

By contrast, the original ALOS-2 reconstruction config stays mostly inside the
same clamp range, with negligible saturation above `40`.

## Current fix

TEL2 now supports a dataset-level raw scaling factor that is applied before
`LogAmplitude`. The default TEL2 setting is:

```yaml
data:
  dataset:
    raw_scale_factor: percentile
    raw_scale_percentile: 99.5
```

This computes a single scalar from the selected TEL2 training-view channels and
divides the complex raw data by that value before downstream transforms.

## Regenerating the report

Run the report against a TEL2 config:

```bash
python scripts/report_tel2_dataset_stats.py \
  configs/tel2_commercial/config_tel2commercial_mischiefreef_20240526_cvnn_poly_lps.yml
```

The report writes:

- `dataset_stats.md`
- before/after `LogAmplitude` histograms
- full-image visualizations
- a sampled patch-row visualization
