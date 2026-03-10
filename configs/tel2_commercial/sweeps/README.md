# TEL2 Reconstruction Parameter Sweeps

These sweeps explore model capacity for the TEL2 commercial reconstruction task while
keeping the data pipeline, optimizer, projection class, and training length fixed.

Common settings:

- dataset: `Tel2Commrcial_v1`
- task: reconstruction with `AutoEncoderWD`
- logger: Aim
- log root: `/data/equiv-cvnn/logs`
- epochs: `400`
- transform: `LogAmplitude`
- patch size / stride: `64 / 64`

Reference points:

- current TEL2 full run: `channels_ratio=16`, `num_layers=4`
- same parameter point in the sweep: `width_cr16_l4`

## Width Sweep

Hold depth fixed at `num_layers=4` and vary the base channel width.

| Label | Config | channels_ratio | num_layers | Total params |
| --- | --- | ---: | ---: | ---: |
| `width_cr08_l4` | `config_tel2commercial_mischiefreef_20240526_cvnn_poly_lps_width_cr08_l4.yml` | 8 | 4 | 760,694 |
| `width_cr12_l4` | `config_tel2commercial_mischiefreef_20240526_cvnn_poly_lps_width_cr12_l4.yml` | 12 | 4 | 1,708,586 |
| `width_cr16_l4` | `config_tel2commercial_mischiefreef_20240526_cvnn_poly_lps_width_cr16_l4.yml` | 16 | 4 | 3,034,846 |
| `width_cr20_l4` | `config_tel2commercial_mischiefreef_20240526_cvnn_poly_lps_width_cr20_l4.yml` | 20 | 4 | 4,739,474 |
| `width_cr24_l4` | `config_tel2commercial_mischiefreef_20240526_cvnn_poly_lps_width_cr24_l4.yml` | 24 | 4 | 6,822,470 |

## Depth Sweep

Hold width fixed at `channels_ratio=12` and vary depth.

| Label | Config | channels_ratio | num_layers | Total params |
| --- | --- | ---: | ---: | ---: |
| `depth_cr12_l2` | `config_tel2commercial_mischiefreef_20240526_cvnn_poly_lps_depth_cr12_l2.yml` | 12 | 2 | 102,986 |
| `depth_cr12_l3` | `config_tel2commercial_mischiefreef_20240526_cvnn_poly_lps_depth_cr12_l3.yml` | 12 | 3 | 424,682 |
| `depth_cr12_l5` | `config_tel2commercial_mischiefreef_20240526_cvnn_poly_lps_depth_cr12_l5.yml` | 12 | 5 | 6,838,442 |

## Suggested execution order

If the goal is to map capacity against reconstruction quality without jumping
immediately to the largest memory footprint, run them in this order:

1. `depth_cr12_l2`
2. `depth_cr12_l3`
3. `width_cr08_l4`
4. `width_cr12_l4`
5. `width_cr16_l4`
6. `width_cr20_l4`
7. `width_cr24_l4`
8. `depth_cr12_l5`

The last two entries are the highest-risk memory points in this sweep and should be
launched only after confirming that the medium-width runs fit comfortably on the target GPU.
