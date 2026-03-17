# TEL2 Reconstruction Parameter Sweeps

The `v2` TEL2 sweep tracks the updated reconstruction recipe used by the refreshed
ALOS2 Poly-LPS configs while keeping the fork-specific workflow intact:

- dataset: `Tel2Commrcial_v1`
- task: reconstruction with `AutoEncoderWD`
- logger: Aim only
- log root: `/data/equiv-cvnn/logs`
- epochs: `500`
- transform: `LogAmplitude`
- patch size / stride: `64 / 64`
- dataset path: remote `$TMPDIR/data`

The old 400-epoch sweep configs remain in this folder as historical references.
The active sweep runner uses the new `v2` files listed below.

Reference point:

- updated TEL2 baseline: `channels_ratio=48`, `num_layers=2`
- same parameter point in the sweep: `width_cr48_l2_v2`

## Width Sweep (`num_layers=2`)

| Label | Config | channels_ratio | num_layers | Total params |
| --- | --- | ---: | ---: | ---: |
| `width_cr24_l2_v2` | `config_tel2commercial_mischiefreef_20240526_cvnn_poly_lps_width_cr24_l2_v2.yml` | 24 | 2 | 408,710 |
| `width_cr36_l2_v2` | `config_tel2commercial_mischiefreef_20240526_cvnn_poly_lps_width_cr36_l2_v2.yml` | 36 | 2 | 917,186 |
| `width_cr48_l2_v2` | `config_tel2commercial_mischiefreef_20240526_cvnn_poly_lps_width_cr48_l2_v2.yml` | 48 | 2 | 1,628,414 |
| `width_cr60_l2_v2` | `config_tel2commercial_mischiefreef_20240526_cvnn_poly_lps_width_cr60_l2_v2.yml` | 60 | 2 | 2,542,394 |
| `width_cr72_l2_v2` | `config_tel2commercial_mischiefreef_20240526_cvnn_poly_lps_width_cr72_l2_v2.yml` | 72 | 2 | 3,659,126 |

## Depth Sweep (`channels_ratio=48`)

| Label | Config | channels_ratio | num_layers | Total params |
| --- | --- | ---: | ---: | ---: |
| `depth_cr48_l1_v2` | `config_tel2commercial_mischiefreef_20240526_cvnn_poly_lps_depth_cr48_l1_v2.yml` | 48 | 1 | 344,510 |
| `depth_cr48_l3_v2` | `config_tel2commercial_mischiefreef_20240526_cvnn_poly_lps_depth_cr48_l3_v2.yml` | 48 | 3 | 6,758,270 |
| `depth_cr48_l4_v2` | `config_tel2commercial_mischiefreef_20240526_cvnn_poly_lps_depth_cr48_l4_v2.yml` | 48 | 4 | 27,266,174 |

## Suggested execution order

1. `depth_cr48_l1_v2`
2. `width_cr24_l2_v2`
3. `width_cr36_l2_v2`
4. `width_cr48_l2_v2`
5. `depth_cr48_l3_v2`
6. `width_cr60_l2_v2`
7. `width_cr72_l2_v2`
8. `depth_cr48_l4_v2`
