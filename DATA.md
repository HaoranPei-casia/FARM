# Reproducing the experimental population

The repository does not redistribute trajectories, images, extracted features,
or experiment results. It does publish the exact trajectory identities and
splits used by the main FARM experiment:

- [`data_indices/farm_v1_500.csv`](data_indices/farm_v1_500.csv): the complete
  500-trajectory population, source sequence identifiers, environment seeds,
  labels, outer roles, inner-validation roles, and task-specific horizons.
- [`data_indices/strict_unseen_fewshot_splits.json`](data_indices/strict_unseen_fewshot_splits.json):
  the exact five-fold unseen-task adaptation/test identities for model seeds
  5, 9, and 11 and nested budgets 5, 10, 20, 25, 30, 35, and 40 per task.

These files contain identifiers only. They are not measurements or model
outputs.

## Population and split protocol

The population contains 50 trajectories for each of ten LIBERO tasks:

| Code | Task | Common length |
| --- | --- | ---: |
| L3 | `open_the_top_drawer_and_put_the_bowl_inside` | 83 |
| L4 | `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | 122 |
| L5 | `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | 105 |
| L6 | `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 120 |
| S5 | `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 105 |
| S6 | `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 128 |
| S7 | `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 112 |
| U1 | `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 183 |
| U2 | `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 125 |
| U3 | `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` | 78 |

The fixed outer split uses seed `20260829`:

- 30 trajectories from each of L3/L4/L5/L6/S5/S6/S7 for training (210 total)
- 20 trajectories from each seen task for seen evaluation (140 total)
- all 50 trajectories from each U1/U2/U3 task for unseen evaluation (150 total)

The fixed inner-validation split uses seed `20270829` and divides the 210 outer
training trajectories into 168 `INNER_TRAIN` and 42 `INTERNAL_VALIDATION`
trajectories. Readout training was repeated with model seeds 5, 9, and 11.

For strict-unseen adaptation, each source seed has five outer folds. Every fold
holds out 10 trajectories per unseen task for testing. Adaptation identities are
nested within the remaining 40 trajectories per task. Adapt-35 therefore means
35 trajectories for each of U1, U2, and U3 (105 total), not 35 trajectories in
total.

## Interpreting sequence identifiers

`farm_v1_500.csv` has one row per trajectory:

- `trajectory_id` is the canonical identity used in FARM experiments.
- `source_trajectory_id` is the selected source sequence identity.
- `historical_source_trajectory_id` identifies the earlier V1 rollout when the
  row was produced by a V1 replay.
- `environment_seed` is the rollout environment seed recorded by the
  authoritative dataset manifest.
- `source_relative_path` is a machine-independent locator below the original
  fully-instrumented LIBERO dataset root.
- `label=0` means success and `label=1` means failure.
- `outer_role` and `inner_role` reproduce the fixed source-training/evaluation
  protocol.
- `common_length` is the exact prefix length used for both training and scoring.

For an `external_v2` row, a source sequence such as
`libero10_task08_rollout000` and its relative path identify the original task and
rollout directly. For a `v1_replay` row, reproduce the rollout using the recorded
historical identity and environment seed with the same V1 policy/environment
pipeline. The custom V1 collection pipeline is not redistributed here, so the
identifier and seed document the selection but do not by themselves replace
that pipeline.

## Producing FARM inputs

After regenerating a listed trajectory, extract the frozen visual features and
save one array per `trajectory_id` as `[T, 768, 1024]`. The experiment used the
last VLA-JEPA predictor block output before predictor normalization/projection,
with action tokens removed. Truncate each trajectory to its listed
`common_length`, then create the five-column FARM manifest described in the
README using the generated `.npy` path.

Do not resample the public identities when reproducing reported experiments;
use the published outer/inner roles and the exact adaptation/test IDs.
