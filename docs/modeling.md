# Modeling notes

Details behind the commands in the README: how windows are labelled, what the models read, how training works and how tuning picks a setting.

## How windows are labelled

Frames are cut into windows of `eval.window_len` frames (about 0.2 s each, measured from the capture timestamps). Windows that span a gap of dropped frames are skipped.

- **`any_lying` (default):** the classifier predicts whether lying **occurred** within the window's 3.2 s. A window is labelled lying if at least one of its frames is lying, and not lying if none is. If the window has several lying frames, the last one decides between `lying_left` and `lying_right` (when `eval.collapse_lying=false`).
- **`last`:** a window takes the label of its last frame.

Why `any_lying`: in the labelled data, lying is annotated in short runs of 1 to 5 frames (0.2 to 1 s), scattered through recordings that last about 6 minutes. With `last`, a window is only positive when it happens to end on one of those frames: with 16-frame windows and stride 8, that kept 7 lying windows out of 1268. Requiring most of the window to be lying (majority) kept none. With `any_lying`, the same data gives 37 lying windows, and each person has some.

Consequences to keep in mind when reading scores:

- **What a positive means:** lying occurred at least once in the window's 3.2 s. It does not mean the person was lying for the whole window, and a positive window may contain 15 non-lying frames. Do not read lying recall or precision as "how long someone lay" or "how well a lying posture is held". They measure detection of lying occurring.
- **What a negative means:** none of the window's frames is lying. A window is not-lying only if lying never occurred in it, so a window next to a lying episode can be a negative.
- Windows overlap by half, so one lying frame can label two neighbouring windows. The 37 lying windows come from 33 lying frames in 6 recordings, so they are not independent.
- Windows are cut after the preprocessing filters, so recordings without a known person are never windowed.

## Models and their inputs

| `model.name` | Input per window | Notes |
| --- | --- | --- |
| `rnn` | the 16 range-Doppler heatmaps (16 x 256 x 16), log-scaled | CNN per frame, GRU over the frames. About 131k parameters. |
| `pointnet` | the radar **point cloud** of the window | Mini PointNet, about 27k parameters. See below. |
| `baseline` | none | Always predicts the most common training label. |

All models use the same windows, folds, preprocessing, class weighting, validation loss and early stopping (`models/torch_classifier.py`), so their results are comparable. Each model declares what it reads with `input_type`, and `make_windows(inputs=...)` builds it.

**The point-cloud input.** The radar reports 0 to 16 points per frame (median 2; 27% of labelled frames have none, and the radar's tracker never reports a track). For a window the points of all its frames are put into one set of at most `eval.max_points` (default 128) points with the columns `x, y, z, doppler, snr, time` (time is 0 for the first frame of the window and 1 for the last) and a flag marking real points versus padding. A 16-frame window holds about 41 points (99th percentile 105), so 128 truncates 0.3% of windows; if there are more, the strongest by SNR are kept. The radar's noise value and its track id (always 255) are not used.

**The mini PointNet** applies the same small MLP (6 to 32 to 64 to 128 features) to every point, pools the valid points with max and mean (padding never reaches the pooled features, and the order of the points does not matter), adds the log of the number of points (lying windows tend to have more) and classifies with a small head. There is no alignment network (T-Net). x, y, z, doppler and snr are standardised with statistics of the training points. Run it with `python main.py model.name=pointnet`; its hyperparameters are under `model.pointnet.*`. Regularisation (all under `model.pointnet.*`): `dropout` (head, 0.5), `point_dropout` (per-point features, 0.2), `weight_decay` (1e-2), and two augmentations that only act while training: `point_drop` (drops 30% of the points) and `jitter` (noise on the standardised features). Set them to 0 to switch them off.

## Training logs, early stopping and checkpoints

Every `python main.py` run writes a folder `results/<timestamp>_<model>/` containing:

| File | What it is |
| --- | --- |
| `report.md` | Short report: settings, pooled results, per-fold and per-person tables, the loss figures and checkpoint paths. |
| `loss_per_fold.png` | One panel per fold: train loss at every step, validation loss at every evaluation, the step whose weights were kept, and where early stopping fired. |
| `loss_mean.png` | Train and validation loss against epochs as the mean over folds, with a shaded band of ±1 standard deviation across folds. A point needs at least two folds still training. |
| `results.json` | Everything above in machine-readable form, including the full per-step loss history of each fold. |
| `checkpoints/fold<k>_<test person>.pt` | The best weights of each fold (the ones kept after early stopping). |

Rebuild the report and figures of an old run with `python -m utils.run_report results/<run>`. The checkpoints load with `BaseModel.load`.

How training works for the `rnn` model:

- **Validation person:** in every fold, one of the training people is held out as validation data (chosen with `eval.seed`), so the validation loss measures generalisation to an unseen person. The test person is never used for validation. The model therefore trains on one person fewer than the fold's training set, which with 4 people is a real cost. With fewer than two training people there is no validation and the model trains for all epochs.
- **Early stopping:** `model.rnn.patience` (default 5). Training stops after that many validation evaluations in a row without a lower validation loss, and the weights with the lowest validation loss are restored. `model.rnn.epochs` (default 30) is only the upper limit.
- **Regularisation** (`model.rnn.*`): `dropout` (GRU output, 0.5), `feature_dropout` (per-frame CNN features, 0.3), `weight_decay` (1e-2) and `noise` (augmentation, noise on the standardised heatmap while training, 0.05). Set them to 0 to switch them off. The PointNet has its own set under `model.pointnet.*`.
- **When validation runs:** `model.rnn.evals_per_epoch` times per epoch (default 1, so patience 5 means 5 epochs). With about 10 to 30 steps per epoch, raising this makes patience cover very few steps; the train loss is always recorded at every step.
- **The loss** in the figures is cross-entropy that treats the classes equally, not the evaluation metric. The train loss weights each class by the inverse of its frequency in the training windows (so it is plain cross-entropy once the training data is balanced). The validation loss is class-balanced: the mean of the per-class mean losses, with class weights that come from `val_weight_labels` (by default the validation set itself). A model that always predicts 50/50 scores ln 2 = 0.693 on it, whatever the class sizes, and that is the level to compare against. During tuning the weights are pooled over all training people of the outer fold (see below). A validation loss hovering around 0.69 means the model has learned nothing that carries over to a new person, and early stopping is then only picking noise.
- **Fixed epochs, no validation** (`train.validation=false`): every fold trains for exactly `model.rnn.epochs` epochs on all its training people, keeps the last weights, and the figures show the train loss only. Use it when one validation person is too noisy to decide anything (with 4 people it is). The epoch count must come from earlier validation curves, never from test scores. Example: `python main.py train.validation=false model.rnn.epochs=3`.
- **Oversampling** (`train.oversample`, default off): repeats randomly chosen training windows of the rarer class until each reaches that fraction of the largest class (`1.0` = balanced, `0.5` = half). It is applied to the training part of each fold only; the validation person and the test person keep their real class balance. It repeats existing windows and adds no new information, so it can make the model memorise the few lying windows. The report's per-fold results and `results.json` record the class counts after resampling.
- **Heatmap preprocessing** (`data.preprocess=[config_4, range_normalise]`, both steps off by default, they rewrite the heatmaps once before windowing, one recording at a time, never pooling statistics across recordings). `range_normalise` subtracts each range bin's noise floor (a low percentile over the recording's time and Doppler axes, smoothed along range with a running median wider than a lying person), so a pixel means "above the floor at this range". The floor is estimated from the whole recording, including frames after the current one, so it works on recorded data but could not run online as is. `temporal_background` subtracts a slow moving average of each cell (`tau_frames`, default 300 frames, about a minute), using earlier frames only so it is causal; anything that stays put for much longer than that, including a person lying still, fades into the background. A recording whose heatmaps do not have the expected shape is left unchanged by both, so keep `config_4` in the list.
- **Heatmap augmentation** (`model.rnn.augment.*`, default off, RNN only): applied to training batches only, never to validation or test windows. Sized for Config_4, the only sensor config used for model development (0.0237 m range bins: that is the bin spacing; the physical range resolution is a coarser 0.042 m), so `config_4` must stay in `data.preprocess`. `range_shift_bins=13` moves the whole scene up to 13 bins (about 0.3 m) along range, as if the sensor hung higher or lower, with fractional shifts interpolated and nothing wrapped around. `floor_jitter=50` adds a smooth random baseline along range (in raw heatmap units; the empty-room floor varies by about 100-180 across range and recordings). `time_reverse=true` plays half of the windows backwards and needs `eval.label_rule=any_lying`, because the `last` rule takes the label of the last frame. There is no frame masking: lying is annotated in runs of 1 to 5 frames and `any_lying` labels a window from as few as one, so masking frames would often erase the evidence while the label stays `lying`. `floor_jitter` is meant for the raw heatmap, so do not combine it with the `range_normalise` preprocessor (which removes the floor it varies). Example: `python main.py model.rnn.augment.range_shift_bins=13 model.rnn.augment.floor_jitter=50`. See `src/data/augment.py`.
- **Point cloud augmentation** (`model.pointnet.augment.*`, default off, PointNet only): training batches only, sized for Config_4. The points are in the sensor frame, where `y` is the boresight (the distance to the floor, about 1.8 m for a person) and the floor plane is `x`-`z`; `z` is not height. `shift_m=0.3` moves a window along the floor, `height_m=0.2` along the boresight (points pushed past the 4.85 m range limit are dropped, and every point's SNR is rescaled by 40*log10(r_old/r_new) dB since power falls as R^-4), `tilt_deg=5` tilts the sensor, `ghost_points=3` adds on average 3 false alarms per window into empty slots, `snr_jitter_db=2` shifts a window's SNR. `yaw_deg=180` (rotation about the boresight) and `mirror=true` change which way a person lies, so they need `eval.collapse_lying=true`. Dropping points and jitter already exist as `model.pointnet.point_drop` and `jitter`. Example: `python main.py model.name=pointnet model.pointnet.augment.shift_m=0.3 model.pointnet.augment.tilt_deg=5`. See `src/data/augment.py`.
- Models that do not train iteratively (`baseline`) have no loss curves; their report just omits that section.

## Tuning details

**Why the validation class weights are pooled.** Settings are ranked by the mean validation loss over the inner validation people, so those losses must be on one scale. If each person's loss were weighted by that person's own class mix, they would not be: a person with only `not_lying` windows gets weight 1 on that class and a plain cross-entropy, which is much easier to get low than a balanced loss on a person who also has `lying` windows. The mean would then reward whichever setting does well on the easiest validation people, and the winner would be the best setting for the easiest case instead of the best setting overall. Pooling the class weights over all people of the outer training fold gives every validation person, and every setting, the same weights. The loss is then a fair comparison across people and settings, and a model that predicts 50/50 scores ln 2 = 0.693 for everyone.

`tune.grid` is a list of overrides of the chosen model's hyperparameters, `model.rnn` or `model.pointnet` (`hidden`, `lr`, `weight_decay`, `epochs`, ... or `width`, `dropout`, `point_drop`, ...). Write a space after each colon (`hidden: 64`, not `hidden:64`), or YAML reads the whole thing as one key; unknown keys are rejected. Tuning is defined for `model.name=rnn` and `model.name=pointnet` (not the baseline), and it replaces the single random validation person of `train.validation`. A single-setting grid is a way to choose only the epoch count. Settings not named in the grid keep the values from the config (or the command line) for the tuning run; the full run applies the saved setting on top of the defaults, so pass the same other overrides (such as `train.oversample`) to both steps.

Caveats:

- With 4 people the inner loop validates on one person at a time, who has only a few lying windows (they overlap, see "How windows are labelled"). Tuning is noisy, so keep the search small.
- The cost is (people - 1) trainings per setting and per outer fold, plus one refit, so 4 x 3 = 12 trainings per setting for 4 people. Run it on the PC (`scripts/pc/train.sh`).
- The outer scores describe the whole procedure (tuning included) on a new person, not the one final model.
- The setting chosen for the final model comes from folds that each saw different people, so it is a vote, not a score.

## Saving and loading a model

Every run saves the best weights of each fold (and `final.pt` for a full run) with the task they were trained for. Load one with:

```python
from models.model import BaseModel

model, task = BaseModel.load("results/<run>/checkpoints/fold1_<person>.pt")
```

`task` is a `utils.task.Task` (target, collapse_lying, window_len, label_rule, max_points). To save by hand, call `model.save(path, task)`. A checkpoint holds the weights, the input standardisation, the class labels and the model settings.
