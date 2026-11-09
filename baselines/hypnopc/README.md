# Hypno-PC: spectral features, pooled PCA and a linear SVM

Baseline folder for the supervised part of the Hypno-PC study: per-epoch spectral features,
PCA pooled over subjects, LinearSVC. Nothing in this folder has been run yet. `upstream/` is a
read-only clone of the authors' code. Claims marked **inferred** come from reading the code or
the data. They are not stated by the authors and have not been checked by running anything.

## Citation and links

- Guendelman M., Shriki O. *The Hypno-PC: uncovering sleep dynamics through principal component
  analysis and hidden Markov modeling of electrophysiological signals.* SLEEP 49(1):zsaf164
  (2026). DOI [10.1093/sleep/zsaf164](https://doi.org/10.1093/sleep/zsaf164).
- Preprint: bioRxiv [10.1101/2025.01.02.631039](https://doi.org/10.1101/2025.01.02.631039)
  (v1, 2 January 2025, CC-BY-NC-ND 4.0; the bioRxiv API lists only v1).
- Code: <https://github.com/miriamguen/high_density_sleep>, cloned at commit
  `d3b0eec8e74ed119d2eef795778d59b6f23e79e4` (2025-06-19, "Bump protobuf from 5.29.3 to
  5.29.5 (#5)"), branch `main`.
- License: **no license**. The repository has no LICENSE or COPYING file, and the GitHub API
  returns `"license": null`. Default copyright applies, so the clone is for reading and
  reference only. We do not redistribute or vendor its code.

Paper versions available here:

| File | What it is | Has the LinearSVC analysis? |
|---|---|---|
| `paper.pdf` | bioRxiv v1 full-text PDF (downloaded) | **No.** v1 has only PCA/ICA/GHMM. The SVM, the 63-PC result and the YASA benchmark were added in revision. |
| `upstream/manuscript_clean_arxiv_update.pdf` | revised manuscript committed in the repo (Word export, 6 June 2025, 96 pp.) | Yes. Its numbers match the published values (κ 0.762, acc 0.826, 63 PCs; κ 0.687 at 5 PCs; YASA κ 0.681). |
| published SLEEP PDF | academic.oup.com returned HTTP 403 to the download | Not downloaded. Put it here as `paper_sleep.pdf` if needed. |

Below, "MS" means `upstream/manuscript_clean_arxiv_update.pdf`. Its section names follow the
published Methods.

## Paper summary

**Problem.** Manual AASM scoring imposes five discrete stages on 30 s epochs. The authors ask
whether a linear, low-dimensional, data-driven representation of overnight PSG + HD-EEG
(1) recovers the hypnogram as a continuous axis, (2) separates the canonical stages, and
(3) supports unsupervised state discovery (GHMM) at 30 s and 4 s resolution.

**Method.** The pipeline has six steps:

1. Filter each modality and re-reference.
2. Compute per-epoch features: EEG relative band powers, aperiodic slope and intercept,
   spectral entropy and total power for every electrode, plus EMG power, EOG power and ECG
   HR/HRV.
3. Z-score each feature within each subject, interpolate, and clip to [−5, 5].
4. Fit PCA on all subjects pooled (90 % of variance: 63 PCs; PC1 is the "Hypno-PC", 42 % of
   variance).
5. Run ICA (Picard) on the PCs that each explain ≥ 2 % of variance (5 PCs).
6. Fit a Gaussian HMM.

As a supervised benchmark of separability, a LinearSVC is trained on the first *k* PCs
(*k* = 1…63) with leave-one-subject-out CV and compared with YASA.

**Data.** ANPHY-Sleep has 29 healthy adults, 83 EEG channels at 1 kHz, 2 EOG, chin and leg EMG,
and ECG. The authors used all 29 nights, including EPCTL08 (MS Suppl. Table S3). Our local copy
has 28 nights; EPCTL08 is missing.

**Main supervised results** (30 s epochs, LOSO over 29 subjects, metrics computed per subject
and then averaged; MS Results and Suppl. Table S1):

| Model | κ | Accuracy | F1 W / R / N1 / N2 / N3 | mean of the 5 stage F1 (inferred "macro-F1") |
|---|---|---|---|---|
| LinearSVC, all 83 EEG + non-EEG, 63 PCs (90 % var.) | **0.762** | 0.826 | 0.840 / 0.798 / 0.452 / 0.869 / 0.877 | 0.767 |
| LinearSVC, same features, 5 PCs | 0.687 | 0.772 | 0.795 / 0.670 / 0.145 / 0.838 / 0.831 | 0.656 |
| LinearSVC, 6 electrodes (F3, F4, C3, C4, O1, O2), 22 PCs | 0.717 | 0.795 | 0.818 / 0.714 / 0.249 / 0.856 / 0.851 | 0.698 |
| LinearSVC, 6 electrodes, 5 PCs | 0.675 | 0.763 | 0.783 / 0.651 / 0.116 / 0.828 / 0.834 | 0.642 |
| YASA (C4-M1, LOC-M2, EMG, age, sex) | 0.681 | 0.763 | 0.685 / 0.744 / 0.167 / 0.824 / 0.870 | 0.658 |
| ICA-GHMM, 4 states, unsupervised (state→stage mapping uses ≤ 1 epoch per training subject) | 0.701 | 0.781 | n/a | n/a |

The "macro-F1" column is the mean of the per-stage F1 values in Suppl. Table S1. Each of those
is itself averaged over subjects, so this is **not** the pooled macro-F1 that our scorer
reports. Our scorer pools all test epochs.

## Native configuration (exact pipeline, from the code)

Paths are relative to `upstream/analysis_code/`. The order of execution is
`run_pipeline.py:42-71`: `preprocessing_main.py` → `decomposition_main.py` →
`supervised_scoring.py`. The later steps (`component_statistics.py`, `hmm_main.py`) are not
part of the supervised baseline. Settings come from `parameters.yaml`.

### Data reading and channels

| Item | Code | Source |
|---|---|---|
| Subject discovery | `glob(f"{DATA_DIR}/*/*/*.txt")`, keeping stems that start with `EPCTL`. This expects `DATA_DIR/EPCTLxx/EPCTLxx/EPCTLxx.txt` with the EDF next to it as `.edf`. | `preprocessing_main.py:44-49`; `preprocessing_utils.py:111` |
| EDF read | `mne.io.read_raw_edf(...)`, then channel names `.replace("-Ref","").upper()` | `preprocessing_utils.py:111-112` |
| EEG channels | every electrode in `Co-registered average positions.pos` is typed `eeg` (upper-cased): all **83** electrodes, including SO1/SO2 (supra-orbital) and ZY1/ZY2 (zygomatic). Every EDF channel not in the mapping keeps MNE's EDF default type `eeg`. In ANPHY all 93 EDF channels are either the 83 positions-file electrodes or one of the 10 listed non-EEG channels, so exactly 83 are EEG (inferred; checked against EPCTL03's EDF header: 1000 Hz, 93 channels, every positions-file name present). | `preprocessing_main.py:64-76`; `parameters.yaml:12-24` |
| Non-EEG channels | EMG: `CHEMG1, CHEMG2, RLEG-, RLEG+, LLEG-, LLEG+`; EOG: `EOG2, EOG1`; ECG: `ECG2, ECG1` | `parameters.yaml:12-24` |
| Excluded channels | none. All 83 EEG are used (`select_channels: None`). The 6-electrode variant sets `["F3","F4","C3","C4","O1","O2"]`, which restricts the EEG feature columns (non-EEG features are kept). | `parameters.yaml:5`; `preprocessing_main.py:120-123`; `decomposition_utils.py:69-74` |
| Montage | positions file → `make_dig_montage(coord_frame="head")`. Only used for plots and for `set_montage`. | `preprocessing_main.py:68-83`; `preprocessing_utils.py:114` |

### Filtering and referencing

All filters use `Raw.filter` with MNE defaults (inferred from the MNE 1.8 API): FIR, `firwin`,
Hamming window, zero phase, `'auto'` transition bandwidth and length. The high-pass and
low-pass are applied as **separate** calls, in this order:

| Step | Code | Source |
|---|---|---|
| 1. High-pass | EEG + ECG 0.5 Hz; EOG 0.3 Hz; EMG 10 Hz | `preprocessing_utils.py:117-119` |
| 2. EEG reference | `set_eeg_reference("average", projection=False)` over the 83 EEG (applied after the high-pass, before the low-pass) | `preprocessing_utils.py:122` |
| 3. Bipolar derivations | `RLEG = RLEG+ − RLEG-`, `LLEG = LLEG+ − LLEG-`, `CHEMG = CHEMG1 − CHEMG2`, `ECG = ECG1 − ECG2` (references dropped). Then `EOGR = EOG1 − EOG2`, keeping EOG1 and EOG2. | `preprocessing_utils.py:125-134` |
| 4. Low-pass | EEG 40 Hz; EOG 15 Hz; EMG 50 Hz; ECG 20 Hz | `preprocessing_utils.py:137-140` |
| Effective EEG band (inferred, MNE `'auto'` rules) | high-pass 0.5 Hz with a 0.5 Hz transition (−6 dB at 0.25 Hz); low-pass 40 Hz with a 10 Hz transition (−6 dB at 45 Hz). The filters act on the whole continuous night. There is **no notch** and **no resampling**: features are computed at 1 kHz. | — |
| Cache | the filtered raw is saved as `<subject>/preprocessed/<subject>_raw.fif` and reused | `preprocessing_utils.py:102-108,143` |
| Signal normalisation | none. Means and SDs are computed, but the z-scoring of the signal is commented out. | `preprocessing_utils.py:150-154` |

### Epoching and hypnogram

| Item | Code | Source |
|---|---|---|
| Epochs | `make_fixed_length_events(raw, duration=30, overlap=0)` from the EDF start; `mne.Epochs(tmin=0, tmax=30, baseline=None, detrend=None)`. This gives 30 001 samples, because MNE includes the endpoint. Other `mne.Epochs` defaults apply, including `reject_by_annotation=True` (inferred). | `preprocessing_utils.py:157-160`; `parameters.yaml:6-7` |
| Hypnogram | tab-separated `stage, time_from_onset, epoch_length`; map `W 0, N1 1, N2 2, N3 3, R 4, L −1` | `preprocessing_utils.py:43-52` |
| Feature-to-label join | feature row *j* is stamped `meas_date + 30·j s`, hypnogram row is stamped `meas_date + time_from_onset`, then an inner join on time. ANPHY hypnogram onsets are multiples of 30 s from 0 (inferred, checked on all 28 local files). | `preprocessing_main.py:200-234` |
| `L` (lights on) epochs | features are extracted for them and saved, then **dropped** before normalisation: `stage ∈ {W, N1, N2, N3, R}`. They therefore do not enter the per-subject statistics, PCA, training or scoring. No artefact rejection is applied: every non-`L` epoch is kept. MS Suppl. Table S3 counts match the number of non-`L` rows (e.g. EPCTL03: 843 rows − 82 `L` = 761). | `decomposition_utils.py:77-80`; `parameters.yaml:56-61` |

### Features (per 30 s epoch)

**EEG PSD.** `Epochs.compute_psd(method="welch", n_fft=min(4·sfreq, n_samples)=4000,
fmin=0.5, fmax=40, remove_dc=False, average="mean", picks="eeg")`
(`preprocessing_main.py:108-135`). MNE Welch defaults (inferred) are `n_per_seg = n_fft = 4000`
(4 s), `n_overlap = 0`, Hamming window and density scaling. This gives 0.25 Hz bins from 0.5 to
40 Hz (159 bins) and 7 segments per epoch: samples 0–27 999, so the last ~2 s are not used.

Per EEG channel there are **20 features** (`extract_spectral_features`,
`preprocessing_utils.py:165-243`):

| Feature | Exact definition | Source |
|---|---|---|
| `root_total_power` | `sqrt(Σ_bins PSD)` over 0.5–40 Hz. This is the bin sum, not an integral, and it is **not log**. | `:187, :194-199` |
| `spectral_entropy` | `scipy.stats.entropy(p)` with `p = PSD/Σ PSD`: Shannon entropy in nats, not normalised by log N | `:188, :200-205` |
| `spectral_slope`, `spectral_intercept` | `L = ln(p · e / min(p))` (log relative PSD shifted so that its minimum is 1), then `Ln = L / Σ L`. `LinearRegression().fit(ln f, Ln)` gives the slope and intercept. This is a single straight-line fit over 0.5–40 Hz on a sum-normalised log spectrum. It is **not** FOOOF or specparam, and **not** a plain log-log fit of the PSD (inferred, from the code). | `:189-192, :207-223` |
| `<band>_power` ×8 | `Σ p` over bins with `fmin ≤ f ≤ fmax` (inclusive, so overlapping bands share bins): low_delta 0.5–1.5, high_delta 1–4, theta 4–8, alpha 8–12, low_sigma 10–13, high_sigma 12–16, beta 15–25, gamma 25–40 Hz | `:225-232`; `parameters.yaml:124-132` |
| `log_<band>_power` ×8 | `Σ Ln` over the same bins | `:234-241` |

That gives 83 × 20 = **1660 EEG features**. Non-EEG features:

| Modality | Features | Definition | Source |
|---|---|---|---|
| EMG | `RLEG_power`, `LLEG_power`, `CHEMG_power` | Welch as above (n_fft 4000) over 10–50 Hz; `Σ PSD` (absolute, bin sum) | `preprocessing_main.py:155-170`; `preprocessing_utils.py:314-338` |
| EOG | `EOG_vertical_left` (EOG1), `EOG_horizontal_right` (EOG2), `EOG_r_l_diff` (EOGR = EOG1 − EOG2) | Welch over 0.3–2 Hz; `Σ PSD` (absolute) | `preprocessing_main.py:172-187`; `preprocessing_utils.py:246-273` |
| ECG | `ECG_heart_rate`, `ECG_SDNN`, `ECG_rmssd` | on the bipolar ECG of the epoch: `s = diff(ecg)²`; `scipy.signal.find_peaks(s, height=mean(s), distance=int(60/220·fs)=272)`; RR = diff(peaks)/fs; HR = 60/mean(RR); **SDNN = std(RR/mean RR)** and **RMSSD = sqrt(mean(diff(RR/mean RR)²))**, both of the mean-normalised RR (dimensionless) | `preprocessing_main.py:191-198`; `preprocessing_utils.py:276-311` |

Total: **1669 features** per epoch (inferred count: 1660 + 3 + 3 + 3).

### Normalisation, PCA, classifier, CV

| Step | Exact behaviour | Source |
|---|---|---|
| Per-subject z-score | for each night, after dropping `L`: `(x − mean)/std`, where the mean and std are taken over all of that night's valid epochs (`np.std`, ddof = 0) | `decomposition_utils.py:93,97,106-108` |
| Missing values | `interpolate("linear", axis=0)` along the night (two-sided in time). Leading NaNs would stay NaN. | `decomposition_utils.py:119-121` |
| Clipping | `clip(−5, 5)`, applied **after** interpolation | `decomposition_utils.py:122` |
| Pooled re-standardisation | concatenate all nights, then `(x − mean)/std` over **all subjects' epochs** (pandas, ddof = 1). The paper does not mention this step. | `decomposition_main.py:89`; `decomposition_utils.py:380-384` |
| PCA | `PCA(n_components=0.90, whiten=False, random_state=42)` fit **and** transformed on the pooled matrix of **all 29 subjects, including every LOSO test subject**. This is transductive: the test subject's features shape the components and the 90 % cut (63 PCs). It is label-free. | `decomposition_main.py:107-127` (`ex=0.90`); `decomposition_utils.py:387-390` |
| PC sign flip | each PC's sign is set so that it correlates positively with the hypnogram coded W 5, R 4, N1 3, N2 2, N3 1, using the labels of all subjects. This has no effect on a linear SVM. | `decomposition_utils.py:408-418`; `parameters.yaml:109-114` |
| PCs used by the SVM | `pca0…pca{k−1}` for k = 1…`max_comp` (63). The headline result uses all 63. | `supervised_scoring.py:93-94,237-238` |
| PC standardisation | **none in the code.** The PC scores go to the SVM raw: zero mean, variance equal to the eigenvalue. | `supervised_scoring.py:240-247` |
| Classifier | `LinearSVC(class_weight="balanced")` with all other defaults. Under the pinned scikit-learn 1.5.2: `C=1.0, penalty="l2", loss="squared_hinge", dual="auto"` (→ primal, since n_samples > n_features), `tol=1e-4, multi_class="ovr", fit_intercept=True, intercept_scaling=1, max_iter=1000, random_state=None`. With scikit-learn 1.3–1.4, which `requirements.txt` allows, `dual` defaults to `True` (inferred from the scikit-learn API). | `supervised_scoring.py:246-247`; `requirements_1.txt:97` |
| Labels | the stage strings `W/N1/N2/N3/R` | `supervised_scoring.py:85,241,244` |
| CV | leave-one-subject-out over the subjects in `all_pc_data.parquet`: train on the other 28, test on 1. No validation set, no hyperparameter search. | `supervised_scoring.py:100-102,240-247` |
| Metrics | per test subject: `cohen_kappa_score`, `accuracy_score`, per-stage precision, recall and F1 (`labels=[W,R,N1,N2,N3]`, `average=None`). The reported value is the mean over subjects (`describe()` / seaborn mean with a 95 % CI). No macro-F1 is computed. | `supervised_scoring.py:253-281,323,329-364` |
| Cached-metrics bug | it checks for `pca_metrics_df_.csv` but writes `pca_metrics_df.csv`, so the cache is never reused (harmless) | `supervised_scoring.py:96-98,288` |
| YASA benchmark | from the preprocessed fif (average-referenced, 0.5–40 Hz): `C4-M1 = C4 − TP9`, `LOC-M2 = EOG1 − TP10`, `EMG1-EMG2 = LLEG − RLEG` (the **leg** bipolar, not chin). `raw.resample(100)`; `yasa.SleepStaging(..., metadata=dict(age, male))`; yasa 0.6.5. Each subject is scored the same way as the SVM. | `supervised_scoring.py:136-187`; `requirements_1.txt:127` |

### Discrepancies between paper (MS) and code

1. **PC standardisation.** MS says "The PCA features were standardised before training". The
   code passes raw PC scores to `LinearSVC` (`supervised_scoring.py:240-247`). The features
   are standardised *before PCA* (`decomposition_utils.py:384`), which is probably what is
   meant (inferred).
2. **Total power.** MS says slope, intercept, entropy and total power were "calculated using
   log-transformed spectral power". In the code, `root_total_power` is `sqrt(Σ PSD)` (not log)
   and entropy uses the linear relative PSD. Only slope, intercept and the extra `log_<band>`
   features use the log spectrum (`preprocessing_utils.py:187-241`).
3. **Band features.** MS lists 8 relative band powers. The code also adds 8 `log_<band>_power`
   features per channel, for 20 features per channel in total (`preprocessing_utils.py:234-241`).
4. **Aperiodic fit.** MS says "linear fits on log-log PSD". The code regresses a shifted,
   sum-normalised log relative PSD on ln f (`preprocessing_utils.py:189-223`). This is not a
   standard aperiodic exponent and has no knee or peak removal.
5. **EOG.** MS describes one derived left−right channel and "total spectral power between 0.3
   and 2 Hz". The code uses three EOG features: EOG1, EOG2 and EOGR = EOG1 − EOG2
   (`preprocessing_utils.py:132-134, 261-269`).
6. **SDNN / RMSSD.** These are computed on RR intervals divided by their mean, so they are
   dimensionless coefficients rather than ms (`preprocessing_utils.py:304-309`). The MS frames
   them as "relative fluctuations", which agrees.
7. **YASA EMG.** MS says "EMG (chin)". The code feeds `LLEG − RLEG`, the leg EMG
   (`supervised_scoring.py:137-142`). The YASA inputs are also average-referenced and low-passed
   at 40 Hz, and C4-M1 is approximated by C4 − TP9 (inferred).
8. **Extra pooled standardisation** before PCA (`decomposition_utils.py:382-384`) is not
   described.
9. **Filters.** MS says "band-pass filtered". The code applies a separate high-pass and
   low-pass, with the EEG average reference taken between them (`preprocessing_utils.py:117-140`).
10. **PCA scope.** MS states that PCA was "pooled across all subjects". The code confirms that
    PCA (and the pooled standardisation) is fit on all subjects, so it includes each LOSO test
    subject. The SVM itself never sees test labels.
11. **Dependency pins.** `requirements.txt` (loose: `mne>=1.5.1`, `scikit-learn>=1.3.0`, …)
    differs from `requirements_1.txt` (full pins). The ReadMe installs from `requirements.txt`
    with Python 3.11.10. `requirements.txt` omits `yasa`, `python-picard`, `plotly` and
    `pyarrow`, which the code imports or needs (`preprocessing_utils.py:16`,
    `decomposition_utils.py:16-24`, `.to_parquet`). The `__pycache__` files are committed and
    were built by CPython 3.11.

### Data-layout differences from our copy (inferred)

- Upstream expects `DATA_DIR/EPCTLxx/EPCTLxx/EPCTLxx.{txt,edf}`. Our copy
  (`data/anphy/ANPHY-Sleep/osfstorage`) is `EPCTLxx/EPCTLxx.{edf,txt}`, and two nights differ:
  EPCTL01 is `EPCTL01 - fixed.edf` + `test1.txt`, and EPCTL02 is `EPCTL02.edf` + `test2.txt`
  (CR-only line endings). Upstream's glob would therefore skip EPCTL01 and EPCTL02. An adapter
  should take its file list from our `data.find_subjects` instead of the upstream glob.
- The positions file has an index column, so it has 5 tab-separated fields. Upstream reads it
  with 4 names; pandas then uses the first field as the index, so `electrode` is the name as
  intended.
- `mne.Epochs` defaults to `reject_by_annotation=True`. If an EDF annotation started with
  `bad`/`edge`, epochs would be dropped silently, and feature row *j* would no longer be epoch
  *j*, which misaligns the time join (`preprocessing_main.py:225-229`). EPCTL03's 14
  annotations are timestamps, `Segment: REC START 84 EEG-PSG`, `A1+A2 OFF` and body positions
  (`supine`, `left side`, …), with none starting with `bad`/`edge`. So nothing is dropped
  there. Only EPCTL03 was checked: reading the EDF+ annotations requires scanning each 4.5 GB
  file. An adapter that cuts epochs by hypnogram onset avoids the issue.

### Requirements

The authors' pinned environment (`upstream/requirements_1.txt`) runs on Python 3.11.10 (ReadMe).
The packages the supervised path needs are:

| Package | Pinned | Used for |
|---|---|---|
| mne | 1.8.0 (`:60`) | EDF reading, filters, epoching, Welch |
| numpy | 1.26.4 (`:66`) | |
| scipy | 1.14.1 (`:98`) | `stats.entropy`, `signal.find_peaks` |
| scikit-learn | 1.5.2 (`:97`) | `PCA`, `LinearSVC`, `LinearRegression`, metrics |
| pandas | 2.2.3 (`:68`) | features and z-scoring |
| pyarrow | 17.0.0 (`:82`) | parquet intermediates |
| joblib | 1.4.2 (`:43`) | parallel feature extraction |
| PyYAML | 6.0.2 (`:92`) | `parameters.yaml` |
| tqdm | 4.66.5 (`:117`) | |
| yasa | 0.6.5 (`:127`) | imported at module level by `preprocessing_utils.py:16` and `supervised_scoring.py:17`; the YASA benchmark only |
| python-picard | 0.7 (`:90`) | ICA, imported by `decomposition_utils.py:24`; not needed for the SVM |

Our `.venv` has mne 1.13.2, scikit-learn 1.9.1, scipy 1.17.1 and numpy 2.4.6, and no yasa.
The functions this baseline needs (Welch, `entropy`, `find_peaks`, `LinearRegression`, `PCA`,
`LinearSVC` with `dual="auto"`) exist there with the same defaults (checked by reading the
installed sources: `psd_array_welch` defaults `n_overlap=0, n_per_seg=None, window="hamming"`;
`LinearSVC` defaults `dual="auto", C=1.0, tol=1e-4, max_iter=1000`). The recommended
re-implementation therefore needs **no new dependency**. Importing upstream modules directly
would pull in yasa and picard, so the plan is to re-implement the
~60 lines of feature code with citations, not to import them (inferred recommendation).

## Native setting on our split (train 18 / val 5 / test 5)

The split is the reference run's fold, written by `baselines.common.reference.write_fold`.
The recipe follows the authors' pipeline step for step. Only the fitting scope changes from
"all subjects" to "training subjects".

1. **Signals and features: as upstream.** Read each EDF with MNE, apply the high-pass, average
   reference, bipolar derivations and low-pass in the upstream order, then compute the 30 s
   epochs, Welch spectra and all 1669 features (EEG + EMG + EOG + ECG). Use our hypnogram
   reader for the file list.
   - Epochs can be cut as `[onset, onset + 30 s)` (30 000 samples) instead of MNE's 30 001.
     The Welch PSDs are identical either way: 4000-sample segments with no overlap use only
     samples 0–27 999 in both cases (inferred). The ECG features differ by at most one sample.
   - This lets every scored hypnogram row inside the recording get a feature row, which our
     scorer requires.
2. **Per-subject z-scoring: unchanged.** It uses only the night's own feature statistics (no
   stage labels; the `L` mask comes from the scorer's lights marks), so it is **not** a
   cross-subject leak. The split question is moot for this step. It is still transductive
   *within* the test night, since it uses whole-night statistics including later epochs. That
   is part of the native method and is kept. Interpolation and clipping are as upstream.
3. **Pooled standardisation and PCA: fit on the 18 training subjects only.** Use the training
   subjects' valid epochs (ddof = 1 for the standardisation; `PCA(n_components=0.90,
   random_state=42)`), then transform val and test with the training mean, SD and components.
   *K*, the number of PCs reaching 90 % of training variance, will not be exactly 63 (18
   subjects instead of 29); record it. The sign flip is irrelevant for the SVM and is skipped.
4. **Classifier as in the code.** `LinearSVC(class_weight="balanced", C=1.0, penalty="l2",
   loss="squared_hinge", dual="auto", tol=1e-4, max_iter=1000, fit_intercept=True)` on the raw
   PC scores. All values go in the YAML (`configs` convention).
   - Native C is the default **1.0**. There is no tuned hyperparameter.
   - Optional disclosed variant: z-score the PCs with training statistics before the SVM. This
     follows the MS wording (discrepancy 1). Report it as a separate row, never as the main
     number.
   - Also report *k* = 5, the paper's second headline, as a secondary row.
5. **Validation subjects are not needed.** There is nothing to select: C is fixed, *K* comes
   from the 90 % rule and there is no early stopping. Train on the 18 training subjects only,
   the same training set every baseline and the reference model use. Report validation metrics
   only as a sanity check. Do not add val to training: that would give this baseline 23
   training subjects while the others have 18.
6. **Scoring.** Predict every scored test epoch keyed by `(subject, onset)` and score with
   `baselines.common.scoring.score_run`.
   - That function pools the test epochs, reports κ and macro-F1 first, and adds per-subject
     values. Also report the paper-style mean over test subjects of per-subject κ and accuracy,
     so the numbers can be set against 0.762.
   - `LinearSVC` has no probabilities. Save a softmax of `decision_function` as the
     `probability` field and label it as a pseudo-probability (inferred choice).
   - Parameter count: PCA `K × 1669` + scaler `2 × 1669` + SVM `5 × (K + 1)`.

Expected deviations from the paper's numbers come from three places: 18 training subjects
instead of 28, PCA fitted on training subjects only (non-transductive), and 5 test subjects
(noisy). None of them is a bug.

## Our configuration (83 EEG, 1 kHz, 0.5–45 Hz + 60 Hz notch + CAR, EEG only, 30 s, 20-epoch causal context)

This is the closest faithful version: the same feature definitions and the same linear model,
computed from **exactly the tensors our model reads**.

1. **Input.** Use our cached scored-epoch arrays from `data.load_subject_epochs`, scaled and
   clipped exactly as `data.EpochDataset.scaled` does: the per-electrode `RobustScaler` is
   fitted on the training subjects' clean scored epochs (`data.build_epoch_loaders`), then
   clipped to [−20, 20]. These are 83 EEG at 1000 Hz, 30 000 samples per epoch, never
   resampled.
2. **Features.** Use the EEG features only, defined exactly as upstream (20 per channel →
   **1660**): Welch with `n_fft = n_per_seg = 4000`, no overlap, Hamming window,
   `remove_dc=False`, 0.5–40 Hz; the same bands, entropy, slope/intercept and
   `root_total_power`.
   - The EMG, EOG and ECG features are dropped (EEG-only protocol).
   - The PSD range stays 0.5–40 Hz even though our signal passes to 45 Hz. This keeps the
     definitions identical. The 40–45 Hz region and the 60 Hz notch lie outside every feature.
3. **The robust scaler barely matters here (inferred).** The per-electrode `RobustScaler` is an
   affine map with one constant gain per electrode, the same for every subject. It affects the
   features as follows:
   - Relative band powers, log band powers, entropy, slope and intercept are all computed from
     `PSD/ΣPSD`, so they are invariant to the gain.
   - The median offset only touches the DC bin, which is excluded. Its leakage into the
     0.5 Hz bin is at the Hamming side-lobe level, because 0.5 Hz lies two 0.25 Hz bins
     from DC, at the first null of the Hamming main lobe.
   - `root_total_power` is multiplied by a per-electrode constant, which any later
     per-feature standardisation removes exactly.
   - The only real effect is the ±20 clip, which flattens high-amplitude (artefact) epochs.

   Feeding the scaled and clipped tensors keeps the input literally identical to our model's.
   The authors' per-subject *feature* z-scoring is a different operation: it removes
   between-night differences in every feature (absolute power above all), not just
   per-electrode gain.
4. **Normalisation: replace the per-night z-score with a training-fitted standardisation
   (recommended).** Our protocol fits every statistic on training subjects and predicts each
   epoch from itself and the 19 epochs before it. Whole-night per-subject z-scoring uses the
   statistics of the entire test night, including epochs after the one being scored. That
   breaks the causal premise and is a form of test-time adaptation.
   - Use a `StandardScaler`-style mean and SD fitted on the training subjects' clean scored
     epochs, apply it to all subjects, and clip to [−5, 5] as upstream does.
   - Do not interpolate along time, because two-sided interpolation is non-causal. Set any NaN
     to 0, the training mean. With EEG-only features a NaN needs a flat channel, so this should
     not occur (inferred).
   - Report the per-night z-score as a labelled, non-causal sensitivity row. It isolates how
     much the authors' normalisation contributes.
5. **PCA.** Fit on the training subjects' clean scored epochs: 90 % of variance → *K*
   (recorded; it will differ from 63 because there are no non-EEG features and only 18
   subjects).
6. **Context: recommended option (b).** The options considered were:

   | Option | What the SVM sees | Assessment |
   |---|---|---|
   | (a) single epoch | PCs of epoch *t* | Exactly the native classifier. It ignores the 19 context epochs that our model gets, so a gap would partly measure context rather than representation. **Report as a reference row.** |
   | **(b) stacked causal window** | `[PC(t−19), …, PC(t)]`, 20·K dims | The linear SVM gets the same 20-epoch causal window as our model and can learn any causal linear filter of the PC trajectories, so no hand-picked summary is needed. It stays in the authors' model family (linear SVM on PCA features). Dimensionality is ~20·K ≈ 1–1.3k against roughly 13–15k training epochs. That is fine for a primal L2 LinearSVC with C = 1. **Recommended main "ours" row.** |
   | (c) current + causal mean | `[PC(t), mean PC(t−19…t−1)]`, 2·K dims | Compact, but it fixes a summary (uniform weights) that (b) already contains as a special case. Report only if (b) overfits. |
   | (d) temporal smoothing or an HMM over SVM outputs | | This adds a second model (the authors' GHMM is unsupervised and separate from the SVM) and departs from the "feature + linear SVM" baseline. Not recommended. |

   Details for (b):
   - PCs are computed per epoch with the single-epoch PCA and then stacked. Context epochs
     include lights-on and artefact epochs, because our model's runs contain them too.
   - Training rows are the clean scored epochs of the training subjects (peak ≤ 500 µV),
     mirroring `EpochDataset` in `train` mode. Test rows are every scored test epoch.
   - For the first 19 epochs of a night, pad by repeating epoch 0's PCs (strictly causal).
     Our `EpochDataset` instead uses the run of epochs 0–19 near the night start, so up to 19
     early epochs per night (mostly lights-off/W) see slightly different context. Say so.
   - Use the same PC treatment as native (raw PCs) at every lag, and the same
     `LinearSVC(class_weight="balanced", C=1.0, …)`.
   - Validation subjects are not used for fitting. If C were ever tuned on val, it would be a
     disclosed deviation and not the main row.
   - Parameter count is `K × 1660` (PCA) + `2 × 1660` (scaler) + `5 × (20K + 1)` (SVM).
   - Controls: per AGENTS.md, compare (a) with (b), and add a shuffled-context control
     (permute the 19 previous epochs) before making any claim that the context's temporal
     order helps.
7. **Scoring.** Same as native: `score_run`, κ and macro-F1 first, accuracy second, per-subject
   values, and the per-subject-mean κ for comparison with the paper.

## Files

| Path | Content |
|---|---|
| `README.md` | this file |
| `paper.pdf` | bioRxiv v1 preprint (no SVM section, see above) |
| `upstream/` | `git clone` of miriamguen/high_density_sleep at `d3b0eec`, not run, no license |
| `upstream/manuscript_clean_arxiv_update.pdf` | revised manuscript with the SVM methods and results |
| `upstream/analysis_code/preprocessing_utils.py` | feature definitions (EEG, EMG, EOG, ECG) |
| `upstream/analysis_code/preprocessing_main.py` | channels, Welch calls, hypnogram join |
| `upstream/analysis_code/decomposition_utils.py` | per-subject z-score, clip, pooled PCA |
| `upstream/analysis_code/supervised_scoring.py` | LOSO LinearSVC and YASA benchmark |
| `upstream/analysis_code/parameters.yaml` | channels, bands, epoch length, valid stages |

## How to run and implementation decisions

```bash
.venv/bin/python -m baselines.run hypnopc native      # or: ours; add --quick for a smoke test
```

The command reads `baselines/settings.yaml` (reference run `runs/exp15_final_fold0`, fold
`baselines/folds/fold0.json`: train 18, val 5, test 5) and `baselines/hypnopc/<config>.yaml`.
It writes to `baselines/results/hypnopc/<config>/`:

- `config.yaml`;
- `predictions.npz` and `metrics.json` of the primary row;
- `rows/<row>/` for the secondary rows.

Per-night feature matrices are cached in `baselines/cache/hypnopc/<config>/`. A `--quick`
run appends `_quick` to both folders. It uses the first 2/1/1 train/val/test nights of the
fold and the first `quick_epochs` (120) epochs of each night.

### Code layout

| File | Content |
|---|---|
| `features.py` | Re-implementation of the upstream feature code, with file:line citations in each docstring: `welch` (MNE `psd_array_welch`, n_fft = n_per_seg = 4000, no overlap, Hamming, `remove_dc=False`), `spectral_features` (20 per electrode), `ecg_features`, `condition_native` (filters and references), `native_night` (EDF → 1669 features) and `ours_night` (our scaled cache → 1660 features). The upstream modules are not imported, because they need yasa and picard and the repository has no license. |
| `runner.py` | `run(spec, settings, out_dir, quick)`: per-night extraction (parallel over nights, cached), normalisation, PCA, one LinearSVC per row, scoring |
| `native.yaml`, `ours.yaml` | every value the code uses |

`spectral_features` was checked against a literal transcription of
`preprocessing_utils.py:187-241` (including the `LinearRegression` fit) on random data.
The maximum absolute difference was 4e-18.

### Decisions

**Native** (`native.yaml`):

- Features are computed exactly as upstream, from the EDFs: 83 EEG including SO1/2 and ZY1/2;
  EOG1, EOG2 and EOG1 − EOG2; bipolar RLEG, LLEG and chin EMG; bipolar ECG. Filters,
  reference, Welch and bands are the authors'.
- Epochs are cut at the hypnogram onsets that our pipeline keeps
  (`baselines.common.native.night_epochs`). This also resolves the EPCTL01/02 file names.
- The night is read and conditioned in stretches of 20 epochs with 10 s of real signal on each
  side (`signals.block_epochs`, `signals.pad_s`), so a 1 kHz night never sits in memory whole.
  Every step is linear and time-invariant, and the longest filter (EOG high-pass 0.3 Hz,
  11 s long) needs 5.5 s on each side, so the result equals whole-night filtering except
  within a few seconds of the recording's ends (inferred).
- Per-night z-score (population SD over the scored epochs), linear interpolation and clipping
  to ±5 are as upstream. `L` and other unscored rows are dropped first. A NaN left after
  interpolation (only possible before the first valid value) raises an error.
- The pooled z-score (sample SD) and `PCA(0.90, random_state=42)` are fitted on the
  **18 training nights only**.
- `LinearSVC(class_weight="balanced", C=1, l2, squared_hinge, dual="auto", tol=1e-4,
  max_iter=1000)` is trained on every scored epoch of the training nights. There is no
  artefact rejection, as upstream.
- Rows:
  - `all_components` (primary, raw PCs, as in the code);
  - `five_components`;
  - `standardised_components` (PCs z-scored on the training nights, as the paper's text says).
- Validation nights are used for nothing but a reported sanity score.

**Ours** (`ours.yaml`):

- Input is our cached 1 kHz epochs (notch, 0.5–45 Hz, CAR over 83), scaled and clipped
  by `data.EpochDataset.scaled`. The center and scale come from `our_loaders(cfg, split)`, i.e.
  `data.build_epoch_loaders` on the fold.
- EEG features only (1660). The PSD range stays 0.5–40 Hz.
- `StandardScaler` and `PCA(0.90, random_state=42)` are fitted on the training nights' clean
  scored epochs (peak ≤ 500 µV). They are applied to every epoch, including lights-on and
  artefact epochs, because those belong to the context. NaN → 0 (the training mean), then
  clip to ±5. There is no per-night normalisation and no interpolation along time, since both
  would use later epochs.
- Context: each epoch's PCs are concatenated with those of the 19 epochs before it, giving
  `20 × K` dimensions. Before the first epoch of a night, epoch 0 is repeated. Our
  `EpochDataset` instead gives the first 19 epochs of a night the run of epochs 0–19, which
  includes later epochs. The baseline is strictly causal there, so up to 19 epochs per night
  (mostly lights-off or W) see different context. The runner checks that the primary row's
  context equals the reference run's `data.context_epochs` (20).
- The same LinearSVC is trained on the clean scored training epochs and tested on every
  scored test epoch.
- Rows: `context` (primary, 20 epochs) and `single_epoch` (context 1).

**Both:**

- Test predictions cover every scored test epoch and are scored by
  `baselines.common.scoring.score_run`: pooled κ and macro-F1 first, per-subject values.
- `metrics.json` also reports, per row:
  - `per_subject_mean` (κ and accuracy computed per test night, then averaged, as the paper
    does);
  - pooled `val` scores;
  - `components`, `input_dimension`, `train_epochs`;
  - `solver_iterations` and `converged`;
  - `parameters`: z-score 2F + PCA mean F + PCA components K·F, plus 2K for standardised PCs,
    plus SVM 5·D + 5.
- `probability` in `predictions.npz` is softmax(`decision_function`), a pseudo-probability
  (LinearSVC has none).
- One deviation from upstream: `random_state: 0` is set on the LinearSVC. With
  `dual="auto"` the full runs have far more samples than dimensions, so liblinear solves the
  primal, which uses no randomness, and the result is unchanged. When there are fewer samples
  than dimensions (the quick test's 280-dimensional context row), liblinear uses the dual
  solver, which shuffles. Without a fixed seed, two identical quick runs gave different κ.

### Smoke tests (2026-11-03, then deleted)

| Configuration | Nights (train/val/test) | Time per night, 120 epochs, 4 in parallel | Total | Result |
|---|---|---|---|---|
| `native --quick` | EPCTL01, 02 / 13 / 06 | 46–117 s | 131 s cold, 6 s from cache | 1669 features, K = 18 PCs (90.3 %), 99 test epochs; all three rows ran and converged |
| `ours --quick` | same | 23–64 s | 114 s cold (including `our_loaders`), 14 s from cache | 1660 features, K = 14 (90.4 %), 99 test epochs; `context` (280 dims, 191 training epochs) hit `max_iter` (dual solver, n < D), `single_epoch` converged |

The κ values of a 2-night smoke test mean nothing and are not reported. Expected full-run
cost, extrapolating linearly from these timings at 850–1050 epochs per night (inferred):

- **native**: about 6–16 min per night. With `jobs: 6` that is about 1–1.5 h for 28 nights.
  Each worker holds about 2 GB.
- **ours**: about 3–8 min per night, so about 0.5 h, plus fitting the robust scaler in
  `our_loaders`.
