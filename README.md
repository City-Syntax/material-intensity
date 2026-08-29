# MIs Database Probabilistic Query Model

This folder contains the current model definition, training workflow, exported artifacts, and evaluation outputs for building material intensity prediction in kg/m².

## Citation

Pei, W. and Ang, Y. Q.: A Harmonised Building Material Intensity Database for
Uncertainty-Aware Building Stock Analysis, Earth System Science Data (in review).

The archived release (data + model artefacts) is available on Zenodo:
https://doi.org/10.5281/zenodo.22137810 (this version; the concept DOI
https://doi.org/10.5281/zenodo.22125059 always resolves to the latest version).

## License

This repository is released under a **dual licence**, split by file, reflecting the licence terms of the underlying sources (see the paper, Sect. 2.1 and Table 1 for the full per-source breakdown):

| File(s) | License |
|---|---|
| `data/processed/Integrated_MI_database_add_Singapore.xlsx` (2,739 records), model code, notebooks, artifacts | [CC BY 4.0](LICENSE) |
| `data/processed/carbenmats_subset_GPL3.csv` (51 records sourced from Röck et al.'s CarbEnMats/GBDB, `ID_marked` prefix `G-`) | [GPL-3.0](LICENSE-GPL-3.0-carbenmats-subset.txt), see [NOTICE](NOTICE-carbenmats-subset.txt) |

The two database files are distributed as separate, independent files (not merged) precisely so the GPL-3.0 terms apply only to the Röck-sourced subset; together they reproduce the full 2,790-record harmonised database described in the paper. Values from Marinova et al. (CC BY-NC-ND 4.0) and Liu et al. (no publisher licence stated) were extracted and re-tabulated as reported facts under this project's own classification schema, not reused under either source's licence terms.

## Standardized Package Layout

This folder is organized for a GitHub-ready model package with a single source of truth for development:

- `prediction_model.ipynb`: primary development workflow.
- `prediction_model.py`: script export of the notebook for reproducible runs.
- `model_classes.py`: lightweight class definitions for model deserialization.
- Runtime artifacts: `model_finalquery.joblib`, `preprocessor.joblib`, `model_info.json`.
- Evaluation artifact: `evaluation_summary.csv`.
- Data inputs: `data/processed/Integrated_MI_database_add_Singapore.xlsx`, `data/processed/carbenmats_subset_GPL3.csv`, `data/processed/provenance_crosswalk/` (citation-only record-level provenance, see below). Per-source compiled extracts (`data/sources/`) and verbatim raw publisher files (`data/raw/`) are excluded from this repository (see License section) and kept locally only, for provenance.
- Split manifest: `fixed_split_manifest.csv`, `build_fixed_split_manifest.py`.

Recommended workflow:

1. Develop and validate in `prediction_model.ipynb`.
2. Export/sync script to `prediction_model.py`.
3. Regenerate artifacts and metrics from the same run.
4. Commit notebook, script, and generated artifacts together.

> Note: `prediction_model.py` is currently the source of truth for the
> released artifacts in this repository; `prediction_model.ipynb` has not
> yet been re-synced with the latest changes to the script (fixed
> ID-based split, single training-target cap). Re-run the notebook workflow
> and re-export before relying on it directly.

## Model Definition

The deployed model is a two-stage FinalQueryModel implemented in [prediction_model.py](prediction_model.py):

1. Stage 1 (ObservationModel): one calibrated XGBClassifier per material predicts the probability that a material value is recorded.
2. Stage 2 (IntensityModel): one quantile XGBRegressor plus one mean XGBRegressor per material predicts conditional intensity.
3. Stage 2 uses inverse-propensity sample weights derived from Stage 1 out-of-fold probabilities.

Quantile outputs are p05, p50, and p95.

## Input and Targets

Features (X_cols) from [model_info.json](artifacts/model_info.json):
- Construction period
- Construction period bucket
- Typology
- Primary Code
- Hybrid Structure
- Country
- Geo_macro

Materials (y_cols):
- Concrete
- Glass
- Steel
- Wood
- Brick

Archetype support is computed using:
- Construction period bucket
- Typology
- Primary Code
- Country

## Integrated MI Database Sources (DOI)

The Integrated_MI_database_add_Singapore.xlsx + carbenmats_subset_GPL3.csv dataset (see License section above) is harmonized from five source databases plus locally collected Singapore records.
Source labels use R-n, N-n, B-n, G-n, and C-n, where n is the original record index in each source.

### Record-level provenance

Per-source compiled extracts (`data/sources/`) are not included in this repository --
only the harmonised database, the GPL-3.0 Röck subset, and a lightweight provenance
crosswalk are released (see License section). To keep record-level traceability (per
the `ID_marked` field, see the manuscript's Data and methods section) without
redistributing the compiled source tables themselves,
`data/processed/provenance_crosswalk/provenance_crosswalk_ALL.csv` maps every
`ID_marked` to (a) the original source's DOI/access point so users can download it
directly themselves, and (b) how to locate that specific record within it:

- **B-n, C-n, G-n** (Heeren & Fishman; Yang, CBMICD1.0; Röck, CarbEnMats/GBDB -- all
  openly licensed, CC BY 4.0 / GPL-3.0-only): `n` is the 0-indexed row/id in the
  source's own publicly downloadable file at the given DOI.
- **R-n, N-n** (Marinova et al., CC BY-NC-ND 4.0; Liu et al., no licence stated by
  publisher): these two sources are themselves compilations of individual case studies
  drawn from many separate primary publications, with no independent row-index of
  their own; the crosswalk instead gives the original primary-literature citation
  (title, authors, journal, DOI where available) for each record.
- **S-n** (Singapore local records): primary data collected by the authors; see the
  manuscript, Sect. 2.1.

- R-n: Global construction materials database and stock analysis of residential buildings between 1970-2050
	DOI: https://doi.org/10.1016/j.jclepro.2019.119146
	License: CC BY-NC-ND 4.0 (values used here are extracted facts, re-tabulated under this project's own schema, not reused under this license's terms — see License section above)
- N-n: Spatiotemporal Characteristics of Global Building Material Intensity Revealed for Circular and Low-Carbon Construction
	DOI: https://doi.org/10.1021/acs.est.5c05684
	License: not stated by publisher (closed-access article; values used here are extracted facts, re-tabulated under this project's own schema)
- B-n: A database seed for a community-driven material intensity research platform
	DOI: https://doi.org/10.1038/s41597-019-0021-x
	License: CC BY 4.0
- G-n: Global Buildings Database Seed on Whole Life Carbon Emissions, Energy Performance, and Material Intensity (GBDB CarbEnMats)
	DOI: https://doi.org/10.21203/rs.3.rs-3373442/v1
	License: GPL-3.0 — distributed separately as data/processed/carbenmats_subset_GPL3.csv, see License section above
- C-n: CBMICD1.0: China's building material intensity coefficient dataset (1949-2015)
	DOI: https://doi.org/10.1016/j.resconrec.2020.104824
	License: CC BY 4.0
- S-n: Singapore local records, compiled directly by the authors. Material quantities for BIM-derived Singapore records were computed with an automated low-LOD BIM material-assessment tool (Pei, W., Wang, X., Yuan, P. F., and Stouffs, R.: From lifecycle material tracking to urban-scale material stock modeling, Resources, Conservation and Recycling, 226, 108659, 2026, https://doi.org/10.1016/j.resconrec.2025.108659).

Data integration includes schema alignment (feature names and units), category normalization, and source-ID tracking for provenance.

## Data Split

The train/validation/test split is a **fixed, ID-based partition** recorded in
[data/processed/fixed_split_manifest.csv](data/processed/fixed_split_manifest.csv)
(generated by [build_fixed_split_manifest.py](build_fixed_split_manifest.py)),
not a re-randomised split. Once a record is assigned to a split it keeps that
assignment across dataset updates; newly added records default to `train`.
Run `python build_fixed_split_manifest.py` after adding new records to extend
the manifest without disturbing the existing benchmark. `prediction_model.py`
reads this manifest and will raise an error if it is missing or out of date.

From [model_info.json](artifacts/model_info.json):

| Split | Rows |
|---|---:|
| Train | 1848 |
| Validation | 396 |
| Test | 396 |

Observed training rows per material:

| Material | n_observed_train |
|---|---:|
| Concrete | 1450 |
| Glass | 920 |
| Steel | 1674 |
| Wood | 1530 |
| Brick | 1259 |

## Current Evaluation Results

From [evaluation_summary.csv](artifacts/evaluation_summary.csv), on the single fixed test split above:

| Material | AUC | MAE | RMSE | R2 | CovU | CovC |
|---|---:|---:|---:|---:|---:|---:|
| Concrete | 0.922 | 442.56 | 687.85 | 0.369 | 0.818 | 0.886 |
| Glass | 0.963 | 1.30 | 2.55 | 0.171 | 0.885 | 0.922 |
| Steel | 0.957 | 27.34 | 98.76 | 0.070 | 0.848 | 0.888 |
| Wood | 0.926 | 9.98 | 18.68 | 0.697 | 0.800 | 0.925 |
| Brick | 0.946 | 250.48 | 966.50 | 0.088 | 0.872 | 0.917 |

Metric notes:
- CovU: uncalibrated empirical coverage of [p05, p95] on observed test rows.
- CovC: coverage after validation-based conformal offset calibration used for evaluation.
- These are single-split figures for quick reference. The manuscript reports
  5-fold cross-validated performance (mean $\pm$ std across folds) as the
  primary result, since a single split's MAE/RMSE for concrete, steel, and
  brick can vary substantially depending on which extreme-valued records fall
  into the test fold; see the paper's Technical Validation section for the
  cross-validated tables.

## Exported Artifacts

The training/export section in [prediction_model.py](prediction_model.py) writes:
- model_finalquery.joblib
- preprocessor.joblib
- evaluation_summary.csv
- model_info.json

Important:
- The exported model artifact is model_finalquery.joblib.
- query() in FinalQueryModel returns calibrated quantiles: the per-material conformal offset delta_m (estimated on the validation split; see `validation_offsets_delta_m` in model_info.json) is applied as `[max(p05 - delta_m, 0), p95 + delta_m]`. Run `python recalibrate_offsets.py` after retraining or updating the database to refresh delta_m without re-fitting Stage 1/Stage 2.
- Deserialization uses [model_classes.py](model_classes.py), a lightweight mirror of the inference-time methods (`predict_proba`, `predict_quantiles`, `predict_means`, `query`) in prediction_model.py's `ObservationModel`/`IntensityModel`/`FinalQueryModel`. `joblib.load("model_finalquery.joblib")` works in a fresh process because the exported model's classes are re-pointed at `model_classes` before pickling (`prediction_model.py` has no `__main__` guard, so importing it directly to resolve classes would re-run the full data-prep/tuning pipeline). If you edit the class definitions in `prediction_model.py`, keep model_classes.py's inference-time methods in sync.

## Main Files In This Folder

- [prediction_model.ipynb](prediction_model.ipynb): notebook workflow.
- [prediction_model.py](prediction_model.py): script workflow and model class definitions.
- [model_finalquery.joblib](artifacts/model_finalquery.joblib): trained FinalQueryModel artifact.
- [preprocessor.joblib](artifacts/preprocessor.joblib): fitted preprocessing pipeline.
- [model_info.json](artifacts/model_info.json): schema and split metadata.
- [evaluation_summary.csv](artifacts/evaluation_summary.csv): summary metrics.

## Environment Setup

Install dependencies from this folder:

```bash
pip install -r requirements.txt
```

For an exact, reproducible environment (pinned versions used to produce the
current `artifacts/`), use [uv](https://docs.astral.sh/uv/) with the
`pyproject.toml` / `uv.lock` in this folder instead:

```bash
uv sync
```

Run full training/export workflow:

```bash
python prediction_model.py
```