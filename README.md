# Articles are inversely associated with inflectional case in Indo-European

This repository contains the data, models, and code used in:

Goldstein, David M. 2026. Articles are inversely associated with inflectional case in Indo-European. _Language_. 

The repository contains a fully reproducible pipeline for all analyses reported in the paper.

## Quick start

```bash
git clone https://github.com/dmgoldstein/case_articles_ie
cd case_articles_ie
```

A working CmdStan installation is additionally required and is not managed by `renv` (see below).

```r
install.packages("renv")
renv::restore()
```

```bash
Rscript run_all.R
```

## Reproducibility

This repository contains all data, models, and scripts required to reproduce the analyses reported in the manuscript. All file paths are
project-relative and resolved using `here::here()`.

### System requirements

- R (tested under version 4.6.0)
- A C++ toolchain compatible with CmdStan
- CmdStan (via the `cmdstanr` R package)
- Sufficient memory and CPU to run Bayesian models (parallel chains are enabled by default)

### Environment reproducibility

Package versions used in the published analyses are recorded in `renv.lock`.

### Repository structure

```
.
├── .Rprofile                         # Activates project-local renv environment
├── renv.lock                         # Reproducible R package environment
├── renv/
│   └── activate.R                    # renv activation script
├── run_all.R                         # Top-level analysis pipeline (run this)
├── data/
│   ├── repo_data.csv                 # Language-level outcome and predictor data
│   ├── mundlak_data.csv              # Language-to-clade assignments (model 11)
│   ├── geo_distances_aligned.rds     # Pairwise great-circle distances
│   └── ieo_fifty.nex                 # 50 posterior phylogenetic trees (NEXUS)
├── scripts/
│   ├── r/
│   │   ├── run_models.R              # Fits models 1-10 (primary analyses)
│   │   ├── run_measurement_error.R   # Fits measurement-error sensitivity grid
│   │   └── run_mundlak.R             # Fits model 11 (Mundlak clade-mean control)
│   └── stan/
│       ├── partial_mediation/        # Stan files for models 1-4
│       ├── complete_mediation/       # Stan files for models 5-8
│       ├── flexible_random_effects/  # Stan files for models 9-10
│       ├── mundlak/                  # Stan file for model 11
│       └── sensitivity/              # Stan file for measurement-error model
└── output/                           # Created on first run; not tracked by git
    ├── stan-csv/                     # Raw CmdStan CSV output
    ├── fit1.rds ... fit11.rds        # Saved CmdStanMCMC objects
    └── measurement_error/
        ├── fit_sigma_0.00.rds        # One fit per grid point
        ├── ...
        ├── summary_sigma_0.00.csv    # Parameter summaries per grid point
        ├── ...
        └── sensitivity_summary.csv   # Combined table across all grid points
```

### Reproducing the analyses

**1. Download the archived replication package:**

The exact version of the repository used in the published analyses corresponds to the tagged GitHub release `v1.0.0` of this repository.

Download the archived release or clone the tagged release directly:

```bash
git clone https://github.com/dmgoldstein/case_articles_ie
cd case_articles_ie
git checkout v1.0.0
```

**2. Install CmdStan (one-time setup):**

```r
cmdstanr::check_cmdstan_toolchain()
cmdstanr::install_cmdstan(cores = parallel::detectCores())
```

**3. Run the full analysis pipeline:**

```bash
Rscript run_all.R
```

This script executes the complete analysis pipeline, fits all models, and generates all outputs required to reproduce the reported results.

It runs three analysis scripts in sequence:

- `scripts/r/run_models.R` fits the ten primary models (models 1–10) corresponding to Table 7 of the manuscript. Each fit is saved as an  
  `.rds` object in `output/`.

- `scripts/r/run_measurement_error.R` fits the measurement-error sensitivity model across eight values of `sigma_cases`
  (0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00). Results are reported in Section S3. Per-grid fits are saved to
  `output/measurement_error/` and a combined summary table is written to `output/measurement_error/sensitivity_summary.csv`.

- `scripts/r/run_mundlak.R` fits the Mundlak-type clade-mean control model (model 11), reported in Section S4. The fit is saved to
  `output/fit11.rds`.

### Model numbering

| Number | Random effects | Mediation | Stan file |
|:------:|:--------------|:----------|:----------|
| 1  | None | Partial | `partial_mediation/repo_model1_no_gp_no_phylo.stan` |
| 2  | Phylogenetic | Partial | `partial_mediation/repo_model2_phylo.stan` |
| 3  | Spatiotemporal GP | Partial | `partial_mediation/repo_model3_stgp.stan` |
| 4  | Phylogenetic + GP | Partial | `partial_mediation/repo_model4_phylo_stgp.stan` |
| 5  | None | Complete | `complete_mediation/repo_model1_complete_mediation.stan` |
| 6  | Phylogenetic | Complete | `complete_mediation/repo_model2_complete_mediation.stan` |
| 7  | Spatiotemporal GP | Complete | `complete_mediation/repo_model3_complete_mediation.stan` |
| 8  | Phylogenetic + GP | Complete | `complete_mediation/repo_model4_complete_mediation.stan` |
| 9  | Phylogenetic + GP, separate scaling | Partial | `flexible_random_effects/repo_model4_sep_scaling.stan` |
| 10 | Phylogenetic + GP, bivariate | Partial | `flexible_random_effects/repo_model4_bivariate.stan` |
| 11 | Phylogenetic + GP, Mundlak control | Partial | `mundlak/repo_model11_mundlak.stan` |
| ME | Measurement-error sensitivity | Partial | `sensitivity/repo_model4_measurement_error.stan` |

All models are estimated using four Markov chains with 4,000 warmup and 4,000 post-warmup iterations per chain and no thinning (thin = 1),
yielding 16,000 post-warmup draws for each fit. The sampler is tuned with a target acceptance rate of $\delta$ = 0.95 and a maximum tree depth of 15 throughout. Primary models (1–11) use seed = 123; the measurement-error sensitivity grid uses seed = 2025.

### Model 11: Mundlak clade-mean control

Model 11 extends the primary model (model 4) by adding, for each language, the mean case inventory of its clade (centered at the
grand mean) as a predictor. This follows the approach of Mundlak (1978) and separates the within-clade association between case inventory and
article use from the between-clade contrast. 

The clade assignments used to compute clade means are provided in `data/mundlak_data.csv`, which maps each of the 94 IE languages to one of 12 clades. The centered clade means are computed at runtime by `scripts/r/run_mundlak.R` from the combination of `mundlak_data.csv` and `repo_data.csv`.

## License

MIT License.