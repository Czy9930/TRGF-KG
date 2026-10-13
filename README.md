# TRGF-KG

This repository provides selected core implementation scripts and typhoon observation data for TRGF-KG, corresponding to the paper:

**A Multi-Step Typhoon Track Prediction Method Integrating Spatiotemporal Trajectory Dynamics and Prior Structural Relations**

## Method Overview

TRGF-KG integrates temporal trajectory dynamics with explicit structural relations in a knowledge graph for multi-step typhoon track prediction.

The temporal branch uses a Transformer to extract block-level representations from historical observations. The relational branch uses a pretrained R-GCN to generate structural representations for the same historical observations. The two representations are aligned through typhoon identifiers, original observation indices, and shared temporal block boundaries.

The fusion module determines the relative contributions of the two representations through adaptive branch weighting and applies bounded, motion-conditioned scale and shift modulation to the relational features. The fused representation jointly predicts eastward and northward position corrections relative to a T-CLIPER reference track, which are converted into longitude and latitude forecasts.

## Repository Structure

```text
TRGF-KG/
├── README.md
├── TRGF-KG.py
├── feature_engineering_h24.py
├── extract_temporal_kg_from_neo4j.py
├── export_enriched_kg.py
├── train_rgcn_enriched.py
├── generate_enriched_causal_embeddings.py
└── input_data/
    ├── past_2000-2022typhoons/
    └── typhoon/
```

## Scripts

| File | Purpose |
| --- | --- |
| `TRGF-KG.py` | Implements temporal modeling, relational representation fusion, motion-conditioned feature modulation, two-stage training, and prediction evaluation. |
| `feature_engineering_h24.py` | Cleans observations, constructs historical input and forecast windows, splits data by year, and standardizes variables using training-set statistics. |
| `extract_temporal_kg_from_neo4j.py` | Exports the base graph structure, entity information, and observation-time mappings from an existing Neo4j database. |
| `export_enriched_kg.py` | Constructs attribute-bin relations for typhoon translation speed and the radius of force-7 winds, producing graph data for R-GCN. Bin thresholds are determined from training-period typhoons. |
| `train_rgcn_enriched.py` | Pretrains R-GCN on the graph structure associated with training-period typhoons and saves encoder parameters and normalization statistics. |
| `generate_enriched_causal_embeddings.py` | Generates historical-window graph embeddings using a frozen R-GCN encoder and checks observation-index alignment and historical-window constraints. |

## Prediction Task

| Setting | Configuration |
| --- | --- |
| Historical input | 12 observation steps |
| Forecast horizon | 24 future observation steps |
| Temporal input variables | Longitude, latitude, wind speed, and central pressure |
| Final output | Longitude and latitude of future positions |
| Training period | 2000–2018 |
| Validation period | 2019–2020 |
| Test period | 2021–2022 |

The 24-step horizon represents 24 future observation positions. Actual forecast lead times are determined by the original observation timestamps. Standardization parameters are fitted exclusively on the training set.

## Data

The `input_data/` directory contains two folders of typhoon records, each with 545 files. The current preprocessing script reads records from `past_2000-2022typhoons/`; the `typhoon/` folder contains CSV files.

The main fields include:

- `OBS`: observation index.
- `tfbh`: typhoon identifier.
- `time`: observation time.
- `lng` and `lat`: longitude and latitude.
- `speed`: wind speed.
- `pressure`: central pressure.
- `move_speed`: typhoon translation speed.
- `radius7`: radius of force-7 winds.

The original data source cited in the paper is *A dataset of typhoon tracks in offshore China*, provided by the Marine Science Data Center, Chinese Academy of Sciences.

[Dataset DOI](https://doi.org/10.12157/IOCAS.20210804.001)

## Usage Notes

This release includes the core scripts listed above and typhoon observation data. Configure the raw-data paths, processed-data paths, and model input locations for your local environment before running the scripts.

The two graph export scripts use an existing Neo4j typhoon database. Supply the database password through the `NEO4J_PASSWORD` environment variable. The main model reads processed trajectory data, pretrained graph encoder information, historical-window graph embeddings, and a T-CLIPER reference forecast file. Use the available command-line options to specify the corresponding paths.

## Main Dependencies

- Python
- PyTorch
- NumPy
- pandas
- scikit-learn
- Neo4j Python Driver
- requests
- tqdm

## Citation

Please cite the corresponding paper when using this method or code in your research. Formal publication details will be added after the paper is published.
