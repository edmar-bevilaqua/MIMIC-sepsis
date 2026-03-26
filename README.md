# MIMIC-Sepsis

A curated benchmark dataset and machine learning framework for modeling sepsis trajectories in ICU patients using the MIMIC-IV database. This project implements a transparent preprocessing pipeline based on Sepsis-3 criteria and provides benchmark tasks for mortality prediction, length-of-stay estimation, vasopressor requirement, and septic shock onset classification.

**Paper:** [MIMIC-Sepsis: A Curated Benchmark for Modeling and Learning from Sepsis Trajectories in the ICU](https://arxiv.org/abs/2510.24500) (Huang et al., IEEE-EMBS BHI 2025)

## Citation

If you use this dataset or code in your research, please cite our paper:

```bibtex
@inproceedings{huang2025mimic,
  title={MIMIC-Sepsis: A Curated Benchmark for Modeling and Learning from Sepsis Trajectories in the ICU},
  author={Huang, Yong and Yang, Zhongqi and Rahmani, Amir M},
  booktitle={IEEE-EMBS International Conference on Biomedical and Health Informatics 2025},
  year={2025}
}
```

## Prerequisites

### 1. Obtain MIMIC-IV Access

Before using this code, you must have credentialed access to the MIMIC-IV database:

1. Create an account on [PhysioNet](https://physionet.org/)
2. Complete the required [CITI training course](https://physionet.org/about/citi-course/) for "Data or Specimens Only Research"
3. Sign the data use agreement for [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)
4. Wait for approval (typically 1-2 weeks)

### 2. Set Up a Local PostgreSQL Database with MIMIC-IV

This project queries MIMIC-IV data from a **local PostgreSQL database**. You must set up this database before running any extraction scripts.

1. **Install PostgreSQL** (version 13+ recommended):
   ```bash
   # macOS (Homebrew)
   brew install postgresql@16
   brew services start postgresql@16

   # Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib
   sudo systemctl start postgresql
   ```

2. **Create the MIMIC database:**
   ```bash
   # Create the database
   createdb mimiciv

   # Create the required schemas
   psql -d mimiciv -c "CREATE SCHEMA IF NOT EXISTS mimiciv_hosp;"
   psql -d mimiciv -c "CREATE SCHEMA IF NOT EXISTS mimiciv_icu;"
   psql -d mimiciv -c "CREATE SCHEMA IF NOT EXISTS mimiciv_derived;"
   ```

3. **Download and load MIMIC-IV data** following the official instructions:
   - Download the MIMIC-IV CSV files from [PhysioNet](https://physionet.org/content/mimiciv/3.1/)
   - Use the official [mimic-code loading scripts](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres) to import the data into PostgreSQL:
     ```bash
     git clone https://github.com/MIT-LCP/mimic-code.git
     cd mimic-code/mimic-iv/buildmimic/postgres
     # Follow the README in that directory to load the CSV files
     ```

4. **Verify your database** is set up correctly:
   ```bash
   psql -d mimiciv -c "SELECT COUNT(*) FROM mimiciv_icu.icustays;"
   ```
   You should see a row count (approximately 76,000+ stays for MIMIC-IV v3.1).

### 3. Configure Database Connection

The extraction scripts connect to PostgreSQL using `psycopg2`. You may need to update the connection parameters in each extraction script (e.g., `host`, `dbname`, `user`, `password`) to match your local setup. Look for lines like:

```python
conn = psycopg2.connect(dbname='mimiciv', user='your_user', password='your_password', host='localhost')
```

### 4. Install Python Dependencies

```bash
# Python 3.8+ required
pip install pandas numpy torch scikit-learn matplotlib seaborn psycopg2-binary
```

Or create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install pandas numpy torch scikit-learn matplotlib seaborn psycopg2-binary
```

## Pipeline Overview

The complete pipeline has **4 stages** that must be run in order:

```
┌─────────────────────────┐
│ Stage 1: Data Extraction │  (SQL queries → CSV files)
│   PostgreSQL / MIMIC-IV  │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Stage 2: Preprocessing   │  (Merge, clean, identify sepsis onset)
│   init_preprocess.py     │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Stage 3: Trajectory      │  (4-hour windows, SOFA/SIRS scores)
│   format_traj.py         │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Stage 4: Benchmarking    │  (Train & evaluate models)
│   benchmark.py           │
└─────────────────────────┘
```

## Usage

### Stage 1: Data Extraction

Each script below queries the PostgreSQL MIMIC-IV database and outputs a CSV file. Run them all (order does not matter within this stage):

```bash
# Core patient data
python src/icustays.py        # → ICU stay records
python src/demog.py           # → Demographics & comorbidities

# Clinical measurements
python src/ce.py              # → Vital signs (heart rate, BP, temp, etc.)
python src/lab_ce.py          # → Lab values from chartevents
python src/lab_le.py          # → Lab values from labevents

# Infection indicators
python src/abx.py             # → Antibiotic prescriptions
python src/culture.py         # → Culture events (blood, urine, CSF)
python src/microbio.py        # → Microbiology results

# Treatment & interventions
python src/mechvent.py        # → Mechanical ventilation status
python src/vaso.py            # → Vasopressor doses (standardized to norepinephrine-equivalent)
python src/fluid.py           # → IV fluid intake (tonicity-corrected)
python src/preadm_fluid.py    # → Pre-admission fluid records
python src/uo.py              # → Urine output
```

**Output:** CSV files are saved to the working directory (one per script).

### Stage 2: Initial Preprocessing

```bash
python src/init_preprocess.py
```

This script:
- Merges microbiology and culture data
- Processes demographics and detects readmissions
- Resolves missing ICU stay IDs using temporal proximity (±48h window)
- Identifies infection onset using **Sepsis-3 criteria** (antibiotic within 24h before positive culture, or culture within 72h before antibiotic)

### Stage 3: Format Patient Trajectories

```bash
python src/format_traj.py
```

This script:
- Creates fixed 4-hour timestep windows spanning 24h before to 72h after infection onset
- Aggregates multiple measurements within each window (mean values)
- Computes derived clinical scores: SOFA, SIRS, P/F ratio, shock index
- Flags sepsis and septic shock status per timestep
- Outputs structured patient trajectories ready for model training

### Stage 4: Run Benchmarks

```bash
python src/benchmark.py
```

The benchmark script trains and evaluates three model architectures across four prediction tasks:

**Models:**
| Model | Description |
|-------|-------------|
| Linear | Logistic/Ridge regression (flattened temporal features) |
| LSTM | 2-layer LSTM with dropout (hidden dim=64) |
| Transformer | Multi-head attention encoder (8 heads, 2 layers) |

**Prediction Tasks:**
| Task | Type | Approach |
|------|------|----------|
| In-Hospital Mortality (IHM) | Binary classification | First 6 hours of trajectory |
| Length of Stay (LOS) | Regression | First 6 hours of trajectory |
| Septic Shock (SS) | Binary classification | Rolling 6h window → predict next 24h |
| Vasopressor Requirement (VR) | Binary classification | Rolling 6h window → predict next 24h |

Data is split 80/20 by patient ID to prevent data leakage.

## Repository Structure

```
src/
├── ReferenceFiles/
│   └── measurement_mappings.json   # Clinical measurement metadata
├── measurements.md                 # Clinical measurements documentation
│
├── Data Extraction (Stage 1):
│   ├── icustays.py                 # ICU stay records
│   ├── demog.py                    # Demographics & Charlson Index
│   ├── ce.py                       # Vital signs (46 measurements)
│   ├── lab_ce.py                   # Lab values from chartevents
│   ├── lab_le.py                   # Lab values from labevents
│   ├── abx.py                      # Antibiotic prescriptions (~150 drugs)
│   ├── culture.py                  # Culture events (27 item types)
│   ├── microbio.py                 # Microbiology results
│   ├── mechvent.py                 # Mechanical ventilation detection
│   ├── vaso.py                     # Vasopressors (5 drugs, standardized)
│   ├── fluid.py                    # IV fluids (tonicity-corrected)
│   ├── preadm_fluid.py             # Pre-admission fluids
│   └── uo.py                       # Urine output (13 types)
│
├── Preprocessing (Stages 2-3):
│   ├── init_preprocess.py          # Merge, clean, identify sepsis onset
│   ├── format_traj.py              # Build 4h-interval trajectories
│   └── data_processor.py           # TimeSeriesDataProcessor class
│
├── Models & Evaluation (Stage 4):
│   ├── linear_model.py             # Scikit-learn linear models
│   ├── lstm_model.py               # PyTorch LSTM
│   ├── transformer_model.py        # PyTorch Transformer encoder
│   ├── metrics.py                  # AUROC, AUPRC, RMSE, MAE, R²
│   └── benchmark.py                # Full benchmarking orchestrator
│
└── Extras:
    └── notes.py                    # Clinical notes extraction (discharge)
```

## Key Results

With treatment variables included (from the paper):

| Task | Metric | Linear | LSTM | Transformer |
|------|--------|--------|------|-------------|
| IHM  | AUROC  | 0.845  | 0.838 | **0.863** |
| IHM  | AUPRC  | 0.512  | 0.507 | **0.550** |
| LOS  | RMSE   | 13.22  | 5.23  | **5.12**  |
| LOS  | MAE    | 3.10   | 2.85  | **2.81**  |
| SS   | AUROC  | 0.881  | 0.885 | **0.925** |
| SS   | AUPRC  | 0.497  | 0.580 | **0.705** |
| VR   | AUROC  | 0.924  | 0.911 | **0.927** |
| VR   | AUPRC  | 0.892  | 0.870 | **0.903** |

Treatment variables significantly improve dynamic prediction tasks (SS, VR), especially for temporal models.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MIMIC-IV](https://physionet.org/content/mimiciv/) database (Johnson et al., 2023)
- [MIT-LCP/mimic-code](https://github.com/MIT-LCP/mimic-code) for database build scripts
