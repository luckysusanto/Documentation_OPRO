# Using OPRO to Adapt Hate Speech Detection System

Hate speech is an increasingly problematic issue in modern society. However, advancement in Hate Speech Detection System still primarily relies on standard transformer-based classifiers. This process is bottlenecked by data collection and the ever shifting definition of hate.

---

## Research Question
1. How competitive is OPRO compared to traditional solutions when we look at it from data point perspective?
2. Can OPRO adapt previous definition of hate speech so that it fits better with current definition of hate speech?

## Expected Output
We expect a playbook/guide book for technical users to use.

## Dataset
Available in folder `Data/Raw`.

Processed using `Data/process_data.py`

Processed data in folder `Data/Processed`

Exemplars are split 1:1 to their corresponding percentage. They are used for comparison purposes (See below)

Data to be used in the experiments are under the folder `Data/Normalized`. See `Data/normalize_data.py` for code.

## Project Goal
1. See Expected Output
2. We want to report whether or not OPRO can beat the SoTA approach of simply finetuning Hate Speech Detectors (HSD) using up-to-date data. 
3. Hypothetically, it should be yes. Then, we want to see, at what data amount would transformer-based HSD finally beat OPRO?

---

## Getting Started
1. cuda version 12.2
2. python version 3.10.12
3. `pip install -r requirements.txt`

### Preparation: Model Caching
1. Set HF_HOME `export HF_HOME=/path/to/your/cache`
2. What models are we running the experiment with? Original setup uses 5 (see the MODEL variable in `0_Setup/cache_model.py`). Change as needed with exact repo id.
3. You can resume it safely even if it fails.
4. You can run `cache_model.py` by itself. Or, if you are in a compute cluster, just `qsub cache_model.sh` (or its equivalance).
---

## Experiment List
1. Getting a baseline.
= [ ] For each model used, Get a baseline performance.