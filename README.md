# Using OPRO to Adapt Hate Speech Detection System

Hate speech is an increasingly problematic issue in modern society. However, advancement in Hate Speech Detection System still primarily relies on standard transformer-based classifiers. This process is bottlenecked by data collection and the ever shifting definition of hate.

---

## Research Question
1. We know that the definition of hate (what counts as hateful rhetorics, what counts as a slur, etc) changes rather rapidly. Can LLM detect this?
2. If the answer to (1) is yes, what does it mean for the future of hate speech detection systems? (a) How much data would we require to "update" our hate speech detection system?

## Expected Output
We expect a playbook/guide book for technical users to use.

## Dataset
Available in folder `Data/Raw`.

Processed using `Data/process_data.py`

Processed data in folder `Data/Processed`

Exemplars are split 1:1 to their corresponding percentage. They are used for comparison purposes (See below)

## Project Goal
1. See Expected Output
2. We want to report whether or not OPRO can beat the SoTA approach of simply finetuning Hate Speech Detectors (HSD) using up-to-date data. 
3. Hypothetically, it should be yes. Then, we want to see, at what data amount would transformer-based HSD finally beat OPRO?

---

## Getting Started
1. cuda version 12.2
2. python version 3.10.12
3. `pip install -r requirements.txt`

