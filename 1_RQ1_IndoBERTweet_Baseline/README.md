# What needs to be done
1. Get a baseline performance of IndoBERTweet on each training pair of `([csis/indoDiscourse]_hate_[xxx]_normalized.csv, [csis/indoDiscourse]_exemplars_nonhate_[xxx]_normalized.csv)`
2. Use seeded run. Check `config.py`'s SEEDS.
3. Report mean + standard deviation per model. Use `aggregate.py`
4. Search for hyperparameter first for each dataset, run it on the full data (100) variants.

# What potentially may need to be changed
1. Have other metrics in mind? Change `data_utils.py`'s compute_metrics
2. Definitely change config.py to better the numbers you want.

# Run Guide
1. `qsub ./run_hp_search.sh` -> Get best hyperparam per source
2. `qsub ./submit_baseline.sh` -> Run all source+partition combination
3. `python aggregate.py --sources csis`
4. `python aggregate.py --sources indoDiscourse`

# To Do List
0. Hyperparameter Searching
- [ ] IndoDiscourse
- [ ] CSIS
1. IndoDiscourse
- [ ] 010
- [ ] 020
- [ ] 030
- [ ] 040
- [ ] 050
- [ ] 060
- [ ] 070
- [ ] 080
- [ ] 090
- [ ] 100
2. CSIS
- [ ] 010
- [ ] 020
- [ ] 030
- [ ] 040
- [ ] 050
- [ ] 060
- [ ] 070
- [ ] 080
- [ ] 090
- [ ] 100

### Footnotes
1. For early stopping mechanism, use the `[csis/indoDiscourse]_val_set_normalized.csv`
2. For performance report, use the `[csis/indoDiscourse]_test_set_normalized.csv`