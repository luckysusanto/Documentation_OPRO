# Transfering Knowledge of 2 Different Dataset
Answers this 3 portion of RQ2:
- [ ] Specifically: Is there negative-transfer when we try finetuning old hate speech detection models on new data?
- [ ] If negative-transfer exist, to what degree? (i.e., does adding more data fix it?)
- [ ] Does finetuning above an already pretrained model create a better model than from scratch?

Assumptions:
1. Practitioner has an already pretrained model
2. Practitioner now has new data/need to update model
3. Concern: if data is not enough, negative transfer might happen instead of transfer learning.

Experiment list:
- [ ] Using IndoBERTweet trained on data A's portion-100 data, how well does it do when further finetuned on data B's portion-xxx data, as xxx increases from 010 to 100?
- [ ] Requires another hyperparameter tuning for futher finetuning.

Notes:
1. config.py generally follows config.py in folder 1_RQ1_IndoBERTweet_Baseline. Unless you know what you are doing, don't edit the section around `# DO NOT EDIT`.