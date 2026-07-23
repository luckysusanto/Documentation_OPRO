import numpy as np 
import torch 

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

import config as C 
from data_utils import load_train_portion, load_split, to_dataset, compute_metrics

def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        C.MODEL_NAME_OR_PATH, local_files_only=C.LOCAL_FILES_ONLY
    )

def _model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        C.MODEL_NAME_OR_PATH, 
        num_labels=C.NUM_LABELS,
        local_files_only=C.LOCAL_FILES_ONLY,
    )

def _training_args(hp: dict, output_dir, seed: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=C.MAX_EPOCHS,
        learning_rate=hp['learning_rate'],
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        per_device_eval_batch_size=64,
        weight_decay=hp["weight_decay"],
        warmup_ratio=hp["warmup_ratio"],
        eval_strategy=C.EVAL_STEPS_STRATEGY,
        save_strategy=C.EVAL_STEPS_STRATEGY,
        load_best_model_at_end=True,
        metric_for_best_model=C.EVAL_METRIC,
        greater_is_better=C.GREATER_IS_BETTER,
        save_total_limit=1,
        logging_strategy="epoch",
        seed=seed,
        data_seed=seed,
        report_to="none",
        fp16=torch.cuda.is_available(),
        disable_tqdm=False,
    )

def build_trainer(hp, source, portion, seed, output_dir, tokenizer):
    set_seed(seed)

    train_df = load_train_portion(source, portion, seed)
    eval_df = load_split(source, "val_set")

    train_ds = to_dataset(train_df, tokenizer)
    eval_ds = to_dataset(eval_df, tokenizer)

    collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model_init=_model_init,
        args=_training_args(hp, output_dir, seed),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=C.EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=C.EARLY_STOPPING_THRESHOLD,
            )
        ],
    )
    return trainer, len(train_df)

def train_once(hp, source, portion, seed, output_dir, tokenizer):
    trainer, n_train = build_trainer(hp, source, portion, seed, output_dir, tokenizer)
    trainer.train()

    # Note, based on current RQ setup, we only need portion 100. However, I'm saving all.
    # Edit as needed future person.
    # if portion == "100": trainer.save_mode(output_dir / "best_model")
    trainer.save_model(output_dir / "best_model")

    test_df = load_split(source, 'test_set')
    test_ds = to_dataset(test_df, tokenizer)
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    history = trainer.state.log_history
    return test_metrics, history, n_train 

# Hyperparameter searching
def run_hp_search(source, tokenizer, output_dir):
    import optuna # Automate hyperparameter searching

    train_df = load_train_portion(source, C.HP_SEARCH_PORTION, C.HP_SEARCH_SEED)
    eval_df = load_split(source, "val_set")
    
    train_ds = to_dataset(train_df, tokenizer)
    eval_ds = to_dataset(eval_df, tokenizer)

    collator = DataCollatorWithPadding(tokenizer)

    set_seed(C.HP_SEARCH_SEED)

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", *C.HP_SPACE["learning_rate"], log=True # To search around 1e-5, 2e-5, ...
            ),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", C.HP_SPACE["per_device_train_batch_size"]
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", *C.HP_SPACE["weight_decay"]
            ),
            "warmup_ratio": trial.suggest_float(
                "warmup_ratio", *C.HP_SPACE["warmup_ratio"]
            ),
        }
    
    base_args = _training_args(C.DEFAULT_HP, output_dir / "hp_search", C.HP_SEARCH_SEED)

    trainer = Trainer(
        model_init=_model_init,
        args=base_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=C.EARLY_STOPPING_PATIENCE)
        ],
    )

    def objective(metrics):
        return metrics[C.EVAL_METRIC]

    best = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=C.HP_N_TRIALS,
        compute_objective=objective
    )

    hp = dict(C.DEFAULT_HP)
    hp.update(best.hyperparameters)
    return hp
