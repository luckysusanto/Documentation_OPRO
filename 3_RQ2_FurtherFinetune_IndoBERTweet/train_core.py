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


def checkpoint_path(transfer_source: str, seed: int):
    return (
        C.BASE_RUNS_DIR
        / transfer_source
        / f"{transfer_source}_{C.BASE_PORTION}_{seed}"
        / "best_model"
    )


def _require_checkpoint(path, transfer_source, seed):
    if not path.is_dir():
        raise FileNotFoundError(
            f"No checkpoint at {path}\n"
            f"  Folder 3 initializes from folder 1's portion-{C.BASE_PORTION} model "
            f"for source '{transfer_source}', seed {seed}.\n"
            f"  Run folder 1 for that cell first. Note checkpoints are gitignored, "
            f"so a fresh clone will not have them."
        )


def _make_model_init(ckpt_path):
    def _model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            str(ckpt_path),
            num_labels=C.NUM_LABELS,
            local_files_only=True,   # always local: this is a path, not a hub id
        )
    return _model_init


def _training_args(hp: dict, output_dir, seed: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=C.MAX_EPOCHS,
        learning_rate=hp["learning_rate"],
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


def build_trainer(hp, target_source, portion, seed, output_dir, tokenizer):
    set_seed(seed)

    transfer_source = C.transfer_source_for(target_source)
    ckpt = checkpoint_path(transfer_source, seed)
    _require_checkpoint(ckpt, transfer_source, seed)

    train_df = load_train_portion(target_source, portion, seed)
    eval_df = load_split(target_source, "val_set")

    train_ds = to_dataset(train_df, tokenizer)
    eval_ds = to_dataset(eval_df, tokenizer)

    collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model_init=_make_model_init(ckpt),
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
    return trainer, len(train_df), transfer_source, ckpt


def evaluate_on(trainer, source, tokenizer, prefix):
    test_df = load_split(source, "test_set")
    test_ds = to_dataset(test_df, tokenizer)
    return trainer.evaluate(eval_dataset=test_ds, metric_key_prefix=prefix)


def train_once(hp, target_source, portion, seed, output_dir, tokenizer):
    trainer, n_train, transfer_source, ckpt = build_trainer(
        hp, target_source, portion, seed, output_dir, tokenizer
    )

    src_before = evaluate_on(trainer, transfer_source, tokenizer, "srcbefore")

    trainer.train()
    trainer.save_model(output_dir / "best_model")

    test_metrics = evaluate_on(trainer, target_source, tokenizer, "test")

    src_after = evaluate_on(trainer, transfer_source, tokenizer, "srcafter")

    test_metrics["_n_train"] = n_train
    test_metrics["_transfer_source"] = transfer_source
    test_metrics["_checkpoint"] = str(ckpt)
    test_metrics["_source_before"] = src_before
    test_metrics["_source_after"] = src_after
    test_metrics["_forgetting"] = (
        src_before.get(f"srcbefore_{C.EVAL_METRIC.replace('eval_', '')}", float("nan"))
        - src_after.get(f"srcafter_{C.EVAL_METRIC.replace('eval_', '')}", float("nan"))
    )

    history = trainer.state.log_history
    return test_metrics, history, n_train


# ------------------------------------------------------------------------------
# Hyperparameter search (continued-fine-tuning setting)
# ------------------------------------------------------------------------------
def run_hp_search(target_source, tokenizer, output_dir):
    import optuna

    transfer_source = C.transfer_source_for(target_source)
    ckpt = checkpoint_path(transfer_source, C.HP_SEARCH_SEED)
    _require_checkpoint(ckpt, transfer_source, C.HP_SEARCH_SEED)

    train_df = load_train_portion(target_source, C.HP_SEARCH_PORTION, C.HP_SEARCH_SEED)
    eval_df = load_split(target_source, "val_set")

    train_ds = to_dataset(train_df, tokenizer)
    eval_ds = to_dataset(eval_df, tokenizer)

    collator = DataCollatorWithPadding(tokenizer)

    set_seed(C.HP_SEARCH_SEED)

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", *C.HP_SPACE["learning_rate"], log=True
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
        model_init=_make_model_init(ckpt),
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
        compute_objective=objective,
    )

    hp = dict(C.DEFAULT_HP)
    hp.update(best.hyperparameters)
    return hp