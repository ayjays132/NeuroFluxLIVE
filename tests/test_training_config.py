from train.trainer import TrainingConfig


def test_training_config_defaults():
    cfg = TrainingConfig(model_name="gpt2", dataset_name="ds", train_split="train")
    assert cfg.grad_accum == 2
    assert cfg.lr == 5e-5
    assert cfg.warmup_steps == 0
    assert cfg.lr_scheduler_type == "linear"
    assert cfg.max_grad_norm == 1.0
    assert cfg.log_file is None
