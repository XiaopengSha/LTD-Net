"""Train the current LTD-Net model with the working-copy SGD defaults."""

from pathlib import Path
from scripts._common import make_data, make_evaluator, parser, read_config, write_json


def main():
    cli = parser(__doc__)
    cli.add_argument("--epochs", type=int)
    cli.add_argument("--skip-test", action="store_true")
    args = cli.parse_args()
    config = read_config(args)

    from lightning.pytorch import seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from anomalib.engine import Engine
    from anomalib.models import Stfpm

    seed_everything(config["seed"], workers=True)
    settings = dict(config["trainer"])
    output = Path(settings.pop("output"))
    if output.exists() and any(output.iterdir()):
        cli.error("Output directory is not empty; choose a new --output to keep prior runs")
    patience = settings.pop("patience")
    data = make_data(config)
    model = Stfpm(**config["model"], evaluator=make_evaluator(), visualizer=False)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_config.json", config)
    if config["data"]["val_split_mode"] == "same_as_test":
        print("[NOTE] Validation and test share samples. AUROC-based selection is not an independent test protocol.")
    checkpoint = ModelCheckpoint(
        dirpath=output / "weights", filename="best", monitor="image_AUROC", mode="max",
        save_top_k=1, save_last=True, auto_insert_metric_name=False,
    )
    engine = Engine(
        **settings, default_root_dir=output,
        callbacks=[EarlyStopping(monitor="train_loss", mode="min", patience=patience), checkpoint],
        logger=CSVLogger(save_dir=str(output), name="logs"),
    )
    engine.fit(model=model, datamodule=data)
    if not checkpoint.best_model_path:
        raise RuntimeError("No best checkpoint was saved; inspect validation and image_AUROC")
    print(f"Best checkpoint: {checkpoint.best_model_path}")
    if not args.skip_test:
        result = engine.test(model=model, datamodule=data, ckpt_path=checkpoint.best_model_path)
        write_json(output / "test_metrics.json", result)


if __name__ == "__main__":
    main()
