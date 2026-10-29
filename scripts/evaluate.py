"""Evaluate a checkpoint with the bundled anomalib metrics and explicit data split."""

from pathlib import Path
from scripts._common import load_checkpoint, make_data, make_evaluator, parser, read_config, write_json


def main():
    cli = parser(__doc__)
    cli.set_defaults(output=Path("outputs/evaluation"))
    cli.add_argument("--checkpoint", required=True, type=Path)
    args = cli.parse_args()
    config = read_config(args)
    from lightning.pytorch import seed_everything
    from anomalib.engine import Engine

    seed_everything(config["seed"], workers=True)
    model = load_checkpoint(args.checkpoint, evaluator=make_evaluator())
    if tuple(config["model"]["image_size"]) != tuple(model.image_size):
        raise ValueError("Config image_size does not match the checkpoint")
    data = make_data(config)
    output = Path(config["trainer"]["output"])
    if output.exists() and any(output.iterdir()):
        cli.error("Output directory is not empty; choose a new --output")
    engine = Engine(
        accelerator=config["trainer"]["accelerator"], devices=1,
        # Engine injects a ModelCheckpoint callback even for test; do not pass
        # enable_checkpointing=False (Lightning rejects that combination).
        default_root_dir=output, logger=False,
    )
    result = engine.test(model=model, datamodule=data)
    write_json(output / "metrics.json", result)
    write_json(output / "evaluation_config.json", config)
    print(result)


if __name__ == "__main__":
    main()
