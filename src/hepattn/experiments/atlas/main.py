"""Top level training script, powered by the lightning CLI."""

import os
import sys

if not sys.flags.no_site:
    os.execv(sys.executable, [sys.executable, "-S", *sys.argv])

import pathlib

from lightning.pytorch.cli import ArgsType

from hepattn.experiments.atlas.lightning_module import MPflow
from hepattn.experiments.atlas.pflow_data import PflowDataModule
from hepattn.utils.cli import CLI

config_dir = pathlib.Path(__file__).parent / "configs"


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=MPflow,
        datamodule_class=PflowDataModule,
        subclass_mode_data=True,
        args=args,
        parser_kwargs={"default_env": True, "fit": {"default_config_files": [f"{config_dir}/base.yaml"]}},
    )


if __name__ == "__main__":
    main()
