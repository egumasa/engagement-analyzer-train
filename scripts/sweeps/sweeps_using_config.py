import typer
from typing import Optional, Dict, Any, Union

from pathlib import Path

from spacy.cli._util import app, Arg, Opt, parse_config_overrides, show_validation_error
from spacy.cli._util import import_code
import spacy
from spacy.training.loop import train
from spacy.training.initialize import init_nlp
from spacy import util
from thinc.api import Config
import wandb
import os 


spacy.require_gpu()

def main(default_config: Path, output_path: Path, 
code_path: Path):
    loaded_local_config = util.load_config(default_config)
    import_code(code_path)

    with wandb.init() as run:
        name_run = run.name
        output_path2 = os.path.join(output_path, name_run)
        os.makedirs( output_path, exist_ok=True )
        sweeps_config = Config(util.dot_to_dict(run.config))
        merged_config = Config(loaded_local_config).merge(sweeps_config)
        nlp = init_nlp(merged_config)
        output_path.mkdir(parents=True, exist_ok=True)
        train(nlp, output_path2, use_gpu=0)

if __name__ == "__main__":
    typer.run(main)
