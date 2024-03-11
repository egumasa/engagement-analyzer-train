import typer
from typing import Optional, Dict, Any, Union
from pathlib import Path

from spacy.cli._util import app, Arg, Opt, parse_config_overrides, show_validation_error
from spacy.cli._util import import_code
from spacy.training.loop import train
from spacy.training.initialize import init_nlp
from spacy import util
from thinc.api import Config
import wandb

## Adding yaml functionality
import yaml
from pathlib import Path


def main(default_config: Path, 
        output_path: Path, 
        code_path: Optional[Path] = Opt(None, "--code", "-c", help="Path to Python file with additional code (registered functions) to be imported"),
        yml_path: Optional[Path] = Opt("scripts/sweeps/sweep1.yml", '--yaml', "-y", help = "Path to yaml config file for sweep")):
    def train_spacy():
        import_code(code_path)
        loaded_local_config = util.load_config(default_config)
        with wandb.init() as run:
            sweeps_config = Config(util.dot_to_dict(run.config))
            merged_config = Config(loaded_local_config).merge(sweeps_config)
            nlp = init_nlp(merged_config)
            output_path.mkdir(parents=True, exist_ok=True)
            train(nlp, output_path, use_gpu=True)

    #original code from github 
    # sweep_config = {"method": "bayes"}
    # metric = {"name": "score", "goal": "maximize"}
    # sweep_config["metric"] = metric
    # parameters_dict = {
    #     "components.trainable_transformer.model.name": {"values": ["egumasa/roberta-base-research-papers"]},
    #     # "training.optimizer.learn_rate": {"min": 0.00005, "max": 0.0001},
    #     "components.spancat.model.reducer.hidden_size": {"values": [384]},
    #     "components.spancat.model.reducer.hidden_size": {"values": [384]},
    #     "training.accumulate_gradient": {"values": [2,8]}
            
    # }
    # sweep_config["parameters"] = parameters_dict

    sweep_config = yaml.safe_load(Path(yml_path).read_text())

    sweep_id = wandb.sweep(sweep_config, project="eng_spacy_sweeps_yml")
    wandb.agent(sweep_id, train_spacy, count=20)


if __name__ == "__main__":
    typer.run(main)
