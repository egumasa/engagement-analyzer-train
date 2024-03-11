import json
import re
import glob
import os
import pandas as pd
from pathlib import Path

from eval.spancat_eval_confusion import evaluate_spancat
import spacy
from spacy.cli._util import app, Arg, Opt, setup_gpu, import_code

import typer


# spacy.prefer_gpu()


def run_test(model_paths: str = Opt('training/1_single_transformer/*', "--data", "-d", help = "data directory"),
             use_gpu: int = Opt(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU"),
             ):
    files = glob.glob(model_paths)

    res = {}
    for file in files:
        # os.system(f"python scripts/eval/evaluate.py {file}/model-best data/engagement_three_test_dev.spacy -c ./scripts/custom_functions.py --output metrics/dev/dev")
        name = file.split(os.path.sep)[-1]
        import_code("./scripts/custom_functions.py")
        print(f"##### Running evaluation for {name} ############")
        print()
        print("===== dev set =======")
        dev = evaluate_spancat(model = os.path.join(file, "model-best"),
                        data_path = Path("data/engagement_three_test_dev.spacy"),
                        output = Path("metrics/sweep/dev"),
                        use_gpu=use_gpu, 
                        return_res = True)
        res[name + "dev"] = dev

        print("===== test set =======")
        test = evaluate_spancat(model = os.path.join(file, "model-best"),
                        data_path = Path("data/engagement_three_test_test.spacy"),
                        output = Path("metrics/sweep/test"),
                        use_gpu=use_gpu, 
                        return_res = True)
        res[name + "test"] = test
        print()
    
    save_name = model_paths.split(os.path.sep)[-2]
    with open(f'metrics/sweep/{save_name}test_summary.json', 'w') as jfile:
        json.dump(res, jfile, indent = 2)


if __name__ == "__main__":
    typer.run(run_test)

