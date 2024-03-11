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



def run_test(model_paths: str = Opt('training/1_single_transformer/*', "--data", "-d", help = "data directory"),
             use_gpu: int = Opt(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU")
             ):
    files = glob.glob(model_paths)

    res = {}
    for file in files:
        # os.system(f"python scripts/eval/evaluate.py {file}/model-best data/engagement_three_test_dev.spacy -c ./scripts/custom_functions.py --output metrics/dev/dev")
        name = file.split(os.path.sep)[-1]
        # print(name)
        fold = re.search(r"_fold(\d)", name).group(1)

        model = os.path.join(file, "model-best")
        print(model)
        import_code("./scripts/custom_functions.py")
        print(f"##### Running evaluation for {name} ############")

        print("===== dev set =======")
        dev = evaluate_spancat(model = model,
                        data_path = Path(f"data/engagement_three_test_dev{fold}.spacy"),
                        output = Path(f"metrics/sweep/dev{fold}"),
                        use_gpu=use_gpu, 
                        return_res = True)
        res[name + "dev"] = dev

        print("===== test set =======")
        test = evaluate_spancat(model = model,
                        data_path = Path(f"data/engagement_three_test_test{fold}.spacy"),
                        output = Path(f"metrics/sweep/test{fold}"),
                        use_gpu=use_gpu, 
                        return_res = True)
        res[name + "test"] = test
        print()
    
    save_name = model_paths.split(os.path.sep)[-2]
    with open(f'metrics/sweep/{save_name}test_summary.json', 'w') as jfile:
        json.dump(res, jfile, indent = 2)


if __name__ == "__main__":
    typer.run(run_test)

