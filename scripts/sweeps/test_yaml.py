import yaml
from pathlib import Path
conf = yaml.safe_load(Path('scripts/sweeps/sweep.yml').read_text())