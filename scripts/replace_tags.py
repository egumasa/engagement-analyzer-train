import os
import re
import glob

# textfile = "Test_set/reviewed2/train.iob"

pairs = [
    ("ATTRIBUTE", "ATTRIBUTION"),
    ("ENDORSE", "ATTRIBUTION"),
    ("PRONOUNCE", "PROCLAIM"),
    ("CONCUR", "PROCLAIM"),
    #  ("CITATION", "SOURCES")
]

# files = glob.glob("assets/EDT_three_20230124_reduced_tags/*.iob")
# files = glob.glob("assets/5_fold_20230124_oversampled_reduced/*.iob")
files = glob.glob("assets/5_fold_20230124_reduced/**/*.iob", recursive=True)

for textfile in files:
    with open(textfile, 'r') as f:
        text = f.read()

        for orig, new in pairs:
            text = re.sub(f"-{orig}\t", f"-{new}\t", text)

        with open(textfile, 'w') as f:
            f.write(text)
