import re
import os
import random
import glob

DOC_DELIMITER = "-DOCSTART- -X- O O\n"


def init_dict(tag, holder: dict):
    if tag not in holder:
        holder[tag] = []


def sort_docs(file: str):
    holder = {"all": []}
    with open(file, 'r') as f:
        iobs = f.read()
        docs = iobs.split(DOC_DELIMITER)

        for doc in docs:
            if len(doc) > 0:
                holder['all'].append(doc)
            # print(doc)
            search = re.findall(r"B-([A-Z]+)", doc)

            for tag in search:
                init_dict(tag, holder)
                holder[tag].append(doc)
    return holder


def _print_tagcount(holder_dict: dict):

    myKeys = list(holder_dict.keys())
    myKeys.sort()
    # print(myKeys)
    sorted_dict = {i: holder_dict[i] for i in myKeys}

    for key, n in sorted_dict.items():
        print(f"{key}: \t {len(n)}")


def add_samples(holder: dict):
    base = holder['all']
    random.seed(2022)
    random.shuffle(base)

    over_sampled = []
    # over_sampled += holder['ENDORSE']
    # over_sampled += holder["CONCUR"]
    # over_sampled += holder['PRONOUNCE']
    # over_sampled += holder['ENDOPHORIC']
    # over_sampled += holder["ATTRIBUTE"]
    # # over_sampled += holder['JUSTIFYING']
    # # over_sampled += holder["COMPARATIVE"]
    # # over_sampled += holder["GOAL"]
    # # over_sampled += holder["DENY"]
    # # over_sampled += holder["EXPOSITORY"]

    # over_sampled += holder['ENDOPHORIC']
    # over_sampled += holder["ENDORSE"]
    # over_sampled += holder["CONCUR"]
    # over_sampled += holder['PRONOUNCE']
    # # over_sampled += holder["COMPARATIVE"]
    # # over_sampled += holder["GOAL"]
    # # over_sampled += holder["EXPOSITORY"]
    # # over_sampled += holder["ATTRIBUTE"]
    # over_sampled += holder['ENDOPHORIC']
    # over_sampled += holder["ENDORSE"]
    # over_sampled += holder["CONCUR"]
    # over_sampled += holder['PRONOUNCE']
    # # over_sampled += holder["GOAL"]
    # # over_sampled += holder["CONCUR"]
    # # over_sampled += holder['PRONOUNCE']
    # # over_sampled += holder["ENDORSE"]
    # # over_sampled += holder["CONCUR"]
    # over_sampled += holder['ENDOPHORIC']
    # over_sampled += holder["CONCUR"]
    # over_sampled += holder["ENDORSE"]
    # # over_sampled += holder["COMPARATIVE"]
    # over_sampled += holder['PRONOUNCE']
    # # over_sampled += holder["GOAL"]

    # over_sampled += holder['ENDOPHORIC']
    # over_sampled += holder["CONCUR"]
    # over_sampled += holder["ENDORSE"]

    # random.seed(1234)
    # random.shuffle(over_sampled)
    # random.seed(5555)
    # random.shuffle(over_sampled)
    # random.seed(858)
    # random.shuffle(over_sampled)
    # random.seed(7436)
    # random.shuffle(over_sampled)

    # over_sampled += base

    return over_sampled


def count_B_tag(over_sampled):
    tag_count = {}

    for doc in over_sampled:
        # print(doc)
        search = re.findall(r"B-([A-Z]+)", doc)
        # print(search)
        for tag in search:
            if tag not in tag_count:
                tag_count[tag] = 0
            else:
                tag_count[tag] += 1

    myKeys = list(tag_count.keys())
    myKeys.sort()
    print(myKeys)
    sorted_dict = {i: tag_count[i] for i in myKeys}

    for key, n in sorted_dict.items():
        print(f"{key}: \t {n}")


def make_dir(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("A directory {} has been created!".format(path))


def write_over_sampled(over_sampled, new_dir, split_name="train"):
    with open(new_dir + f"/{split_name}", 'w') as outf:
        # outf.write(DOC_DELIMITER)
        for doc in over_sampled:
            outf.write(doc.strip())
            outf.write("\n\n")
            outf.write(DOC_DELIMITER)


def main():
    folder_name = "5_fold_20230124_oversampled_reduced"
    files = glob.glob(f'assets/{folder_name}/*.iob')
    # date = '20230109'
    # file = f"assets/{date}/train.iob"
    files.sort()
    for file in files:
        c_split = os.path.split(file)[-1]
        print(f"====== {c_split} =====")

        holder = sort_docs(file)
        _print_tagcount(holder)


main()
