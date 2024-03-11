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
    for key, list in holder_dict.items():
        print(f"{key}: \t {len(list)}")


def add_samples(holder: dict):
    base = holder['all']
    random.seed(2022)
    random.shuffle(base)

    over_sampled = []
    over_sampled += holder['ENDORSE']
    over_sampled += holder["CONCUR"]
    over_sampled += holder['PRONOUNCE']
    over_sampled += holder['ENDOPHORIC']
    over_sampled += holder["ATTRIBUTE"]
    # over_sampled += holder['JUSTIFYING']
    # over_sampled += holder["COMPARATIVE"]
    # over_sampled += holder["GOAL"]
    # over_sampled += holder["DENY"]
    # over_sampled += holder["EXPOSITORY"]

    over_sampled += holder['ENDOPHORIC']
    over_sampled += holder["ENDORSE"]
    over_sampled += holder["CONCUR"]
    over_sampled += holder['PRONOUNCE']
    # over_sampled += holder["COMPARATIVE"]
    # over_sampled += holder["GOAL"]
    # over_sampled += holder["EXPOSITORY"]
    # over_sampled += holder["ATTRIBUTE"]
    over_sampled += holder['ENDOPHORIC']
    over_sampled += holder["ENDORSE"]
    over_sampled += holder["CONCUR"]
    over_sampled += holder['PRONOUNCE']
    # over_sampled += holder["GOAL"]
    # over_sampled += holder["CONCUR"]
    # over_sampled += holder['PRONOUNCE']
    # over_sampled += holder["ENDORSE"]
    # over_sampled += holder["CONCUR"]
    over_sampled += holder['ENDOPHORIC']
    over_sampled += holder["CONCUR"]
    over_sampled += holder["ENDORSE"]
    # over_sampled += holder["COMPARATIVE"]
    over_sampled += holder['PRONOUNCE']
    # over_sampled += holder["GOAL"]

    over_sampled += holder['ENDOPHORIC']
    over_sampled += holder["CONCUR"]
    over_sampled += holder["ENDORSE"]

    # over_sampled += holder['PRONOUNCE']
    # over_sampled += holder['ENDOPHORIC']
    # over_sampled += holder["CONCUR"]
    # over_sampled += holder["ENDORSE"]

    # over_sampled += holder["CONCUR"]
    # over_sampled += holder["ENDORSE"]
    # over_sampled += holder['ENDOPHORIC']

    # over_sampled += holder["CONCUR"]
    # over_sampled += holder["ENDORSE"]
    # over_sampled += holder['ENDOPHORIC']

    random.seed(1234)
    random.shuffle(over_sampled)
    random.seed(5555)
    random.shuffle(over_sampled)
    random.seed(858)
    random.shuffle(over_sampled)
    random.seed(7436)
    random.shuffle(over_sampled)

    over_sampled += base

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

    for key, list in tag_count.items():
        print(f"{key}: \t {list}")


def make_dir(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("A directory {} has been created!".format(path))


def write_over_sampled(over_sampled, new_dir, split_name="train"):
    with open(new_dir + f"/{split_name}.iob", 'w') as outf:
        # outf.write(DOC_DELIMITER)
        for doc in over_sampled:
            outf.write(doc.strip())
            outf.write("\n\n")
            outf.write(DOC_DELIMITER)


def main():
    folder_name = "EDT_three_20230124"
    files = glob.glob(f'assets/{folder_name}/*.iob')
    # date = '20230109'
    # file = f"assets/{date}/train.iob"

    for file in files:
        c_split = os.path.split(file)[-1].replace(".iob","")
        print(f"====== {c_split} =====")

        holder = sort_docs(file)
        _print_tagcount(holder)
        over_sampled = add_samples(holder)
        print("====== Oversampled =====")

        count_B_tag(over_sampled)


        random.seed(111)
        random.shuffle(over_sampled)
        random.seed(1234)
        random.shuffle(over_sampled)
        random.seed(555)
        random.shuffle(over_sampled)
        random.seed(9494)
        random.shuffle(over_sampled)
        random.seed(687)
        random.shuffle(over_sampled)
        len(over_sampled)

        new_dir = f"assets/{folder_name}_oversampled/"
        make_dir(new_dir)
        write_over_sampled(over_sampled, new_dir, split_name=c_split)
    
    # date = '20230120_tdt_simple'
    # file = f"assets/{date}/train.iob"
    # new_dir = f"assets/{date}_oversample2/"

    # print("====== Train =====")

    # holder = sort_docs(file)
    # _print_tagcount(holder)
    # over_sampled = add_samples(holder)
    # print("====== Oversampled =====")

    # count_B_tag(over_sampled)

    # random.seed(111)
    # random.shuffle(over_sampled)
    # random.seed(1234)
    # random.shuffle(over_sampled)
    # random.seed(555)
    # random.shuffle(over_sampled)
    # random.seed(9494)
    # random.shuffle(over_sampled)
    # random.seed(687)
    # random.shuffle(over_sampled)
    # len(over_sampled)
    # make_dir(new_dir)
    # write_over_sampled(over_sampled, new_dir)

    # print("====== DEV =====")
    # file = f"assets/{date}/dev.iob"
    # new_dir = f"assets/{date}_oversample2/"

    # holder = sort_docs(file)
    # _print_tagcount(holder)
    # over_sampled = add_samples(holder)
    # print("=== Oversampled ====")
    # count_B_tag(over_sampled)

    # random.seed(111)
    # random.shuffle(over_sampled)
    # random.seed(1234)
    # random.shuffle(over_sampled)
    # len(over_sampled)
    # make_dir(new_dir)
    # write_over_sampled(over_sampled, new_dir, 'dev')



# random.seed(808)
# random.shuffle(over_sampled)


# 20221212 tentative oversample 2
# over_sampled = holder['ENDOPHORIC']
# over_sampled += holder['PRONOUNCE']
# over_sampled += holder["CONCUR"]
# over_sampled += holder["ENDORSE"]
# over_sampled += holder["COMPARATIVE"]
# over_sampled += holder["ATTRIBUTE"]
# over_sampled += holder["DENY"]
# over_sampled += holder["EXPOSITORY"]

# over_sampled += holder['ENDOPHORIC']
# over_sampled += holder["COMPARATIVE"]
# over_sampled += holder["CONCUR"]
# over_sampled += holder['PRONOUNCE']
# over_sampled += holder["ENDORSE"]
# over_sampled += holder["EXPOSITORY"]
# over_sampled += holder["ATTRIBUTE"]

# over_sampled += holder["CONCUR"]
# over_sampled += holder['PRONOUNCE']
# over_sampled += holder["ENDORSE"]
# over_sampled += holder["CONCUR"]
# over_sampled += holder['ENDOPHORIC']


main()
