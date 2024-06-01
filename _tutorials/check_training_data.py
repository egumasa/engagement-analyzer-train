#%%
import os


os.chdir('/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/SFLAnalyzer/Engagement_span_finder')
import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")

training = DocBin().from_disk("data/engagement_test_train.spacy")

docs = list(training.get_docs(nlp.vocab))
print(len(docs))

#%%
lengths = []
for doc in docs:
    # print(doc)
    spanG = doc.spans['sc']
    # print(spanG)
    for x in spanG:

        lengths.append(len(x))
        if len(x) > 15:
            print(f"{x.label_}\tSPAN: {x.text}")



#%%
dev = DocBin().from_disk("data/engagement_dev.spacy")

docs = list(dev.get_docs(nlp.vocab))
print(len(docs))

#%%
from statistics import mean, median, stdev, quantiles

mean(lengths)
median(lengths)
max(lengths)
quantiles(lengths)

sorted(lengths, reverse=True)

from collections import Counter, OrderedDict

# creating a list with the keys
item_len = dict(Counter(lengths))
print(item_len)
# %%
for n, size in sorted(item_len.items()):
    print(f"{n}: {size}")
# %%
import matplotlib.pyplot as plt

plt.bar(list(item_len.keys()), item_len.values(), color='g')
plt.show()
# %%
