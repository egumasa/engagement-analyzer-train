#%%
import spacy
from spacy import displacy

nlp = spacy.load(
    '/Users/masakieguchi/Dropbox/0_Projects/0_basenlp/0_NLPisFUN/spacy-experimental/projects/Engagement_span_finer_test/packages/en_engagement_config_trf-0.0.1/en_engagement_config_trf/en_engagement_config_trf-0.0.1'
)


#%%
def render(text: str):
    doc = nlp(text)
    displacy.serve(doc, style="span")


# %%

text = "It is no doubt that previous minister had cheated."
doc = nlp(text)

displacy.render(doc, style="span", jupyter=True)

# %%
text = "It is no doubt that previous minister had cheated."
doc = nlp(text)

displacy.serve(doc, style="span")

#%%
text = "Even though there were a lot of issues, we could perhaps succeed at the end."
doc = nlp(text)

displacy.serve(doc, style="span")

#%%

text = "As the name implies, it seemed as if eating children was one of the main focuses of anti-witch writings."
doc = nlp(text)

displacy.serve(doc, style="span")

#%%
text = "Further, however, Nider describes a man, Stadelin, who caused a woman to miscarry seven times and, along with Hoppo, caused infertility of both people and animals (157–158) reinforcing that witches threaten reproduction in general. Also linking homosexuality to witches fulfills the same purpose."
doc = nlp(text)
displacy.serve(doc, style="span")

#%%
render(
    "Hence Bernardino emphasizes the importance of everyone’s contribution to the condemnation of witches (137)."
)

#%%
render(
    "As discussants correctly pointed out, Bernardino of Siena, Martin Le Franc, and the anonymous author of the Errores Gazariorum all have an even more aggressive campaign against witches than did the authors of our previous readings."
)

#%%
render(
    "All five show that dependence is associated with greater inequality. More specifically, five studies demonstrate that investment dependence – investment by foreign firms in a society’s domestic economy increases economic inequality."
)

#%%
render(
    "This may be true, but I contend that a telephone call to a person who has been robbed takes only a couple of minutes and shows that someone cares."
)

#%%
render(
    "Sure, he broke rules. Yes, he ducked and dived. Admittedly, he was badly behaved. But look at what he achieved."
)

#%%
render(
    "His attack came as the Aboriginal women involved in the case demanded a female minister examine the religious beliefs they claim are inherent in their fight against a bridge to the island near Goolwa in South Australia."
)
#%%
span = doc.spans
for s in span['sc']:
    print(s.label_)
# %%
