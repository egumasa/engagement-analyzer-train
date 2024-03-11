
import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM

# acad_tokenizer = AutoTokenizer.from_pretrained(
#     "egumasa/roberta-base-academic3")
# acad_model = AutoModelForMaskedLM.from_pretrained(
#     "egumasa/roberta-base-academic3")


# uw_tokenizer = AutoTokenizer.from_pretrained(
#     "egumasa/roberta-base-university-writing2")

# uw_model = AutoModelForMaskedLM.from_pretrained(
#     "egumasa/roberta-base-university-writing2")


mask_filler_base = pipeline(
    "fill-mask", model="roberta-base", use_auth_token=True, top_k = 10
)

mask_filler_acad = pipeline(
    "fill-mask", model="egumasa/roberta-base-academic3", use_auth_token=True, top_k = 10
)

mask_filler_univ_writing = pipeline(
    "fill-mask", model="egumasa/roberta-base-university-writing2", use_auth_token=True, top_k = 10
)

text1 = "The goal of this paper is to <mask>."


def print_table_fill(text):
	preds1 = mask_filler_base(text)
	preds2 = mask_filler_acad(text)
	preds3 = mask_filler_univ_writing(text)


	table = []
	for pred1, pred2, pred3 in zip(preds1, preds2, preds3):
		table.append([pred1['token_str'], pred1['score'], pred2['token_str'], pred2['score'], pred3['token_str'], pred3['score']])

	df = pd.DataFrame(table, columns = ['RoBERTa-base', 'Score', 'RoBERTa-Academic', 'Score', "RoBERTa-student-writings", "Score"], index=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
	print(df)

print_table_fill(text1)

print_table_fill("The goal of this research is to <mask>.")


# print_table_fill("Previous research <mask> that the benefits of a college education are not distributed equally across society.")
# print_table_fill("Previous research <mask> that the degree of stress was a key determinant of the adverse health consequences of discrimination.")

k = 40
mask_filler_base = pipeline(
    "fill-mask", model="roberta-base", use_auth_token=True, top_k = k
)

mask_filler_acad1 = pipeline(
    "fill-mask", model="egumasa/roberta-base-academic", use_auth_token=True, top_k = k
)
mask_filler_acad2 = pipeline(
    "fill-mask", model="egumasa/roberta-base-academic3", use_auth_token=True, top_k = k
)
mask_filler_acad3 = pipeline(
    "fill-mask", model="egumasa/roberta-base-research-papers", use_auth_token=True, top_k = k
)
mask_filler_acad4 = pipeline(
    "fill-mask", model="egumasa/roberta-base-finetuned-academic", use_auth_token=True, top_k = k
)

mask_filler_univ_writing1 = pipeline(
    "fill-mask", model="egumasa/roberta-base-university-writing", use_auth_token=True, top_k = k
)
mask_filler_univ_writing2 = pipeline(
    "fill-mask", model="egumasa/roberta-base-university-writing2", use_auth_token=True, top_k = k
)

# mask_filler_univ_writing = pipeline(
#     "fill-mask", model="egumasa/roberta-base-research-papers", use_auth_token=True, top_k = 10
# )



text1 = "The goal of this paper is to <mask>."


def print_table_compare(text, model1, model2, model3, model4, model5, model6, topk=10):
	preds1 = model1(text)
	preds2 = model2(text)
	preds3 = model3(text)
	preds4 = model4(text)
	preds5 = model5(text)
	preds6 = model6(text)


	table = []
	for pred1, pred2, pred3, pred4, pred5, pred6 in zip(preds1, preds2, preds3, preds4, preds5, preds6):
		table.append([pred1['token_str'], pred1['score'], pred2['token_str'], pred2['score'], pred3['token_str'], pred3['score'],  pred4['token_str'], pred4['score'],  pred5['token_str'], pred5['score'],  pred6['token_str'], pred6['score']])

	df = pd.DataFrame(table, columns = ['M1', 'Score', 'M2', 'Score', 'M3', 'Score', 'M4', 'Score', 'M5', 'Score', 'M6', 'Score', ], index=[k + 1 for k in range(topk)])
	print(df)

print_table_compare(text1, mask_filler_acad1, mask_filler_acad2, mask_filler_acad3, mask_filler_acad4, mask_filler_univ_writing1, mask_filler_univ_writing2, topk=k)


