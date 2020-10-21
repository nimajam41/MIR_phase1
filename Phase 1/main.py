import pandas as pd
from nltk import word_tokenize
from collections import Counter
import matplotlib
from nltk.corpus import stopwords
import nltk

english_columns = ["description", "title"]
english_df = pd.read_csv("data/ted_talks.csv", usecols=english_columns)
x = len(english_df)
processed_titles = []
processed_descriptions = []
all_tokens = []
for i in range(x):
    title = english_df.iloc[i]["title"]
    description = english_df.iloc[i]["description"]
    title_tokenized = word_tokenize(title)
    description_tokenized = word_tokenize(description)
    case_folded_title = [word.lower() for word in title_tokenized]
    case_folded_description = [word.lower() for word in description_tokenized]
    removed_punctuation_title = [word.lower() for word in case_folded_title if word.isalpha()]
    removed_punctuation_description = [word.lower() for word in case_folded_description if word.isalpha()]
    processed_titles += [removed_punctuation_title]
    processed_descriptions += [removed_punctuation_description]
    all_tokens += [word for word in removed_punctuation_title]
    all_tokens += [word for word in removed_punctuation_description]

frequency_counter = Counter(all_tokens)
tokens_size = len(all_tokens)
sorted_token_counter = frequency_counter.most_common(len(frequency_counter))
sorted_token_ratio = [(c[0], c[1] / tokens_size) for c in sorted_token_counter]
stop_words = [sorted_token_counter[i][0] for i in range(40)]
print(tokens_size)
final_tokens = []
for i in range(len(english_df)):
    processed_titles[i] = [word for word in processed_titles[i] if word not in stop_words]
    processed_descriptions[i] = [word for word in processed_descriptions[i] if word not in stop_words]
    final_tokens += [word for word in processed_titles[i]]
    final_tokens += [word for word in processed_descriptions[i]]
print(processed_titles[0])
