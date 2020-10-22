import pandas as pd
from nltk import word_tokenize
from collections import Counter
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk


def prepare_text(documents, lang):
    if lang == "English":
        processed_documents = []
        all_tokens = []
        for i in range(len(documents)):
            document = documents[i]
            parts = []
            for part in document:
                tokenized_part = word_tokenize(part)
                case_folded_part = [word.lower() for word in tokenized_part]
                removed_punctuation_part = [word for word in case_folded_part if word.isalpha()]
                parts += [removed_punctuation_part]
                all_tokens += [word for word in removed_punctuation_part]
            processed_documents += [parts]
        frequency_counter = Counter(all_tokens)
        tokens_size = len(all_tokens)
        sorted_token_counter = frequency_counter.most_common(len(frequency_counter))
        sorted_token_ratio = [(c[0], c[1] / tokens_size) for c in sorted_token_counter]
        stop_words = [sorted_token_counter[i][0] for i in range(40)]
        r = range(40)
        y = [sorted_token_counter[i][1] for i in range(40)]
        plt.bar(r, y, color="red", align="center")
        plt.title("Stopwords Frequencies")
        plt.xticks(r, stop_words, rotation="vertical")
        plt.show()
        final_tokens = []
        stemmer = SnowballStemmer("english")
        for i in range(len(documents)):
            parts = processed_documents[i]
            for j in range(len(parts)):
                parts[j] = [word for word in parts[j] if word not in stop_words]
                parts[j] = [stemmer.stem(word) for word in parts[j]]
                final_tokens += [word for word in parts[j]]
        return final_tokens, processed_documents


english_columns = ["description", "title"]
english_df = pd.read_csv("data/ted_talks.csv", usecols=english_columns)
x = len(english_df)
english_documents = []
for i in range(x):
    title = english_df.iloc[i]["title"]
    description = english_df.iloc[i]["description"]
    english_documents += [[title, description]]
english_documents_tokens, english_structured_documents = prepare_text(english_documents, "English")
print(english_documents_tokens)
