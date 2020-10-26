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
        remaining_terms = [(sorted_token_counter[i][0], sorted_token_counter[i][1]) for i in
                           range(40, len(frequency_counter))]
        r = range(40)
        y = [sorted_token_counter[i][1] for i in range(40)]
        plt.bar(r, y, color="red", align="center")
        plt.title("Stopwords Frequencies")
        plt.xticks(r, stop_words, rotation="vertical")
        plt.show()
        final_tokens = []
        stemmer = SnowballStemmer(lang.lower())
        for i in range(len(documents)):
            parts = processed_documents[i]
            for j in range(len(parts)):
                parts[j] = [word for word in parts[j] if word not in stop_words]
                parts[j] = [stemmer.stem(word) for word in parts[j]]
                final_tokens += [word for word in parts[j]]
        return final_tokens, processed_documents, remaining_terms


def positional(input_list):  # input_list is english_structured_documents
    positional_index_creation = dict()
    # positional_index = {
    #   term: {
    #      docID: {     docID is now line in excel file
    #         col: [pointers]     where col = title|description
    #      }
    #   }
    # }

    for docID in range(len(input_list)):
        for col in range(2):
            for ind in range(len(input_list[docID][col])):
                term = input_list[docID][col][ind]
                if term not in positional_index_creation.keys():  # new term
                    positional_index_creation[term] = dict()
                if docID not in positional_index_creation[term].keys():  # our term is found in new docID
                    positional_index_creation[term][docID] = dict()
                if col == 0:
                    if "title" not in positional_index_creation[term][docID].keys():  # term in this title for first time
                        positional_index_creation[term][docID]["title"] = [ind]
                    else:
                        positional_index_creation[term][docID]["title"] += [ind]

                elif col == 1:
                    if "description" not in positional_index_creation[term][docID].keys():  # term in this desc for first time
                        positional_index_creation[term][docID]["description"] = [ind]
                    else:
                        positional_index_creation[term][docID]["description"] += [ind]

    return positional_index_creation


def bigram(input_list):
    bigram_creation = dict()
    # bigram = {
    #   sub_term: [terms]
    # }

    for docID in range(len(input_list)):
        for col in range(2):
            for ind in range(len(input_list[docID][col])):
                term = input_list[docID][col][ind]
                for i in range(0, len(term), 1):
                    if i == 0:
                        sub_term = "$" + term[0]
                    elif i == len(term) - 1:
                        sub_term = term[-1] + "$"
                    else:
                        sub_term = term[i:i+2]

                    if sub_term not in bigram_creation.keys():
                        bigram_creation[sub_term] = [term]
                    elif term not in bigram_creation[sub_term]:
                        bigram_creation[sub_term] += [term]
    return bigram_creation


english_columns = ["description", "title"]
english_df = pd.read_csv("data/ted_talks.csv", usecols=english_columns)
x = len(english_df)
english_documents = []
for i in range(x):
    title = english_df.iloc[i]["title"]
    description = english_df.iloc[i]["description"]
    english_documents += [[title, description]]
english_documents_tokens, english_structured_documents, english_terms = prepare_text(english_documents, "English")

positional_index = positional(english_structured_documents)
bigram_index = bigram(english_structured_documents)
print(positional_index["creativ"])
print(bigram_index["v$"])
