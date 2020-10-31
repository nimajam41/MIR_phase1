import pandas as pd
from nltk import word_tokenize
from collections import Counter
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
import pickle


def prepare_text(documents, lang, stop_words):
    if lang == "english":
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
        if len(stop_words) == 0:
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
        else:
            remaining_terms = []
        final_tokens = []
        stemmer = SnowballStemmer(lang.lower())
        for i in range(len(documents)):
            parts = processed_documents[i]
            for j in range(len(parts)):
                parts[j] = [word for word in parts[j] if word not in stop_words]
                parts[j] = [stemmer.stem(word) for word in parts[j]]
                final_tokens += [word for word in parts[j]]
        return final_tokens, processed_documents, remaining_terms, stop_words


def positional(input_list, positional_index_creation, start, end):  # input_list is english_structured_documents
    # positional_index = {
    #   term: {
    #      docID: {     docID is now line in excel file
    #         col: [pointers]     where col = title|description
    #      }
    #   }
    # }

    for docID in range(start - 1, end):
        for col in range(2):
            for ind in range(len(input_list[docID - start + 1][col])):
                term = input_list[docID - start + 1][col][ind]
                if term not in positional_index_creation.keys():  # new term
                    positional_index_creation[term] = dict()
                    positional_index_creation[term]["cf"] = dict()
                    positional_index_creation[term]["cf"] = 0
                if docID not in positional_index_creation[term].keys():  # our term is found in new docID
                    positional_index_creation[term][docID] = dict()
                positional_index_creation[term]["cf"] += 1
                if col == 0:
                    if "title" not in positional_index_creation[term][
                        docID].keys():  # term in this title for first time
                        positional_index_creation[term][docID]["title"] = [ind]
                    else:
                        positional_index_creation[term][docID]["title"] += [ind]

                elif col == 1:
                    if "description" not in positional_index_creation[term][
                        docID].keys():  # term in this desc for first time
                        positional_index_creation[term][docID]["description"] = [ind]
                    else:
                        positional_index_creation[term][docID]["description"] += [ind]


def bigram(input_list, bigram_creation, start, end):
    # bigram = {
    #   sub_term: [terms]
    # }

    for docID in range(start - 1, end):
        for col in range(2):
            for ind in range(len(input_list[docID - start + 1][col])):
                term = input_list[docID - start + 1][col][ind]
                for i in range(-1, len(term), 1):
                    if i == -1:
                        sub_term = "$" + term[0]

                    elif i == len(term) - 1:
                        sub_term = term[-1] + "$"
                    else:
                        sub_term = term[i:i + 2]

                    if sub_term not in bigram_creation.keys():
                        bigram_creation[sub_term] = [term]
                    elif term not in bigram_creation[sub_term]:
                        bigram_creation[sub_term] += [term]


def insert(documents, lang, bigram_index, positional_index):
    doc_tokens, docs_structured, doc_terms, doc_stops = prepare_text(documents, lang, stop_words_dic[lang])
    document_tokens[lang] += [word for word in doc_tokens]
    structured_documents[lang] += [doc for doc in docs_structured]
    terms[lang] += [term for term in doc_terms if term not in terms[lang]]
    bigram(docs_structured, bigram_index, docs_size[lang] + 1, docs_size[lang] + len(documents))
    positional(docs_structured, positional_index, docs_size[lang] + 1, docs_size[lang] + len(documents))
    docs_size[lang] += len(documents)
    for _ in range(len(documents)):
        deleted_documents[lang] += [False]
    return bigram_index, positional_index


def delete(documents, doc_id, bigram_index, positional_index, deleted_list):
    if doc_id >= len(deleted_list):
        print("docID (" + str(doc_id + 1) + ") doesn't exist!")
        return
    if not deleted_list[doc_id]:
        document = documents[doc_id]
        for part in document:
            for term in part:
                positional_index[term]["cf"] -= 1
                if doc_id in positional_index[term].keys():
                    del positional_index[term][doc_id]
                if positional_index[term]["cf"] == 0:
                    first = '$' + term[0]
                    last = term[len(term) - 1] + '$'
                    if term in bigram_index[first]:
                        bigram_index[first].remove(term)
                    if len(bigram_index[first]) == 0:
                        del bigram_index[first]
                    if term in bigram_index[last]:
                        bigram_index[last].remove(term)
                    if len(bigram_index[last]) == 0:
                        del bigram_index[last]
                    for i in range(0, len(term) - 1):
                        s = term[i:i + 2]
                        if term in bigram_index[s]:
                            bigram_index[s].remove(term)
                        if len(bigram_index[s]) == 0:
                            del bigram_index[s]
                    del positional_index[term]
        deleted_list[doc_id] = True
        docs_size[lang] -= 1
    else:
        print("this docID (" + str(doc_id + 1) + ") does not exist in the documents set!")


def jaccard_similarity(query, term, lang):
    query_bigrams = []
    for i in range(0, len(query) - 1):
        query_bigrams += [query[i:i + 2]]
    intersect_counter = 0
    for bichar in query_bigrams:
        if bichar in bigram_index[lang].keys() and term in bigram_index[lang][bichar]:
            intersect_counter += 1
    return intersect_counter / (len(query_bigrams) + len(term) - 1 - intersect_counter)


def correction_list(word, lang, threshold):
    word_bigrams = []
    for i in range(0, len(word) - 1):
        word_bigrams += [word[i:i + 2]]
    suggested_terms = []
    for bichar in word_bigrams:
        if bichar in bigram_index[lang].keys():
            for term in bigram_index[lang][bichar]:
                if term not in suggested_terms:
                    if jaccard_similarity(word, term, lang) > threshold:
                        suggested_terms += [term]
    return suggested_terms


def edit_distance(query, term):
    dp = [[0 for _ in range(len(term) + 1)] for _ in range(len(query) + 1)]
    for i in range(len(query) + 1):
        dp[i][0] = i
    for j in range(len(term) + 1):
        dp[0][j] = j
    for i in range(1, len(query) + 1):
        for j in range(1, len(term) + 1):
            if query[i - 1] == term[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
    return dp[len(query)][len(term)]


english_columns = ["description", "title"]
english_df = pd.read_csv("data/ted_talks.csv", usecols=english_columns)
x = len(english_df)
collections = {"english": [], "persian": []}
document_tokens = {"english": [], "persian": []}
structured_documents = {"english": [], "persian": []}
terms = {"english": [], "persian": []}
stop_words_dic = {"english": [], "persian": []}
bigram_index = {"english": dict(), "persian": dict()}
positional_index = {"english": dict(), "persian": dict()}
docs_size = {"english": 0, "persian": 0}
deleted_documents = {"english": 0, "persian": 0}

for i in range(x):
    title = english_df.iloc[i]["title"]
    description = english_df.iloc[i]["description"]
    collections["english"] += [[title, description]]
while True:
    split_text = input().split()
    if len(split_text) == 0:
        print("not a valid command!")
        continue
    if split_text[0] == "prepare":
        if len(split_text) != 2:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if (not lang == "english") and (not lang == "persian"):
            print("this language " + lang + " is not supported")
        else:
            document_tokens[lang], structured_documents[lang], terms[lang], stop_words_dic[lang] = prepare_text(
                collections[lang], lang, [])
            docs_size[lang] = len(structured_documents[lang])
            deleted_documents[lang] = [False for _ in range(len(structured_documents[lang]))]
    elif split_text[0] == "create":
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        if split_text[1] == "bigram":
            lang = split_text[2]
            if (not lang == "english") and (not lang == "persian"):
                print("this language " + lang + " is not supported")
            else:
                bigram(structured_documents[lang], bigram_index[lang], 1, docs_size[lang])
        elif split_text[1] == "positional":
            lang = split_text[2]
            if (not lang == "english") and (not lang == "persian"):
                print("this language " + lang + " is not supported")
            else:
                positional(structured_documents[lang], positional_index[lang], 1, docs_size[lang])
        else:
            print("not a valid command!")
    elif split_text[0] == "bigram":
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if (not lang == "english") and (not lang == "persian"):
            print("this language " + lang + " is not supported")
        else:
            biword = split_text[2]
            if len(biword) != 2:
                print(biword + " is not a biword!")
            else:
                if biword in bigram_index[lang].keys():
                    print(bigram_index[lang][biword])
                else:
                    print("biword (" + biword + ") doesn't match any word in " + lang + " documents.")
    elif split_text[0] == "positional":
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if (not lang == "english") and (not lang == "persian"):
            print("this language " + lang + " is not supported")
        else:
            term = split_text[2]
            if term in positional_index[lang].keys():
                print(positional_index[lang][term])
            else:
                print("term (" + term + ") doesn't match any term in " + lang + " documents.")
    elif split_text[0] == "exit":
        # edit_distance("nima", "sepehr")
        exit()
    elif split_text[0] == "tokens":
        if len(split_text) != 2:
            print("not a valid command!")
            continue
        lang = split_text[1]
        print(document_tokens[lang])
    elif split_text[0] == "stopwords":
        if len(split_text) != 2:
            print("not a valid command!")
            continue
        lang = split_text[1]
        print(stop_words_dic[lang])
    elif split_text[0] == "terms":
        if len(split_text) != 2:
            print("not a valid command!")
            continue
        lang = split_text[1]
        print(terms[lang])
    elif split_text[0] == "delete":
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if (not lang == "english") and (not lang == "persian"):
            print("this language " + lang + " is not supported")
        else:
            doc_id = int(split_text[2])
            delete(structured_documents[lang], doc_id - 1, bigram_index[lang], positional_index[lang],
                   deleted_documents[lang])
    elif split_text[0] == "insert":
        if len(split_text) != 4:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if (not lang == "english") and (not lang == "persian"):
            print("this language " + lang + " is not supported")
        else:
            doc_number = int(split_text[2])
            part_number = int(split_text[3])
            new_docs = []
            for _ in range(doc_number):
                new_docs += [[]]
                for i in range(part_number):
                    t = input()
                    new_docs[-1] += [t]
            insert(new_docs, lang, bigram_index[lang], positional_index[lang])
    elif split_text[0] == "save":
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        type_of_indexing = split_text[1]
        lang = split_text[2]
        if (not lang == "english") and (not lang == "persian"):
            print("this language " + lang + " is not supported")
        else:
            if lang == "english" and type_of_indexing == "positional":
                with open('positional_english_indexing', 'wb') as pickle_file:
                    pickle.dump(positional_index["english"], pickle_file)
                    pickle_file.close()
            elif lang == "english" and type_of_indexing == "bigram":
                with open('bigram_english_indexing', 'wb') as pickle_file:
                    pickle.dump(bigram_index["english"], pickle_file)
                    pickle_file.close()
            elif lang == "persian" and type_of_indexing == "positional":
                with open('positional_persian_indexing', 'wb') as pickle_file:
                    pickle.dump(positional_index["persian"], pickle_file)
                    pickle_file.close()
            elif lang == "persian" and type_of_indexing == "bigram":
                with open('bigram_persian_indexing', 'wb') as pickle_file:
                    pickle.dump(bigram_index["persian"], pickle_file)
                    pickle_file.close()
            print(type_of_indexing + " " + lang + " indexing saved successfully")
    elif split_text[0] == "load":
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        type_of_indexing = split_text[1]
        lang = split_text[2]
        if (not lang == "english") and (not lang == "persian"):
            print("this language " + lang + " is not supported")
        else:
            if lang == "english" and type_of_indexing == "positional":
                with open('positional_english_indexing', 'rb') as pickle_file:
                    positional_index["english"] = pickle.load(pickle_file)
                    pickle_file.close()
            elif lang == "english" and type_of_indexing == "bigram":
                with open('bigram_english_indexing', 'rb') as pickle_file:
                    bigram_index["english"] = pickle.load(pickle_file)
                    pickle_file.close()
            elif lang == "persian" and type_of_indexing == "positional":
                with open('positional_persian_indexing', 'rb') as pickle_file:
                    positional_index["persian"] = pickle.load(pickle_file)
                    pickle_file.close()
            elif lang == "persian" and type_of_indexing == "bigram":
                with open('bigram_persian_indexing', 'rb') as pickle_file:
                    bigram_index["persian"] = pickle.load(pickle_file)
                    pickle_file.close()
            print(type_of_indexing + " " + lang + " indexing loaded successfully")
    elif split_text[0] == "jaccard":
        if len(split_text) != 4:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if (not lang == "english") and (not lang == "persian"):
            print("this language " + lang + " is not supported")
        else:
            print(jaccard_similarity(split_text[2], split_text[3], lang))
    elif split_text[0] == "correctionlist":
        if len(split_text) != 3:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if (not lang == "english") and (not lang == "persian"):
            print("this language " + lang + " is not supported")
        else:
            corrected_list = []
            threshold = 0.4
            while len(corrected_list) == 0:
                corrected_list = correction_list(split_text[2], lang, threshold)
                threshold -= 0.1
            print(corrected_list)
    elif split_text[0] == "editdistance":
        if len(split_text) != 4:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if (not lang == "english") and (not lang == "persian"):
            print("this language " + lang + " is not supported")
        else:
            print(edit_distance(split_text[2], split_text[3]))
    elif split_text[0] == "query":
        if len(split_text) != 2:
            print("not a valid command!")
            continue
        lang = split_text[1]
        if (not lang == "english") and (not lang == "persian"):
            print("this language " + lang + " is not supported")
        else:
            query = input()
            document = [[query]]
            query_tokens, _, _, _ = prepare_text(document, lang, stop_words_dic[lang])
            correct_query = True
            correction = []
            for token in query_tokens:
                if token not in positional_index[lang].keys():
                    correct_query = False
                    threshold = 0.4
                    suggested_list = []
                    while len(suggested_list) == 0:
                        suggested_list = correction_list(token, lang, threshold)
                        threshold -= 0.1
                    edit_distances = []
                    for term in suggested_list:
                        edit_distances += [edit_distance(token, term)]
                    ind = edit_distances.index(min(edit_distances))
                    correction += [suggested_list[ind]]
                else:
                    correction += [token]

            if correct_query:
                print("no spell correction needed!")
            else:
                str = "suggested correction for the query:"
                for word in correction:
                    str += (" " + word)
                print(str)
    else:
        print("not a valid command!")
# prepare english
# create bigram english
# create positional english
# positional english jupyt
# insert english 1 2
# hi Jupyter!
# city is available
# positional english jupyt
# positional english korppoo
# bigram english rp
# delete english 2550
# positional english korppoo
# bigram english rp
# exit
