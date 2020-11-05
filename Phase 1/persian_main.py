from __future__ import unicode_literals

import re
import xml.etree.ElementTree as ET
from collections import Counter
import matplotlib.pyplot as plt

from hazm import *

tree = ET.parse('data/Persian.xml')
root = tree.getroot()
titles = []
descriptions = []


def remove_punctuation_from_word(selected_word, punctuation_list):
    final_word = ""
    for a in selected_word:
        if a not in punctuation_list:
            final_word += a
    return final_word


for child in root:
    for sub_child in child:
        if sub_child.tag == '{http://www.mediawiki.org/xml/export-0.10/}title':
            titles.append(sub_child.text)
        if sub_child.tag == '{http://www.mediawiki.org/xml/export-0.10/}revision':
            revision = sub_child
            for x in revision:
                if x.tag == '{http://www.mediawiki.org/xml/export-0.10/}text':
                    descriptions.append(x.text)

punctuation = ['!', '"', "'", '#', '(', ')', '*', '-', ',', '.', '/', ':', '[', ']', '|', ';', '?', '،', '...', '$',
               '{',
               '}', '=', '==', '===', '>', '<', '>>', '<<', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹', '۰', '«', '||',
               '""', "''", "&", "'''", '"""', '»', '', '–', "؛", "^"]
normalizer = Normalizer()
for i in range(len(titles)):
    titles[i] = normalizer.normalize(titles[i])
    titles[i] = word_tokenize(titles[i])
    descriptions[i] = normalizer.normalize(descriptions[i])
    descriptions[i] = word_tokenize(descriptions[i])
    titles_array = []
    descriptions_array = []
    for x in titles[i]:
        for word in x.split('|'):
            if len(re.findall(r'([a-zA-Z]+)', word)) == 0:
                titles_array.append(word)
    for x in descriptions[i]:
        for word in x.split('|'):
            if len(re.findall(r'([a-zA-Z]+)', word)) == 0:
                descriptions_array.append(word)
    titles[i] = titles_array
    descriptions[i] = descriptions_array

stemmer = Stemmer()

all_tokens = []
dictionary = []
for i in range(len(titles)):
    title_arr = []
    description_arr = []
    for x in titles[i]:
        # x = remove_punctuation_from_word(x, punctuation)
        if x not in punctuation and len(x) > 0:
            all_tokens.append(stemmer.stem(x))
            title_arr.append(stemmer.stem(x))
    for x in descriptions[i]:
        # x = remove_punctuation_from_word(x, punctuation)
        if x not in punctuation and len(x) > 0:
            all_tokens.append(stemmer.stem(x))
            description_arr.append(stemmer.stem(x))
    dictionary.append([title_arr, description_arr])


frequency_counter = Counter(all_tokens)
tokens_size = len(all_tokens)
sorted_token_counter = frequency_counter.most_common(len(frequency_counter))
sorted_token_ratio = [(c[0], c[1] / tokens_size) for c in sorted_token_counter]
stop_words = [sorted_token_counter[i][0] for i in range(45)]
r = range(45)
y = [sorted_token_counter[i][1] for i in range(45)]
plt.bar(r, y, color="red", align="center")
plt.title("Stopwords Frequencies")
plt.xticks(r, stop_words, rotation="vertical")
plt.show()

