from nltk import word_tokenize, SnowballStemmer

a = "Marvin Minsky's arch, eclectic, charmingly offhand talk on health, overpopulation, kazem"
x = word_tokenize(a)
b = [word.lower() for word in x]
c = [word for word in b if word.isalpha()]
stemmer = SnowballStemmer("english")
d = []
for word in c:
    d.append(stemmer.stem(word))
print(d)