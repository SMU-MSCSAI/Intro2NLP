from nltk.corpus import inaugural, PlaintextCorpusReader
import nltk
from CorpusReader_TFIDF import *


# nltk.download('inaugural')
# print(inaugural.words())
nltk.download('punkt')
print(len(inaugural.words()))
print(inaugural.fileids())
print(inaugural.sents(['1789-washington.txt']))

