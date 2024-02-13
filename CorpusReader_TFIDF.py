import nltk
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
from nltk.corpus import stopwords
import math

class CorpusReader_TFIDF:
    """
    Initializes the CorpusReader_TFIDF with specified parameters.

    :param corpus: An NLTK corpus object.
    :param tf: Term frequency calculation method ('raw' or 'log').
    :param idf: Inverse document frequency calculation method ('base' or 'smooth').
    :param stopWord: Stopword removal strategy ('none', 'standard', or filepath to custom stopwords).
    :param toStem: Boolean indicating whether to apply stemming.
    :param stemFirst: Boolean indicating whether to stem before removing stopwords.
    :param ignoreCase: Boolean indicating whether to ignore case.
    """
    def __init__(self, corpus, tf="raw", idf="base", stopWord="none", toStem=False, stemFirst=False, ignoreCase=True):
        self.corpus = corpus
        self.tf_method = tf
        self.idf_method = idf
        self.stopWord =  stopWord 
        self.toStem = toStem
        self.stemFirst = stemFirst
        self.ignoreCase = ignoreCase
        self.stemmer = SnowballStemmer("english") if toStem else None
        self.idf_values = {}
        self.tfidf_vectors = {}
        nltk.download('stopwords')
        

    ####### Shared methods for both TF and IDF calculations #######
    def fileids(self):
        return self.corpus.fileids()

    def raw(self, fileids=None):
        return self.corpus.raw(fileids=fileids)

    def words(self, fileids=None):
        original_words = self.corpus.words(fileids=fileids)
        processed_words = self._preprocess_words(original_words)
        return processed_words

    ####### Methods for TF-IDF calculations #######
    def tfidf(self, fileid, returnZero=False):
        """Calculate TF-IDF for a specific document, optionally including terms with a TF-IDF value of 0."""
        # Ensure IDF values are calculated
        if not self.idf_values:
            self.idf1() # Ensure IDF values are calculated

        # Retrieve the processed words for the specified document
        words = self.words(fileids=fileid)
        # Calculate term frequency for these words
        tf = self._calculate_tf(words)

        # Calculate TF-IDF vector
        tfidf_vector = {}
        for word, tf_val in tf.items():
            idf_val = self.idf_values.get(word, 0)
            tfidf_score = tf_val * idf_val
            if tfidf_score > 0 or returnZero:
                tfidf_vector[word] = tfidf_score
            elif returnZero and idf_val == 0:  # Explicitly handle the case where a term is not in the corpus
                tfidf_vector[word] = 0

        return tfidf_vector


    def tfidfAll(self, returnZero=False):
        """Calculate and return TF-IDF vectors for all documents in the corpus."""
        self.idf1() # Ensure IDF values are calculated
        for fileid in self.fileids():
            self.tfidf_vectors[fileid] = self.tfidf(fileid)
        return self.tfidf_vectors

    def tfidfNew(self, words):
        """Calculate TF-IDF for a new document represented by a list of words."""
        if not self.idf_values:  # Ensure IDF values are calculated
            self._calculate_idf()
        processed_words = self._preprocess_words(words)
        tf = self._calculate_tf(processed_words)
        tfidf = {word: tf_val * self.idf_values.get(word, 0) for word, tf_val in tf.items()}
        return tfidf

    def idf1(self):
        """Return the IDF values for all terms in the corpus."""
        if not self.idf_values:  # Ensure IDF values are calculated
            self._calculate_idf()
        return self.idf_values
    
    # Cosine similarity requires comparing the TF-IDF vectors of two documents.
    def cosine_sim(self, fileid1, fileid2):
        """Calculate the cosine similarity between two documents."""
        tfidf1 = self.tfidf(fileid1)
        tfidf2 = self.tfidf(fileid2)
        
        # Calculate dot product
        dot_product = sum(tfidf1.get(word, 0) * tfidf2.get(word, 0) for word in tfidf1)
        
        # Calculate norms
        norm1 = math.sqrt(sum(val ** 2 for val in tfidf1.values()))
        norm2 = math.sqrt(sum(val ** 2 for val in tfidf2.values()))
        
        # Compute cosine similarity
        if norm1 * norm2 == 0:
            return 0  # Avoid division by zero
        else:
            return dot_product / (norm1 * norm2)
   

    def cosine_sim_new(self, words, fileid):
        """Calculate the cosine similarity between a new document (represented by a list of words)
        and the document specified by fileid."""
        # Calculate TF-IDF for the new document using the provided list of words
        tfidf_new_doc = self.tfidfNew(words)

        # Calculate TF-IDF for the existing document specified by fileid
        tfidf_existing_doc = self.tfidf(fileid)

        # Calculate dot product
        dot_product = sum(tfidf_new_doc.get(word, 0) * tfidf_existing_doc.get(word, 0) for word in tfidf_new_doc)

        # Calculate norms
        norm_new_doc = math.sqrt(sum(val ** 2 for val in tfidf_new_doc.values()))
        norm_existing_doc = math.sqrt(sum(val ** 2 for val in tfidf_existing_doc.values()))

        # Compute cosine similarity
        if norm_new_doc * norm_existing_doc == 0:
            return 0  # Avoid division by zero
        else:
            return dot_product / (norm_new_doc * norm_existing_doc)

    def query(self, words):
        """Return a list of (document, cosine_sim) tuples that calculate the cosine similarity between
        the 'new' document (specified by the list of words as the document) and each document in the corpus.
        The list is ordered in decreasing order of cosine similarity.
        """
        # Calculate TF-IDF for the new document represented by the list of words
        tfidf_new_doc = self.tfidfNew(words)

        # Calculate cosine similarity between the new document and each document in the corpus
        similarity_scores = []
        for fileid in self.fileids():
            cosine_similarity = self.cosine_sim(fileid, words)
            similarity_scores.append((fileid, cosine_similarity))

        # Sort the list by cosine similarity in decreasing order
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        return similarity_scores
    
    ####### Helper methods #######
    # Use dynamic preproccessing instead of preproccessing the whole corpus we leave the original corpus untouched
    # and preproccess the words on the on each method call
    def _preprocess_words(self, words):
        # Load stop words
        stop_words = set()
        if self.stopWord == "standard":
            stop_words = set(stopwords.words("english"))
        elif self.stopWord != "none":
            stop_words = self._load_stopwords_from_file(self.stopWord)

        # Case normalization
        if self.ignoreCase:
            words = [word.lower() for word in words]

        # Stem stop words if stemFirst is True
        if self.stemFirst and self.toStem:
            stop_words = {self.stemmer.stem(word) for word in stop_words}

        # Apply stemming and stop words removal
        if self.toStem:
            if self.stemFirst:  # Stem first, then remove stop words
                stemmed_words = [self.stemmer.stem(word) for word in words]
                return [word for word in stemmed_words if word not in stop_words]
            else:  # Remove stop words first, then stem
                filtered_words = [word for word in words if word not in stop_words]
                return [self.stemmer.stem(word) for word in filtered_words]
        else:
            return [word for word in words if word not in stop_words]
        
    def _calculate_tf(self, words):
        """Calculate term frequency for a list of words."""
        tf = {}
        for word in words:
            if word in tf:
                tf[word] += 1
            else:
                tf[word] = 1
        
        if self.tf_method == "log":
            for word, count in tf.items():
                tf[word] = 1 + math.log(count, 2)
        
        return tf
    
    def _calculate_idf(self):
        """Calculate inverse document frequency for all words in the corpus."""
        df = {}
        total_docs = len(self.fileids())
        for fileid in self.fileids():
            words = set(self.words(fileids=fileid))  # Ensure uniqueness
            for word in words:
                if word in df:
                    df[word] += 1
                else:
                    df[word] = 1
        
        for word, count in df.items():
            if self.idf_method == "smooth":
                self.idf_values[word] = math.log(1 + total_docs / (1 + count), 2)
            else:  # "base"
                self.idf_values[word] = math.log(total_docs / count, 2)
    
    def _load_stopwords_from_file(self, filename):
        """Load stopwords from a given file."""
        try:
            with open(filename, 'r') as file:
                stopwords_list = file.read().splitlines()
            return set(stopwords_list)
        except FileNotFoundError:
            print(f"Stopwords file {filename} not found.")
            return set()