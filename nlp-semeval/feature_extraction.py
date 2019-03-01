import codecs
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class FeatureExtractor:

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.w2i = []
        self.i2w = []
        self.glove_wvecs = []
        self.max_len = 0
        # self.build_glove_wvecs()

    def tfidf_fit_transform(self, corpus):
        return self.vectorizer.fit_transform(corpus)

    def tfidf_transform(self, tweet):
        return self.vectorizer.transform([tweet])[0]

    def glove_get_max_len(self):
        return self.max_len

    def glove_set_max_len(self, corpus):
        self.max_len = max([len(sentence.split()) for sentence in corpus])

    def glove_transform(self, tweet):
        transformed = np.zeros((self.max_len, 300))
        words = tweet.split()
        for i in range(0, len(words)):
            transformed[i] = self.glove_transform_word(words[i])
        return np.transpose(transformed)

    def glove_transform_word(self, word):
        if word in self.w2i:
            return self.glove_wvecs[self.w2i[word]]
        else:
            return self.glove_wvecs[0]

    def build_glove_wvecs(self):
        # this is a large file, it will take a while to load in the memory!
        with codecs.open('glove/glove.6B.300d.txt', 'r', 'utf-8') as f:
            index = 0
            for line in tqdm(f.readlines()):
                # Ignores the first line - first line typically contains vocab, dimensionality
                if len(line.strip().split()) > 3:
                    (word, vec) = (line.strip().split()[0],
                                   list(map(float, line.strip().split()[1:])))
                    self.glove_wvecs.append(vec)
                    self.w2i.append((word, index))
                    self.i2w.append((index, word))
                    index += 1

        self.w2i = dict(self.w2i)
        self.i2w = dict(self.i2w)
        self.glove_wvecs = np.array(self.glove_wvecs)
