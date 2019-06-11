import os
from os.path import join, abspath
import numpy as np

fn = join(abspath('..'), 'ngrams.npz')
char_map = {'Alef' : 0, 
            'Ayin' : 1, 
            'Bet' : 2, 
            'Dalet' : 3, 
            'Gimel' : 4, 
            'He' : 5, 
            'Het' : 6, 
            'Kaf' : 7, 
            'Kaf-final' : 8, 
            'Lamed' : 9, 
            'Mem' : 10, 
            'Mem-medial' : 11, 
            'Nun-final' : 12, 
            'Nun-medial' : 13, 
            'Pe' : 14, 
            'Pe-final' : 15, 
            'Qof' : 16, 
            'Resh' : 17, 
            'Samekh' : 18, 
            'Shin' : 19, 
            'Taw' : 20, 
            'Tet' : 21, 
            'Tsadi-final' : 22, 
            'Tasdi-final' : 22, # catch typo in dataset
            'Tsadi-medial' : 23, 
            'Tsadi' : 23,       # catch typo in dataset
            'Waw' : 24, 
            'Yod' : 25, 
            'Zayin' : 26
        }

hebrew_map = {
            0:              u'\u05D0',
            1:              u'\u05E2',
            2:              u'\u05D1',
            3:              u'\u05D3',
            4:              u'\u05D2',
            5:              u'\u05D4',
            6:              u'\u05D7',
            7:              u'\u05DB',
            8:              u'\u05DA',
            9:              u'\u05DC',
            10:             u'\u05DD',
            11:             u'\u05DE',
            12:             u'\u05DF',
            13:             u'\u05E0',
            14:             u'\u05E4',
            15:             u'\u05E3',
            16:             u'\u05E7',
            17:             u'\u05E8',
            18:             u'\u05E1',
            19:             u'\u05E9',
            20:             u'\u05EA',
            21:             u'\u05D8',
            22:             u'\u05E5',
            23:             u'\u05E6',
            24:             u'\u05D5',
            25:             u'\u05D9',
            26:             u'\u05D6'
        }

## needed for the final program, as YOLO now returns hebrew
rev_hebrew_map = {}
for key, val in hebrew_map.items(): 
    rev_hebrew_map[val] = key


class Bayesian_processor():
    """docstring for Bayesian_processor"""
    def __init__(self):
        self.ngrams = np.load(fn)
        self.unigrams = self.ngrams['unigrams']
        self.bigrams = self.ngrams['bigrams']


    def process_word(self, predicted_word):

        """
        P(letter | surrounding letters) = P(surrounding letters | letter)
                                          * p(letter) / P(surrounding letters)

                                        = P(previous letter | letter)
                                          * p(letter) / p(previous letter)
                                          +
                                          P(next letter | letter)
                                          * p(letter) / p(next letter)
        """
        posterior_word = np.zeros((len(predicted_word), len(self.unigrams)), dtype=np.double)

        ## forward pass
        for idx, prior_softmax in enumerate(predicted_word):

            if idx > 0:
                previous_softmax = predicted_word[idx-1]  # or posterior word?
                ## get index of previous letter
                previous_prior = max(previous_softmax)
                previous_letter = previous_softmax.index(previous_prior)

                for jdx, unigram_prob in enumerate(self.unigrams):

                    bigram_prob = self.bigrams[previous_letter, jdx]
                    if bigram_prob == 0.:
                        bigram_prob = unigram_prob ## very naive approach. improve later

                    posterior_word[idx, jdx] += bigram_prob * prior_softmax[jdx] / previous_prior

        ## backward pass
        for idx in range(len(predicted_word) - 2, -1, -1):
            prior_softmax = predicted_word[idx]
            next_softmax = predicted_word[idx+1]  # or posterior word?
            ## get index of next letter
            next_prior = max(next_softmax)
            next_letter = next_softmax.index(next_prior)

            for jdx, unigram_prob in enumerate(self.unigrams):
                bigram_prob = self.bigrams[jdx, next_letter]
                if bigram_prob == 0.:
                    bigram_prob = unigram_prob
                posterior_word[idx, jdx] += bigram_prob * prior_softmax[jdx] / next_prior
        return posterior_word

    def normalize_posteriors(self, word):
        """Normalize probabilites per letter
        (e.g., [0.1, 0.2, 0.1] to [0.25, 0.5, 0.25])
        """
        return [[p / sum(posteriors) for p in posteriors]
                for posteriors in word]

    def print_word(self, word, title=None):
        if title is not None:
            print(f"{title}:")

        # Word as string
        print(''.join([hebrew_map[letter.index(max(letter))] for letter in word]))

        # Word as separate probabilities
        [print(f"{hebrew_map[letter.index(max(letter))]}\t(p = {max(letter):.2f})") for letter in word]
        print()

    def append_word_to_file(self, word, file):
        # TBA
        wordstring = ''.join([hebrew_map[letter.index(max(letter))]
                              for letter in word])




if __name__ == "__main__":
    # When running this script standalone, use this example:

    # Construct mock prediction softmax (of length (n x 27) )
    predicted_word = [
        [0.8, 0.1, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1, 0.1],
        [0.2, 0.6, 0.2, 0.8, 0.1, 0.1, 0.2, 0.6, 0.2, 0.8, 0.1, 0.1, 0.2, 0.6, 0.2, 0.8, 0.1, 0.1, 0.2, 0.6, 0.2, 0.8, 0.1, 0.1, 0.3, 0.1, 0.3],
        [0.4, 0.4, 0.2, 0.8, 0.1, 0.1, 0.2, 0.6, 0.2, 0.8, 0.1, 0.1, 0.2, 0.6, 0.2, 0.8, 0.1, 0.1, 0.2, 0.6, 0.2, 0.8, 0.1, 0.1, 0.3, 0.1, 0.3],
        [0.4, 0.0, 0.6, 0.8, 0.1, 0.1, 0.2, 0.6, 0.2, 0.8, 0.1, 0.1, 0.2, 0.6, 0.2, 0.8, 0.1, 0.1, 0.2, 0.6, 0.2, 0.8, 0.1, 0.1, 0.3, 0.1, 0.3],
        [0.3, 0.3, 0.4, 0.8, 0.1, 0.1, 0.2, 0.6, 0.2, 0.8, 0.1, 0.1, 0.2, 0.6, 0.2, 0.8, 0.1, 0.1, 0.2, 0.6, 0.2, 0.8, 0.1, 0.1, 0.3, 0.1, 0.3]
    ]

    processor = Bayesian_processor()
    posterior_word = processor.process_word(predicted_word)
    posterior_word = processor.normalize_posteriors(posterior_word)

    processor.print_word(predicted_word, "Predicted word (network output)")
    processor.print_word(posterior_word, "Normalized word (after bigrams)")
