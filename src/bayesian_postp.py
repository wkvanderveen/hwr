import os
from os.path import join, abspath
from sliding_window import SlidingWindow
import numpy as np
from write_to_file import write_to_file

FINAL_OUTPUT_SAVE_PATH = "../data/program_output/"
FINAL_NAME = "output"

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
        self.trigrams = self.ngrams['trigrams']

        ## parameters for simple interpolation
        self.TRIGRAM_IMPORTANCE = 0.33
        self.BIGRAM_IMPORTANCE = 0.33
        self.UNIGRAM_IMPORTANCE = 1.0 - self.TRIGRAM_IMPORTANCE - self.BIGRAM_IMPORTANCE
        self.MINIMAL_SAME_CHAR_SEQ_LEN = 3

    def process_word(self, predicted_word):
        """
        P( z | xy) = lambda * P(xyz) / P(xy) + mu * P(yz) / P(y) + (1.0 - lambda - mu) * P(z) / P(sum unigrams) 
        """
        posterior_word = np.zeros((len(predicted_word), len(self.unigrams)), dtype=np.double)

        ## forward pass
        for idx, prior_softmax in enumerate(predicted_word):
            if idx > 1: ## use trigrams
                first_softmax = predicted_word[idx-2]
                second_softmax = predicted_word[idx-1]
                nr_classes = len(self.unigrams)


                for third_letter in range(nr_classes):
                    letter_probs = []
                    for second_letter in range(nr_classes):
                        for first_letter in range(nr_classes):
                            trigram_divisor = self.bigrams[first_letter, second_letter] ## bigram-prob used for division

                            trigram_prob = self.trigrams[first_letter, second_letter, third_letter] / trigram_divisor
                            bigram_prob = self.bigrams[second_letter, third_letter] / self.unigrams[second_letter]
                            unigram_prob = self.unigrams[third_letter] / np.sum(self.unigrams)

                            prob = trigram_prob * self.TRIGRAM_IMPORTANCE + bigram_prob * self.BIGRAM_IMPORTANCE +\
                                 unigram_prob * self.UNIGRAM_IMPORTANCE

                            # take priors of all letters into account
                            prob *= prior_softmax[third_letter] * second_softmax[second_letter] * first_softmax[first_letter]

                            letter_probs.append(prob)
                    posterior_word[idx, third_letter] += max(letter_probs)

        ## backward pass
        for idx in range(len(predicted_word) - 3, -1, -1):
            prior_softmax = predicted_word[idx]
            third_softmax = predicted_word[idx+2]
            second_softmax = predicted_word[idx+1]
            nr_classes = len(self.unigrams)


            for first_letter in range(nr_classes):
                letter_probs = []
                for second_letter in range(nr_classes):
                    for third_letter in range(nr_classes):
                        trigram_divisor = self.bigrams[second_letter, third_letter] ## bigram-prob used for division

                        trigram_prob = self.trigrams[first_letter, second_letter, third_letter] / trigram_divisor
                        bigram_prob = self.bigrams[first_letter, second_letter] / self.unigrams[second_letter]
                        unigram_prob = self.unigrams[first_letter] / np.sum(self.unigrams)

                        prob = trigram_prob * self.TRIGRAM_IMPORTANCE + bigram_prob * self.BIGRAM_IMPORTANCE +\
                             unigram_prob * self.UNIGRAM_IMPORTANCE

                        prob *= prior_softmax[first_letter] * second_softmax[second_letter] * third_softmax[third_letter]

                        letter_probs.append(prob)
                posterior_word[idx, first_letter] += max(letter_probs)
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

    def probs_to_one_hot(self, arr):
        arr_len = len(arr)
        arr = np.array(arr)
        new = np.zeros(arr_len, dtype = int)
        new[np.argmax(arr)] = 1
        new = new.tolist()
        return new

    # This function will produce output such that each character in a same character sequence only occurs once. 
    def filter_on_seq_of_same_chars(self, probabilities):
        one_hots = []
        trash_indices = []
        #convert softmax arrays to one hot arrays
        for arr in probabilities:
            one_hot = self.probs_to_one_hot(arr)
            one_hots.append(one_hot)
        for idx in range(0, len(one_hots)):
            try:
                if one_hots[idx] == one_hots[idx+1]:
                    trash_indices.append(idx+1)
            except:
                pass
        one_hots = [j for i, j in enumerate(one_hots) if i not in trash_indices]
        return one_hots

    # This function filters the character sequence on single occuring characters in a sequence.
    # These characters are regarded as noise. E.g. in the ence AAABAACCCCCC, B would be regarded as noise
    # The function also downscales the char array such that only same char sequences with length >= self.MINIMAL_SAME_CHAR_SEQ_LEN
    # are kept. 
    def sequence_denoiser(self, probabilities):
        one_hots = []
        trash_indices = []
        denoise_distance = 2
        #convert softmax arrays to one hot arrays
        for arr in probabilities:
            one_hot = self.probs_to_one_hot(arr)
            one_hots.append(one_hot)
        for idx in range(0, len(one_hots)):
            try:
                for kernel_idx in range(1, self.MINIMAL_SAME_CHAR_SEQ_LEN):
                    if not (one_hots[idx] == one_hots[idx+kernel_idx+1]):
                        trash_indices.append(idx+kernel_idx+1)
            except:
                pass
        one_hots = [j for i, j in enumerate(one_hots) if i not in trash_indices]
        return one_hots

    def apply_postprocessing(self, probabilities):
        probabilities = self.sequence_denoiser(probabilities)
        probabilities = self.filter_on_seq_of_same_chars(probabilities)
        posteriors = self.process_word(probabilities)
        posteriors = self.normalize_posteriors(posteriors)
        posteriors = self.filter_on_seq_of_same_chars(posteriors) # Filter on same-char-sequences, these may be produced by the n-grams post processing
        final_sentence = ""
        for letter_probs in posteriors:
            best_letter_val = max(letter_probs)
            best_letter_index = letter_probs.index(best_letter_val)

            final_sentence += hebrew_map[best_letter_index]

        return final_sentence




if __name__ == "__main__":
    # When running this script standalone, use this example:

    # Construct mock prediction softmax (of length (n x 27) )
    processor = Bayesian_processor()
    sw = SlidingWindow()
    image_file = "../data/backup_val_lines/line5.jpg"
    sw.load_image(image_file)
    # posterior_word = processor.process_word(predicted_word)
    # posterior_word = processor.normalize_posteriors(posterior_word)

    # print(posterior_word)

    # processor.print_word(predicted_word, "Predicted word (network output)")
    # processor.print_word(posterior_word, "Normalized word (after bigrams)")
    predicted_sentence = sw.get_letters()
    sentence = processor.apply_postprocessing(predicted_sentence)
    write_to_file(sentence, FINAL_OUTPUT_SAVE_PATH, FINAL_NAME)
    print(sentence)