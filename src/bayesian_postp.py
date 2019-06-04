import os


class Bayesian_processor(object):
    """docstring for Bayesian_processor"""
    def __init__(self, bigrams, letters_dir=None, chars=None):
        super(Bayesian_processor, self).__init__()
        self.bigrams = bigrams
        if chars is None:
            self.chars = os.listdir(letters_dir)
        else:
            self.chars = chars

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
        posterior_word = []
        for i, prior_softmax in enumerate(predicted_word):
            posterior_letter = [0] * len(chars)

            if i > 0:
                previous_softmax = predicted_word[i-1]  # or posterior word?
                previous_prior = max(previous_softmax)
                previous_letter = previous_softmax.index(previous_prior)
                for j, char in enumerate(self.chars):
                    this_bigram = self.chars[previous_letter] + char
                    posterior_letter[j] += bigrams[this_bigram] * prior_softmax[j] / previous_prior


            if i < len(predicted_word)-1:
                next_softmax = predicted_word[i+1]  # or posterior word?
                next_prior = max(next_softmax)
                next_letter = next_softmax.index(next_prior)
                for j, char in enumerate(self.chars):
                    this_bigram = char + self.chars[next_letter]
                    posterior_letter[j] += bigrams[this_bigram] * prior_softmax[j] / next_prior

            posterior_word.append(posterior_letter)
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
        print(''.join([self.chars[letter.index(max(letter))] for letter in word]))

        # Word as separate probabilities
        [print(f"{self.chars[letter.index(max(letter))]}\t(p = {max(letter):.2f})") for letter in word]
        print()

    def append_word_to_file(self, word, file):
        # TBA
        word = ''.join([self.chars[letter.index(max(letter))] for letter in word])
        pass




if __name__ == "__main__":
    # When running this script standalone, use this example:

    # Define alphabet
    chars = 'ABC'

    # Define bigram probabilities
    bigrams = {
        'AA': 0.3,
        'AB': 0.2,
        'AC': 0.5,

        'BA': 0.1,
        'BB': 0.2,
        'BC': 0.7,

        'CA': 0.3,
        'CB': 0.9,
        'CC': 0.3
    }

    # Construct mock prediction softmax
    predicted_word = [
        [0.8, 0.1, 0.1],
        [0.2, 0.6, 0.2],
        [0.4, 0.4, 0.2],
        [0.4, 0.0, 0.6],
        [0.3, 0.3, 0.4]
    ]

    processor = Bayesian_processor(bigrams=bigrams,
                                   chars=chars)
    posterior_word = processor.process_word(predicted_word)
    posterior_word = processor.normalize_posteriors(posterior_word)

    processor.print_word(predicted_word, "Predicted word (network output)")
    processor.print_word(posterior_word, "Normalized word (after bigrams)")
