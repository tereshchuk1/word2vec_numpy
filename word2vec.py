import numpy as np
from collections import defaultdict


class word2vec():
    def __init__(self, settings):
        self.n = settings['n']
        self.lr = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']

    def generate_training_data(self, corpus):
        """Build vocabulary and generate (target, context) index pairs."""
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1

        # Save word counts for weighted negative sampling P_n(w) = freq^0.75 / Z
        self.word_counts = word_counts
        self.v_count = len(word_counts.keys())
        self.words_list = list(word_counts.keys())
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []
        for sentence in corpus:
            sent_len = len(sentence)
            for i, word in enumerate(sentence):
                w_target = self.word_index[sentence[i]]
                w_context = []
                for j in range(i - self.window, i + self.window + 1):
                    # Skip target word and out-of-bounds indices
                    if j != i and j <= sent_len - 1 and j >= 0:
                        w_context.append(self.word_index[sentence[j]])
                training_data.append([w_target, w_context])
        return training_data

    def subsample_corpus(self, corpus, t=0.001):
        """Discard frequent words, reduces noise from uninformative words like 'the', 'and'"""
        total_words = sum(self.word_counts.values())
        freq_relative = {word: count / total_words
                         for word, count in self.word_counts.items()}
        subsampled = []
        for sentence in corpus:
            new_sentence = [
                word for word in sentence
                if np.random.random() < min(1, np.sqrt(t / freq_relative[word]))
            ]
            if len(new_sentence) > 1:
                subsampled.append(new_sentence)
        return subsampled

    def train(self, training_data):
        """
        Skip-gram with Negative Sampling (SGNS)
        """
        # w1: target word embeddings, w2: context word embeddings
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))
        self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))

        # Precompute P_n(w) = freq^0.75 / Z, smoothed unigram distribution
        freq = np.array([self.word_counts[self.index_word[i]] for i in range(self.v_count)])
        self.neg_prob = freq ** 0.75
        self.neg_prob /= self.neg_prob.sum()

        for i in range(self.epochs):
            self.loss = 0
            for w_t, w_c in training_data:
                # Embedding lookup, equivalent to one-hot multiply but faster
                h = self.w1[w_t]

                for pos_idx in w_c:
                    u_pos = np.dot(self.w2[:, pos_idx], h)
                    p_pos = self.sigmoid(u_pos)
                    self.loss += -np.log(p_pos + 1e-10)

                    # d(-log(sigmoid(u)))/du = sigmoid(u) - 1
                    grad_pos = (p_pos - 1)

                    neg_indices = self.get_negative_samples(w_t, pos_idx)
                    grad_neg_sum = np.zeros(self.n)

                    for neg_idx in neg_indices:
                        u_neg = np.dot(self.w2[:, neg_idx], h)
                        p_neg = self.sigmoid(u_neg)
                        self.loss += -np.log(1 - p_neg + 1e-10)

                        # d(-log(1 - sigmoid(u)))/du = sigmoid(u)
                        grad_neg = p_neg
                        # Accumulate gradients from all negative samples before updating w1
                        grad_neg_sum += grad_neg * self.w2[:, neg_idx]
                        self.w2[:, neg_idx] -= self.lr * grad_neg * h

                    w2_pos_old = self.w2[:, pos_idx].copy()
                    self.w2[:, pos_idx] -= self.lr * grad_pos * h
                    # w1 updated once per context word, accumulates signal from pos + all neg
                    self.w1[w_t] -= self.lr * (grad_pos * w2_pos_old + grad_neg_sum)

            if (i + 1) % 10 == 0:
                print('Epoch:', i + 1, "Loss:", self.loss)

    def get_negative_samples(self, target_idx, pos_idx, k=5):
        """
        Sample k negatives using P_n(w) = freq^0.75 / Z.
        Samples k*2 candidates in one call for efficiency, then filters.
        """
        exclude = {target_idx, pos_idx}
        neg_samples = []
        while len(neg_samples) < k:
            # One vectorized call instead of k individual calls
            candidates = np.random.choice(self.v_count, size=k * 2, p=self.neg_prob)
            neg_samples += [idx for idx in candidates if idx not in exclude]
        return neg_samples[:k]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def evaluate(self, word, top_n=5):
        """Find most similar words using cosine similarity on w1 embeddings."""
        if word not in self.word_index:
            print(f"Word '{word}' not in vocabulary")
            return

        idx = self.word_index[word]
        vec = self.w1[idx]

        # Vectorized cosine similarity against all embeddings at once
        dot_products = self.w1 @ vec
        norms = np.linalg.norm(self.w1, axis=1) * np.linalg.norm(vec)
        similarities = dot_products / norms

        # Exclude the word itself
        similarities[idx] = -np.inf

        top_indices = np.argsort(similarities)[::-1][:top_n]

        print(f"\nMost similar to '{word}':")
        for i in top_indices:
            print(f"  {self.index_word[i]}: {similarities[i]:.3f}")
