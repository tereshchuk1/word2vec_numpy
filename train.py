import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from word2vec import word2vec

nltk.download('gutenberg', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus import gutenberg

# Corpus
raw_sents = gutenberg.sents('shakespeare-hamlet.txt')

corpus = [
    [re.sub(r'[^a-z]', '', word.lower()) for word in sent]
    for sent in raw_sents
]

corpus = [
    [word for word in sent if word]
    for sent in corpus
    if len([w for w in sent if w]) > 1
]

print(f"Sentences: {len(corpus)}")

# Settings

settings = {
    'window_size': 2,   # context window +- center word
    'n': 15,            # embedding dimensions (hidden layer size)
    'epochs': 100,      # number of training epochs
    'learning_rate': 0.0001
}

# Train
w2v = word2vec(settings)

# First pass: build vocabulary and word counts
training_data = w2v.generate_training_data(corpus)

# Subsample frequent words to improve rare word representations
corpus = w2v.subsample_corpus(corpus)

# Second pass on subsampled corpus
training_data = w2v.generate_training_data(corpus)

w2v.train(training_data)

# Evaluate
w2v.evaluate("hamlet")
w2v.evaluate("king")
w2v.evaluate("death")

# Visualize
def plot_embeddings(w2v, top_n=100, filename="assets/embedding_viz.png"):
    """Plot t-SNE of the top_n most frequent words."""
    top_words = sorted(w2v.word_counts, key=w2v.word_counts.get, reverse=True)[:top_n]
    indices = [w2v.word_index[w] for w in top_words]

    vectors = w2v.w1[indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(top_words) - 1))
    reduced = tsne.fit_transform(vectors)

    plt.figure(figsize=(14, 10))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5, s=20)
    for i, word in enumerate(top_words):
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=8)
    plt.title(f"Top {top_n} Word Embeddings (t-SNE)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved to {filename}")


plot_embeddings(w2v, top_n=100)
