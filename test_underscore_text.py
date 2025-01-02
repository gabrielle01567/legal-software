

import nltk as nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample text
test_text = (
    "The quick brown fox jumps over the lazy dog. "
    "This is a simple sentence. "
    "It is used to test keyboard and other devices. "
    "The dog is not happy."
)

# Download NLTK data if not already available
#nltk.download('punkt_tab', quiet=True)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1. Split the text into sentences.
sentences = nltk.sent_tokenize(test_text)

# 2. Tokenize sentences and build a set of unique words in one step.
all_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
unique_words = list({word for tokens in all_tokens for word in tokens})

# 3. Create a term-document matrix efficiently.
word_to_index = {word: idx for idx, word in enumerate(unique_words)}
term_doc_matrix = np.zeros((len(unique_words), len(sentences)))

for col_idx, tokens in enumerate(all_tokens):
    for word in tokens:
        row_idx = word_to_index[word]
        term_doc_matrix[row_idx, col_idx] += 1

print("Term-Document Matrix:\n", term_doc_matrix)

# 4. Apply Singular Value Decomposition (SVD).
U, S, Vt = np.linalg.svd(term_doc_matrix, full_matrices=False)

# Print the shapes of the decomposed matrices and the first few singular values.
print(f"U shape: {U.shape}, S shape: {S.shape}, Vt shape: {Vt.shape}")
print("\nFirst few singular values:", S[:5])

# 5. Prepare data for box-and-whisker plots.
# Convert term-document matrix to a DataFrame for easier analysis.
term_doc_df = pd.DataFrame(term_doc_matrix, index=unique_words, columns=[f"Sentence {i+1}" for i in range(len(sentences))])

# Melt the DataFrame for Seaborn's boxplot (long-form data required).
melted_df = term_doc_df.reset_index().melt(id_vars='index', var_name='Sentence', value_name='Word Frequency')
melted_df.rename(columns={'index': 'Word'}, inplace=True)

# Filter out rows with zero frequencies for cleaner plots.
melted_df = melted_df[melted_df['Word Frequency'] > 0]

# 6. Create box-and-whisker plots with Seaborn.
plt.figure(figsize=(12, 6))
sns.boxplot(data=melted_df, x='Sentence', y='Word Frequency', palette='coolwarm')
plt.title("Box-and-Whisker Plot of Word Frequencies by Sentence")
plt.xlabel("Sentence")
plt.ylabel("Word Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
from nltk.tokenize import sent_tokenize, word_tokenize

def preprocess_text(text):
    sentences = sent_tokenize(text)
    cleaned_sentences = [clean_sentence(sentence) for sentence in sentences]
    return cleaned_sentences

def clean_sentence(sentence):
    tokens = word_tokenize(sentence)
    tokens = [token.lower() for token in tokens if token.isalnum()]
    return " ".join(tokens)

