import torch
from torch.nn.utils.rnn import pad_sequence
from gensim.corpora.dictionary import Dictionary
from gensim.utils import simple_preprocess
from joblib import Parallel, delayed

def preprocess_dataset(X, y, max_words=15):

    # Check for incongruent lengths
    if (len(X) != len(y)): return None, None, None, None

    # Define timeout for parallelization
    to = 10000000

    # Nested function for producing a capped-length tokenized vector
    def get_preprocessed(X : str, y : str, max_words=max_words):

        X_tokens = simple_preprocess(X, min_len=0)
        y_tokens = simple_preprocess(y, min_len=0)

        if (len(X_tokens) > max_words or len(y_tokens) > max_words):
            return None, None
        else:
            return X_tokens, y_tokens

    # Tokenize sentences
    Xy_tokenized = Parallel(n_jobs=6, timeout=to)(delayed(get_preprocessed)(X[i], y[i])
                                                  for i in range(0, len(X)))

    # Filter Nones
    Xy_tokenized = [Xy_tokenized[i] for i in range(0, len(Xy_tokenized)) if Xy_tokenized[i] != (None, None)]

    # Parse resultant list
    X_tokenized = [Xy_tokenized[i][0] for i in range(0, len(Xy_tokenized))]
    y_tokenized = [Xy_tokenized[i][1] for i in range(0, len(Xy_tokenized))]

    # Initialize dictionaries with the start-of-sequence token
    X_dict = Dictionary([['<sos>', '<pad>']])
    y_dict = Dictionary([['<sos>', '<pad>']])

    # Add sentences to dictionaries
    X_dict.add_documents(X_tokenized)
    y_dict.add_documents(y_tokenized)

    # Get token id'ed vectors
    X_preprocessed = Parallel(n_jobs=6, timeout=to)(delayed(X_dict.doc2idx)(X_tokenized[i])
                                                    for i in range(0, len(X_tokenized)))
    y_preprocessed = Parallel(n_jobs=6, timeout=to)(delayed(y_dict.doc2idx)(y_tokenized[i])
                                                    for i in range(0, len(y_tokenized)))

    # Convert to tensors
    X_preprocessed = Parallel(n_jobs=6, timeout=to)(delayed(torch.tensor)(X_preprocessed[i], dtype=torch.long)
                                                    for i in range(0, len(X_preprocessed)))
    y_preprocessed = Parallel(n_jobs=6, timeout=to)(delayed(torch.tensor)(y_preprocessed[i], dtype=torch.long)
                                                    for i in range(0, len(y_preprocessed)))

    # Combine tensors into a single list for consistent padding
    all_sentences = X_preprocessed + y_preprocessed

    # Pad tensors
    all_sentences_pad = pad_sequence(all_sentences, batch_first=True, padding_value=X_dict.token2id.get('<pad>')).to("cpu")

    # Split tensor back into X and Y
    X_preprocessed, y_preprocessed = torch.split(all_sentences_pad, len(X_preprocessed))

    # The long-awaited statement
    return X_preprocessed, y_preprocessed, X_dict, y_dict
