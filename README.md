The purpose of this project is to test and compare two different seq2seq machine learning architectures. 

Both of these architectures utilize an RNN-based decoder, the difference between them lies in how they each
generate the initial hidden state for this decoder. One takes a more standard approach and runs the data through 
an LSTM model, using the final hidden state of this model as the initial hidden state of the decoder. The 
second uses a convolutional-based model as the encoder. This hybrid model passes the input sequence through 
a 1-D convolutional layer, followed by a 1-D max pooling layer. A grid search was used to tune the 
hyperparameters.

The data used to train this model comes from Tatoeba: J. Tiedemann, 2012, Parallel Data, Tools and Interfaces in OPUS. In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC 2012)
In order to replicate the results, the data must first be retrieved from: https://opus.nlpl.eu/datasets/Tatoeba?pair=en&fr
Note that as of project completion, the most recent version is v2026-07-08.
