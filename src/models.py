import torch
import torch.nn as nn
from collections import OrderedDict

# This class is based on the one found in the PyTorch documentation
# S. Robertson. "NLP From Scratch: Translation with a Sequence to Sequence Network and Attention." 
# PyTorch Documentation. https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html.
class LSTM2LSTM(nn.Module):

    def __init__(self,
                 input_space,
                 output_space,
                 hidden_space,
                 layers=1,
                 max_words=15,
                 sos_token_id=0,
                 device="cpu",
                 #tf_probability=0.2,
                 activation=nn.LeakyReLU()):

        super().__init__()

        # Initialize numeric/str fields
        self.device = device
        self.max_words = max_words
        self.sos_token_id = sos_token_id
        #self.tf_probability = tf_probability

        # Encoder - LSTM
        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    ("embedding", nn.Embedding(input_space, hidden_space)),
                    ("lstm", nn.LSTM(hidden_space, hidden_space, layers, batch_first=True))
                ]
            )
        )

        # Decoder - LSTM
        self.decoder = nn.Sequential(
            OrderedDict(
                [
                    ("embedding", nn.Embedding(output_space, hidden_space)),
                    ("lstm", nn.LSTM(hidden_space, hidden_space, layers, batch_first=True)),
                    ("activation", activation),
                    ("linear", nn.Linear(hidden_space, output_space)),
                    ("out", nn.Softmax(dim=-1))
                ]
            )
        )

    def forward(self, input_tens):

        # Check teacher forcing
        #if isinstance(src[0], dict):

            # If a dictionary was passed, teacher forcing is enabled
        #    input_tens = torch.tensor([src_entry["src"] for src_entry in src], dtype=torch.long)
        #    target_tens = torch.tensor([src_entry["trg"] for src_entry in src], dtype=torch.long)

        #else:

            # Else, teacher forcing is disabled
        #    input_tens = torch.tensor(src, dtype=torch.long)
        #    target_tens = None

        # Determine batch size
        batch_size = input_tens.size(0)

        # Run input through encoder
        enc_out, enc_hx = self.encoder.forward(input_tens)

        # Initialize decoder inputs
        dec_next_input = torch.full((batch_size, 1), self.sos_token_id).to(self.device)
        dec_next_hx = enc_hx

        # Initialize output vector
        output_vector = []

        for i in range(0, self.max_words):

            # Determine if teacher forcing will be used for this character
            #use_tf = torch.rand(1)[0] <= self.tf_probability

            #print(dec_next_input)

            # Run the previous input and hidden state through the decoder once
            dec_output = self.decoder.embedding.forward(dec_next_input)
            dec_output, dec_hidden = self.decoder.lstm.forward(dec_output, dec_next_hx)
            dec_output = self.decoder.activation.forward(dec_output)
            dec_output = self.decoder.linear.forward(dec_output)
            dec_output = self.decoder.out.forward(dec_output)

            #print(dec_output)

            # Save the probabilities for the next word
            output_vector.append(dec_output)

            # Determine next input (tf or not)
            #if use_tf and target_tens != None:

                # Feed target next time
            #    dec_next_input = target_tens[:, 1].unsqueeze(-1)

            #else:

            # Determine the most probable next word as a number
            dec_next_input = torch.argmax(dec_output, dim=2)

            # Next hidden state is the last hidden state
            dec_next_hx = dec_hidden

        # Return sequence of probability predictions
        outputs = torch.cat(output_vector, dim=1)
        return outputs

# This class is based on the one found in the PyTorch documentation
# S. Robertson. "NLP From Scratch: Translation with a Sequence to Sequence Network and Attention." 
# PyTorch Documentation. https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html.
# This one builds on the design by adding an alternative encoder.
class Conv2LSTM(nn.Module):

    def __init__(self,
                 input_space,
                 output_space,
                 hidden_space,
                 conv_kernel_size=1,
                 pool_kernel_size=1,
                 layers=1,
                 max_words=15,
                 sos_token_id=0,
                 device="cpu",
                 #tf_probability=0.2,
                 activation=nn.LeakyReLU()):

        super().__init__()

        # Initialize numeric/str fields
        self.device = device
        self.max_words = max_words
        self.sos_token_id = sos_token_id
        #self.tf_probability = tf_probability

        # Encoder - Convolutional
        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    ("embedding", nn.Embedding(input_space, hidden_space)),
                    ("conv", nn.Conv1d(max_words, 1, conv_kernel_size)),
                    ("pool", nn.MaxPool1d(pool_kernel_size, stride=1))
                ]
            )
        )

        # Decoder - LSTM
        self.decoder = nn.Sequential(
            OrderedDict(
                [
                    ("embedding", nn.Embedding(output_space, hidden_space)),
                    ("lstm", nn.LSTM(hidden_space, hidden_space, layers, batch_first=True)),
                    ("activation", activation),
                    ("linear", nn.Linear(hidden_space, output_space)),
                    ("out", nn.Softmax(dim=-1))
                ]
            )
        )

    def forward(self, input_tens):

        # Check teacher forcing
        #if isinstance(src[0], dict):

            # If a dictionary was passed, teacher forcing is enabled
        #    input_tens = torch.tensor([src_entry["src"] for src_entry in src], dtype=torch.long)
        #    target_tens = torch.tensor([src_entry["trg"] for src_entry in src], dtype=torch.long)

        #else:

            # Else, teacher forcing is disabled
        #    input_tens = torch.tensor(src, dtype=torch.long)
        #    target_tens = None

        # Determine batch size
        batch_size = input_tens.size(0)

        # Run input through encoder
        enc_hx = self.encoder.forward(input_tens).swapdims(0, 1)
        enc_hx = (enc_hx, enc_hx)

        # Initialize decoder inputs
        dec_next_input = torch.full((batch_size, 1), self.sos_token_id).to(self.device)
        dec_next_hx = enc_hx

        # Initialize output vector
        output_vector = []

        for i in range(0, self.max_words):

            # Determine if teacher forcing will be used for this character
            #use_tf = torch.rand(1)[0] <= self.tf_probability

            #print(dec_next_input)

            # Run the previous input and hidden state through the decoder once
            dec_output = self.decoder.embedding.forward(dec_next_input)
            dec_output, dec_hidden = self.decoder.lstm.forward(dec_output, dec_next_hx)
            dec_output = self.decoder.activation.forward(dec_output)
            dec_output = self.decoder.linear.forward(dec_output)
            dec_output = self.decoder.out.forward(dec_output)

            #print(dec_output)

            # Save the probabilities for the next word
            output_vector.append(dec_output)

            # Determine next input (tf or not)
            #if use_tf and target_tens != None:

                # Feed target next time
            #    dec_next_input = target_tens[:, 1].unsqueeze(-1)

            #else:

            # Determine the most probable next word as a number
            dec_next_input = torch.argmax(dec_output, dim=2)

            # Next hidden state is the last hidden state
            dec_next_hx = dec_hidden

        # Return sequence of probability predictions
        outputs = torch.cat(output_vector, dim=1)
        return outputs
