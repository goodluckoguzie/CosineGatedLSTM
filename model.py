import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from typing import Optional, Tuple


class RAUCell(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers=1,dropout=0):
        super(RAUCell, self).__init__()
        # Initialize GRU layer
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True,num_layers=num_layers,dropout=dropout)
        # Initialize parameters for attention
        # Weights for computing attention
        self.weight_c = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.bias_c = nn.Parameter(torch.Tensor(hidden_size))
        self.input_mapped = nn.Linear(input_size, hidden_size)

        self.weight_hat_h = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.bias_hat_h = nn.Parameter(torch.Tensor(hidden_size))

        # Reset parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weights and biases
        nn.init.kaiming_uniform_(self.weight_c, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_hat_h, a=math.sqrt(5))
        nn.init.zeros_(self.bias_c)
        nn.init.zeros_(self.bias_hat_h)

    def create_prv_output(self, output):
        batch_size, seq_len, hidden_size = output.shape
        zero_tensor = torch.zeros(batch_size, 1, hidden_size, device=output.device)
        prv_output = torch.cat((zero_tensor, output[:, :-1, :]), dim=1)
        return prv_output

    def forward(self, x):

        output, hidden = self.gru(x)
        # Replace the previous method with the new method to get prv_output
        prv_output = self.create_prv_output(output)

        # Concatenate x and hidden for computing attention
        combined = torch.cat((x, prv_output), dim=2)  # This concatenates along the feature dimension
    
        c_t = torch.tanh(F.linear(combined, self.weight_c, self.bias_c))
        a_t = F.softmax(c_t, dim=2)
        hat_h_t = torch.tanh(F.linear(combined, self.weight_hat_h, self.bias_hat_h))
        hat_h_t = a_t * hat_h_t

        # Combine GRU output with attention-based modifications
        output = output + hat_h_t


        return output,hidden



class CGLSTMCellv1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(CGLSTMCellv1, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.input_mapped = nn.Linear(input_size, hidden_size)
        # self.out_mapped = nn.Linear(hidden_size * 2, hidden_size)
        self.out_mapped = nn.Linear(hidden_size + hidden_size , hidden_size)

    def create_prv_output(self, output):
        batch_size, seq_len, hidden_size = output.shape
        zero_tensor = torch.zeros(batch_size, 1, hidden_size, device=output.device)
        prv_output = torch.cat((zero_tensor, output[:, :-1, :]), dim=1)
        return prv_output
    def forward(self, x):
        # Map input for cosine similarity calculation
        input_mapped = self.input_mapped(x)
        output, hidden = self.lstm(x)  # Output: [batch_size, seq_len, hidden_size], Hidden: [num_layers, batch_size, hidden_size]


        prv_output = self.create_prv_output(output)
        gate_ic =F.cosine_similarity(input_mapped, prv_output, dim=1, eps=1e-12).unsqueeze(1)
        gate_co = F.cosine_similarity(input_mapped, output, dim=1, eps=1e-12).unsqueeze(1)
        ht = ((input_mapped*gate_ic) + output )* gate_co 

        ht = self.out_mapped(torch.cat((ht, output), dim=2)) * gate_co

        return  ht
    
    
class CGLSTMCellv0(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(CGLSTMCellv0, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.input_mapped = nn.Linear(input_size, hidden_size)
        # self.out_mapped = nn.Linear(hidden_size * 2, hidden_size)
        self.out_mapped = nn.Linear(hidden_size + hidden_size , hidden_size)

    def create_prv_output(self, output):
        batch_size, seq_len, hidden_size = output.shape
        zero_tensor = torch.zeros(batch_size, 1, hidden_size, device=output.device)
        prv_output = torch.cat((zero_tensor, output[:, :-1, :]), dim=1)
        return prv_output
    def forward(self, x):
        # Map input for cosine similarity calculation
        input_mapped = self.input_mapped(x)
        output, hidden = self.gru(x)  # Output: [batch_size, seq_len, hidden_size], Hidden: [num_layers, batch_size, hidden_size]

        # Use the last layer's hidden state and adjust for batch_first
        # last_layer_hidden = hidden[-1, :, :].unsqueeze(1).expand(-1, x.size(1), -1)
        # # # Average the cell states over the layer dimension
        # # cell_mapped_avg = cn.mean(dim=0)
        # # cell_mapped = cell_mapped_avg.unsqueeze(1).expand(-1, x.size(1), -1)
        prv_output = self.create_prv_output(output)
        gate_ic =F.cosine_similarity(input_mapped, prv_output, dim=1, eps=1e-12).unsqueeze(1)
        # gate_ic = torch.sigmoid((gate_ic + 1) / 2)
        at_co = F.cosine_similarity(input_mapped, output, dim=1, eps=1e-12).unsqueeze(1)
        # at_co = torch.sigmoid((at_co + 1) / 2)
        ht = ((input_mapped*gate_ic) + output )* at_co 
        # # ht = self.out_mapped(torch.cat((ht,input_mapped, output), dim=2))
        # ht = self.out_mapped(torch.cat((ht,x, output), dim=2))
        ht = self.out_mapped(torch.cat((ht, output), dim=2)) * at_co
        # # ht = self.out_mapped(torch.cat((ht, output), dim=2)) + output

        return  ht
    
    

#####################################################Transformer #############################################

from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0,nhead=5):

        super(TransformerModel, self).__init__()
        
        # Defining the model's parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Linear layer to get the inputs to the correct size
        self.input_linear = nn.Linear(input_size, hidden_size)

        # Define the Transformer layers
        # transformer_layer = nn.TransformerEncoderLayer(
        #     d_model=hidden_size, 
        #     nhead=nhead
        # )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead,
            dropout=dropout, 
            # batch_first=True  # Add this line if your PyTorch version supports it
        )

        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers)

        # Output layer to map the hidden representation to the desired output size
        self.output_linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Transform input to the correct dimension
        x = self.input_linear(x)
        
        # For Transformer, we need (sequence_length, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)
        
        # Pass the sequence through the Transformer layers
        transformer_out = self.transformer_encoder(x)
        
        # Transform output to the desired size
        output = transformer_out.permute(1, 0, 2)

        return output

####################################################     ADDING PROBLEM  #######################################################################

class AP_RecurrentModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model_type='GRU'):
        super(AP_RecurrentModel, self).__init__()
        self.hidden_size = hidden_size
        self.model_type = model_type
        if model_type == 'GRU':
            self.recurrent_layer = nn.GRU(input_size, hidden_size, batch_first=True)
        elif model_type == 'LSTM':
            self.recurrent_layer = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif model_type == 'RAUCell':
            self.recurrent_layer = RAUCell(input_size, hidden_size)
        elif model_type == 'CGGRU':
            self.recurrent_layer = CGLSTMCellv0(input_size, hidden_size)
        elif model_type == 'CGLSTM':
            self.recurrent_layer = CGLSTMCellv1(input_size, hidden_size)
        elif model_type == 'Transformer':
            self.recurrent_layer = TransformerModel(input_size, hidden_size, num_layers=1, dropout=0.0)
        else:
            raise ValueError("Invalid model type. Choose 'GRU', 'LSTM', 'RAUCell', 'CGLSTMCellv0', 'CGLSTMCellv1', or 'Transformer'.")

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if self.model_type in ['GRU', 'LSTM','RAUCell']:
            output, _ = self.recurrent_layer(x)
            last_output = output[:, -1, :]
        elif self.model_type in ['CGLSTM','CGGRU', 'Transformer']:
            # Adjust these according to the actual behavior of these models
            output = self.recurrent_layer(x)
            last_output = output[:, -1, :]

        else:
            raise ValueError("Model type not supported in forward method.")

        out = self.fc(last_output)
        return out

####################################################     FASHION-MNIST  #######################################################################

class FM_RecurrentModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model_type='GRU'):
        super(FM_RecurrentModel, self).__init__()
        self.hidden_size = hidden_size
        self.model_type = model_type

        if model_type == 'GRU':
            self.recurrent_layer = nn.GRU(input_size, hidden_size,batch_first=True)
        elif model_type == 'LSTM':
            self.recurrent_layer = nn.LSTM(input_size, hidden_size,batch_first=True)
        elif model_type == 'RAUCell':
            self.recurrent_layer = RAUCell(input_size, hidden_size)
        elif model_type == 'CGGRU':
            self.recurrent_layer = CGLSTMCellv0(input_size, hidden_size)
        elif model_type == 'CGLSTM':
            self.recurrent_layer = CGLSTMCellv1(input_size, hidden_size)
        elif model_type == 'Transformer':
            # Assuming num_layers is fixed for simplicity, adjust as needed
            self.recurrent_layer = TransformerModel(input_size, hidden_size, num_layers=1, dropout=0)
        else:
            raise ValueError("Invalid model type. Choose 'GRUCell', 'LSTMCell', 'RAUCell', 'CGLSTMCellv0', 'CGLSTMCellv1', or 'Transformer'.")

        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        if self.model_type in ['GRU', 'LSTM', 'RAUCell']:
            output, _ = self.recurrent_layer(x)
            last_output = output[:, -1, :]
            # print(last_output.shape)

        elif self.model_type in ['CGLSTM','CGGRU', 'Transformer']:
            # Adjust these according to the actual behavior of these models
            output = self.recurrent_layer(x)

            last_output = output[:, -1, :]

        else:
            raise ValueError("Model type not supported in forward method.")

        # out = self.fc(last_output)
        # return out
        out = self.fc(last_output)
        return out



#############################################################ROW-WISE###################################################


class RowWise_RecurrentModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model_type='GRU'):
        super(RowWise_RecurrentModel, self).__init__()
        self.hidden_size = hidden_size
        self.model_type = model_type

        if model_type == 'GRU':
            self.recurrent_layer = nn.GRU(input_size, hidden_size,batch_first=True)
        elif model_type == 'LSTM':
            self.recurrent_layer = nn.LSTM(input_size, hidden_size,batch_first=True)
        elif model_type == 'RAUCell':
            self.recurrent_layer = RAUCell(input_size, hidden_size)
        elif model_type == 'CGGRU':
            self.recurrent_layer = CGLSTMCellv0(input_size, hidden_size)
        elif model_type == 'CGLSTM':
            self.recurrent_layer = CGLSTMCellv1(input_size, hidden_size)
        elif model_type == 'Transformer':
            # Assuming num_layers is fixed for simplicity, adjust as needed
            self.recurrent_layer = TransformerModel(input_size, hidden_size, num_layers=1, dropout=0)
        else:
            raise ValueError("Invalid model type. Choose 'GRUCell', 'LSTMCell', 'RAUCell', 'CGLSTMCellv0', 'CGLSTMCellv1', or 'Transformer'.")

        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        if self.model_type in ['GRU', 'LSTM','RAUCell']:
            output, _ = self.recurrent_layer(x)
            last_output = output[:, -1, :]
            # print(last_output.shape)

        elif self.model_type in ['CGLSTM','CGGRU', 'Transformer']:
            # Adjust these according to the actual behavior of these models
            output = self.recurrent_layer(x)

            last_output = output[:, -1, :]

        else:
            raise ValueError("Model type not supported in forward method.")

        # out = self.fc(last_output)
        # return out
        out = self.fc(last_output)
        return out
################################################ Sentiment Analysis #############################################


class SA_RecurrentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, model_type='GRU', dropout=0.5):
        super(SA_RecurrentModel, self).__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.dropout = dropout

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if model_type == 'RAUCell':
            self.recurrent_layer = RAUCell(embedding_dim, hidden_size, num_layers=2,dropout=dropout)
        elif model_type == 'LSTM':
            self.recurrent_layer = nn.LSTM(embedding_dim, hidden_size,batch_first=True, num_layers=2,dropout=dropout)
        elif model_type == 'CGGRU':
            self.recurrent_layer = CGLSTMCellv0(embedding_dim, hidden_size, num_layers=2,dropout= self.dropout)
        elif model_type == 'CGLSTM':
            self.recurrent_layer = CGLSTMCellv1(embedding_dim, hidden_size, num_layers=2,dropout= self.dropout)
        elif model_type == 'GRU':
            self.recurrent_layer = nn.GRU(embedding_dim, hidden_size,batch_first=True, num_layers=2,dropout=dropout)
        elif model_type == 'Transformer':
            # Assuming num_layers is fixed for simplicity, adjust as needed
            self.recurrent_layer = TransformerModel(embedding_dim, hidden_size, num_layers=2, dropout=dropout)
        else:
            raise ValueError("Invalid model type. Choose among 'GRUCell', 'LSTMCell', 'RAUCell', 'CGLSTMCellv0', 'CGLSTMCellv1', or 'Transformer'.")

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Sigmoid activation function for binary classification
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        embeds = self.embedding(x)

        if self.model_type in ['GRU', 'LSTM','RAUCell']:
            output, _ = self.recurrent_layer(embeds)


        elif self.model_type in ['CGLSTM','CGGRU', 'Transformer']:
            # Adjust these according to the actual behavior of these models
            output = self.recurrent_layer(embeds)

        else:
            raise ValueError("Model type not supported in forward method.")

        last_output = output[:, -1]

        out = self.sigmoid(self.fc(last_output))
        # out = self.fc(last_output)
        return out
#######################################  LanguageModel #########################


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, model_type='GRU'):
        super(LanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.model_type = model_type
        self.hidden_dim =hidden_dim

        # Initialize the appropriate model based on model_type
        if model_type == 'RAUCell':
            self.rnn = RAUCell(embedding_dim, hidden_dim)
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim,batch_first=True)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim,batch_first=True)
        elif model_type == 'CGGRU':
            self.rnn = CGLSTMCellv0(embedding_dim, hidden_dim)
        elif model_type == 'CGLSTM':
            self.rnn = CGLSTMCellv1(embedding_dim, hidden_dim)
        elif model_type == 'Transformer':
            # For the Transformer, hidden_dim is used as d_model
            self.rnn = TransformerModel(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout_rate)
        else:
            raise ValueError("Invalid model type. Choose 'GRUCell', 'LSTMCell', 'RAUCell', 'CGLSTMCellv0', 'CGLSTMCellv1', or 'Transformer'.")

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))

        if self.model_type in ['CGLSTM','CGGRU', 'Transformer']:
            if embedded.dim() == 2:
                embedded = embedded.unsqueeze(1)  # Add a sequence length of 1
            output = self.rnn(embedded)
            batch_size, seq_len, _ = output.size()  

        elif self.model_type in ['LSTM','GRU','RAUCell']:
            if embedded.dim() == 2:
                embedded = embedded.unsqueeze(1)  # Add a sequence length of 1
            output,_ = self.rnn(embedded)
            batch_size, seq_len, _ = output.size() 
        else:
            print("choose the right model")
        #     if embedded.dim() == 2:
        #         embedded = embedded.unsqueeze(1)  # Add a sequence length of 1
        #     batch_size, seq_len, _ = embedded.size()

        #     if isinstance(self.rnn, (RAUCell)):
        #         hidden = torch.zeros(batch_size, self.rnn.hidden_size, device=text.device)
        #     output = []
        #     for timestep in range(seq_len):
        #         if isinstance(self.rnn, (RAUCell)):
        #             hidden = self.rnn(embedded[:, timestep, :], hidden)
        #             output.append(hidden.unsqueeze(1))
            # if self.model_type == 'RAUCell':
            #     output = torch.cat(output, dim=1)
            #     output = self.dropout(output)

        # Apply the fully connected layer to the output
        decoded = self.fc(output.reshape(-1, self.hidden_dim if self.model_type in ['LSTM','GRU','CGLSTM','CGGRU', 'Transformer','RAUCell'] else self.rnn.hidden_size))
        return decoded.view(-1 if self.model_type in ['LSTM','GRU','CGLSTM','CGGRU', 'Transformer','RAUCell'] else batch_size, seq_len, self.vocab_size)





# class CGLSTMCellv0(nn.Module):
#     """
#     A custom LSTM cell implementation that integrates cosine similarity-based gating mechanisms,
#     with the option to apply layer normalization for each gate within the cell. This cell is designed
#     to process sequences of data by taking into account the similarity between different states and inputs,
#     potentially improving the model's capacity to learn from complex temporal dynamics.

#     Parameters:
#     - input_size (int): The number of expected features in the input `x`
#     - hidden_size (int): The number of features in the hidden state `h`
#     - bias (bool, optional): If `False`, the layer will not use bias weights. Default is `True`.
#     - batch_first (bool, optional): If `True`, the input and output tensors are provided
#       as (batch, seq, feature) instead of (seq, batch, feature). Default is `False`.
#     - layer_norm (bool, optional): If `True`, applies layer normalization to the gates,
#       helping to stabilize the learning process. Default is `True`.
#     """

#     def __init__(self, input_size: int, hidden_size: int, bias: bool = True, batch_first: bool = True, layer_norm: bool = True) -> None:
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.bias = bias
#         self.batch_first = batch_first
        
#         # Linear layer to transform concatenated input and hidden state into gate values
#         self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)
        
#         # Conditional initialization of layer normalization modules or identity modules
#         # for each gate based on the layer_norm flag
#         if layer_norm:
#             self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
#         else:
#             self.layer_norm = nn.ModuleList([nn.Identity() for _ in range(4)])
        
#         # Linear layer to map input to hidden size, facilitating cosine similarity comparisons
#         self.input_mapped = nn.Linear(input_size, hidden_size)

#     def forward(self, input: Tensor, states: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
#         """
#         Defines the computation performed at each call, processing the input tensor and
#         previous states to produce the next hidden state and cell state.

#         Parameters:
#         - input (Tensor): The input sequence to the LSTM cell.
#         - states (Optional[Tuple[Tensor, Tensor]]): A tuple containing the initial hidden state
#           and the initial cell state. If `None`, both states are initialized to zeros.

#         Returns:
#         - Tuple[Tensor, Tensor]: The next hidden state and cell state as a tuple.
#         """
#         # Unpack or initialize hidden and cell states
#         hx, cx = states if states is not None else (None, None)
        
#         if hx is None:
#             hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
#         if cx is None:
#             cx = torch.zeros_like(hx)
        
#         # Map input for use in cosine similarity calculations
#         input_mapped = self.input_mapped(input)
#         # Concatenate input and hidden state, then apply linear transformation for gate computations
#         gates = self.linear(torch.cat([input, hx], dim=1))

#         # Split the result into four parts for the LSTM gates and apply layer normalization
#         i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)
#         i_gate = self.layer_norm[0](i_gate)
#         f_gate = self.layer_norm[1](f_gate)
#         g_gate = self.layer_norm[2](g_gate)
#         o_gate = self.layer_norm[3](o_gate)

#         # Apply sigmoid or tanh activations to gate outputs
#         i_gate = torch.sigmoid(i_gate)
#         f_gate = torch.sigmoid(f_gate)
#         g_gate = torch.tanh(g_gate)
#         o_gate = torch.sigmoid(o_gate)

#         # Update the cell state and hidden state
#         cx = f_gate * cx + i_gate * g_gate
#         hx = o_gate * torch.tanh(cx)
        
#         # Calculate the average cell state for use in cosine similarity
#         cell_mapped_avg = cx.mean(dim=0, keepdim=True).expand_as(cx)
        
#         # Compute cosine similarity gates for input-cell average and input-hidden comparisons
#         gate_ic = F.cosine_similarity(input_mapped, cell_mapped_avg, dim=1, eps=1e-6).unsqueeze(1)
#         gate_co = F.cosine_similarity(input_mapped, hx, dim=1, eps=1e-6).unsqueeze(1)

#         # Normalize and apply sigmoid to similarity scores to modulate the final output
#         gate_ic = torch.sigmoid((gate_ic + 1) / 2)
#         gate_co = torch.sigmoid((gate_co + 1) / 2)

#         # Combine modulated hidden states as the final output
#         hx_modulated = hx * gate_ic + hx * gate_co

#         return hx_modulated, cx



# class CGLSTMCellv1(nn.Module):
#     """
#     Custom LSTM Cell with Cosine Gate and Layer Normalization.

#     This LSTM cell introduces a cosine similarity-based gating mechanism to modulate the input and hidden states' interaction. Additionally, it incorporates layer normalization for stabilizing the training process.

#     Parameters:
#     - input_size: The number of expected features in the input `x`
#     - hidden_size: The number of features in the hidden state `h`
#     - bias: If `False`, then the layer does not use bias weights `b_ih` and `b_hh`. Default: `True`
#     - batch_first: If `True`, then the input and output tensors are provided as (batch, seq, feature). Default: `False`
#     - layer_norm: If `True`, applies layer normalization to each gate. Default: `True`
#     """

#     def __init__(self, input_size: int, hidden_size: int, bias: bool = True, batch_first: bool = True, layer_norm: bool = False) -> None:
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.bias = bias
#         self.batch_first = batch_first
#         # Linear transformation combining input and hidden state
#         self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)
        
#         # Conditional layer normalization or identity based on `layer_norm` flag
#         if layer_norm:
#             self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
#         else:
#             self.layer_norm = nn.ModuleList([nn.Identity() for _ in range(4)])
        
#         # Linear mapping of input to hidden size for cosine similarity
#         self.input_mapped = nn.Linear(input_size, hidden_size)

#     def forward(self, input: Tensor, states: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
#         """
#         Defines the computation performed at every call.

#         Applies the LSTM cell operation with an additional cosine similarity-based gating mechanism to modulate the inputs and the hidden state output.

#         Parameters:
#         - input: The input tensor containing features
#         - states: The tuple of tensors containing the initial hidden state and the initial cell state. If `None`, both `hx` and `cx` are initialized as zeros.

#         Returns:
#         - Tuple containing the output hidden state and the new cell state.
#         """
#         hx, cx = states if states is not None else (None, None)
        
#         # Initialize hidden and cell states if not provided
#         if hx is None:
#             hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
#         if cx is None:
#             cx = torch.zeros_like(hx)
        
#         # Map input for cosine similarity calculation
#         input_mapped = self.input_mapped(input)

#         # Calculate attention weights using cosine similarity between mapped input and hidden state
#         gate_ic = F.cosine_similarity(input_mapped, hx, dim=1, eps=1e-6).unsqueeze(1)
#         attention_weights = torch.sigmoid(gate_ic)
#         # Modulate input with attention weights
#         input = input + (attention_weights * input)

#         # Compute gate values
#         gates = self.linear(torch.cat([input, hx], dim=1))

#         # Split the concatenated gates and apply layer normalization
#         i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)
#         i_gate = self.layer_norm[0](i_gate)
#         f_gate = self.layer_norm[1](f_gate)
#         g_gate = self.layer_norm[2](g_gate)
#         o_gate = self.layer_norm[3](o_gate)

#         # Apply non-linear activations
#         i_gate = torch.sigmoid(i_gate)
#         f_gate = torch.sigmoid(f_gate)
#         g_gate = torch.tanh(g_gate)
#         o_gate = torch.sigmoid(o_gate)

#         # Update cell state and hidden state
#         cx = f_gate * cx + i_gate * g_gate
#         hx = o_gate * torch.tanh(cx)

#         # Cosine similarity for modulating the final output
#         gate_ic = F.cosine_similarity(hx, cx, dim=1, eps=1e-6).unsqueeze(1)
#         gate_co = torch.sigmoid((gate_ic + 1) / 2)
#         hx_modulated = hx + (hx * gate_co)

#         return hx_modulated, cx



# class RAUCell(nn.Module):
#     """
#     Recurrent Attention Unit (RAU) cell.

#     This class implements a variation of the GRU cell, incorporating an additional
#     attention mechanism. The RAU cell computes attention-based hidden states 
#     alongside the standard reset and update gates of a GRU.
#     """
#     def __init__(self, input_size, hidden_size):
#         """
#         Initializes the RAUCell.

#         Args:
#             input_size (int): The number of expected features in the input `x`.
#             hidden_size (int): The number of features in the hidden state.
#         """
#         super(RAUCell, self).__init__()
#         self.hidden_size = hidden_size
#         self.input_size = input_size

#         # Weights for computing reset and update gates
#         self.weight_xr = nn.Parameter(torch.Tensor(hidden_size, input_size))
#         self.weight_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.bias_r = nn.Parameter(torch.Tensor(hidden_size))
        
#         self.weight_xz = nn.Parameter(torch.Tensor(hidden_size, input_size))
#         self.weight_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.bias_z = nn.Parameter(torch.Tensor(hidden_size))
        
#         # Weights for computing candidate hidden state
#         self.weight_xh = nn.Parameter(torch.Tensor(hidden_size, input_size))
#         self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.bias_h = nn.Parameter(torch.Tensor(hidden_size))
        
#         # Weights for computing attention
#         self.weight_c = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
#         self.bias_c = nn.Parameter(torch.Tensor(hidden_size))
        
#         self.weight_hat_h = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
#         self.bias_hat_h = nn.Parameter(torch.Tensor(hidden_size))

#         self.reset_parameters()

#     def reset_parameters(self):
#         """
#         Initializes weights using a uniform distribution with range based on hidden size.
#         """
#         stdv = 1.0 / self.hidden_size ** 0.5
#         for weight in self.parameters():
#             nn.init.uniform_(weight, -stdv, stdv)

#     def forward(self, x, hidden):
#         """
#         Defines the forward pass of the RAUCell.

#         Args:
#             x (Tensor): The input tensor at the current time step.
#             hidden (Tensor): The hidden state tensor from the previous time step.

#         Returns:
#             Tensor: The updated hidden state.
#         """
#         # Concatenate x and hidden for computing attention
#         combined = torch.cat((x, hidden), 1)
        
#         # Compute attention weights
#         c_t = F.linear(combined, self.weight_c, self.bias_c)
#         a_t = F.softmax(c_t, dim=1)
        
#         # Compute attention-based hidden state
#         hat_h_t = F.relu(F.linear(combined, self.weight_hat_h, self.bias_hat_h))
#         hat_h_t = a_t * hat_h_t
        
#         # Compute reset and update gates
#         r_t = torch.sigmoid(F.linear(x, self.weight_xr, self.bias_r) + F.linear(hidden, self.weight_hr))
#         z_t = torch.sigmoid(F.linear(x, self.weight_xz, self.bias_z) + F.linear(hidden, self.weight_hz))
        
        # # Compute candidate hidden state
        # h_tilde = torch.tanh(F.linear(x, self.weight_xh, self.bias_h) + F.linear(r_t * hidden, self.weight_hh))
        
#         # Compute the final hidden state
#         h_t = (1 - z_t) * h_tilde + z_t * hidden + hat_h_t

#         return h_t





########################################################################
# class CGLSTMCellv1(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1,dropout=0):
#         super(CGLSTMCellv1, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.input_mapped = nn.Linear(input_size, hidden_size)
#         self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True,num_layers=num_layers,dropout=dropout)
        
#         for name, param in self.lstm.named_parameters():
#             if 'weight_ih' in name:
#                 nn.init.xavier_uniform_(param.data)
#             elif 'weight_hh' in name:
#                 nn.init.xavier_uniform_(param.data)
#             elif 'bias' in name:
#                 param.data.fill_(0)
#     def create_prv_output(self, output):
#         batch_size, seq_len, hidden_size = output.shape
#         zero_tensor = torch.zeros(batch_size, 1, hidden_size, device=output.device)
#         prv_output = torch.cat((zero_tensor, output[:, :-1, :]), dim=1)
#         return prv_output

#     def forward(self, x):
#         # Map input for cosine similarity calculation
#         input_mapped = self.input_mapped(x)
#         # Initialize hidden state and cell state
#         # hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         hidden_initialized = torch.zeros(input_mapped.size(0), input_mapped.size(1), input_mapped.size(2)).to(x.device)


#         # # Calculate the average cell state for use in cosine similarity
#         # hidden_mapped_avg = hidden.mean(dim=0).unsqueeze(1).expand(-1, input_mapped.size(1), -1)

#         # # Adjust hidden to match input_mapped dimensions. Use only the last layer's hidden state.
#         # # Note: You might want to use a different strategy depending on your model's requirements.
#         # hidden_adjusted = hidden_mapped_avg  # Taking the last layer's hidden state

#         # Calculate attention weights using cosine similarity
#         # Ensure hidden is adjusted to have the same dimensions as input_mapped for cosine similarity calculation.
#         gate_ic = F.cosine_similarity(input_mapped, hidden_initialized, dim=2, eps=1e-6).unsqueeze(-1)
#         attention_weights = torch.sigmoid(gate_ic)
        
#         # Modulate input with attention weights
#         input_modulated = input_mapped + (attention_weights * input_mapped)
        
#         # Proceed with LSTM
#         lstm_out, (hn, cn) = self.lstm(input_modulated)
        
#         # Calculate the average cell state for use in cosine similarity
#         cell_mapped_avg = cn.mean(dim=0).unsqueeze(1).expand(-1, input_mapped.size(1), -1)
#             # Replace the previous method with the new method to get prv_output
#         prv_output = self.create_prv_output(lstm_out)


#         # Compute cosine similarity gates
#         # gate_co = F.cosine_similarity(cell_mapped_avg, lstm_out, dim=2, eps=1e-6).unsqueeze(-1)
#         gate_co = F.cosine_similarity(prv_output, lstm_out, dim=2, eps=1e-6).unsqueeze(-1)
        
#         # Normalize and apply sigmoid
#         gate_co = torch.sigmoid((gate_co + 1) / 2)
        
#         # Combine modulated hidden states as the final output
#         output = lstm_out * gate_co
        
#         return output



        
# class CGLSTMCellv1(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1):
#         super(CGLSTMCellv1, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.input_mapped = nn.Linear(input_size, hidden_size)
#         self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
#         for name, param in self.lstm.named_parameters():
#             if 'weight_ih' in name:
#                 nn.init.xavier_uniform_(param.data)
#             elif 'weight_hh' in name:
#                 nn.init.xavier_uniform_(param.data)
#             elif 'bias' in name:
#                 param.data.fill_(0)
        
#     def forward(self, x):
#         # Initialize hidden state and cell state
#         hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
#         # Map input for cosine similarity calculation
#         input_mapped = self.input_mapped(x)
#         # Calculate the average cell state for use in cosine similarity
#         hidden_mapped_avg = hidden.mean(dim=0).unsqueeze(1).expand(-1, input_mapped.size(1), -1)

#         # Adjust hidden to match input_mapped dimensions. Use only the last layer's hidden state.
#         # Note: You might want to use a different strategy depending on your model's requirements.
#         hidden_adjusted = hidden_mapped_avg  # Taking the last layer's hidden state

#         # Calculate attention weights using cosine similarity
#         # Ensure hidden is adjusted to have the same dimensions as input_mapped for cosine similarity calculation.
#         gate_ic = F.cosine_similarity(input_mapped, hidden_adjusted, dim=2, eps=1e-6).unsqueeze(-1)
#         attention_weights = torch.sigmoid(gate_ic)
        
#         # Modulate input with attention weights
#         input_modulated = input_mapped + (attention_weights * input_mapped)
        
#         # Proceed with LSTM
#         lstm_out, (hn, cn) = self.lstm(input_modulated)
        
#         # Calculate the average cell state for use in cosine similarity
#         cell_mapped_avg = cn.mean(dim=0).unsqueeze(1).expand(-1, input_mapped.size(1), -1)
        
#         # Compute cosine similarity gates
#         gate_co = F.cosine_similarity(cell_mapped_avg, lstm_out, dim=2, eps=1e-6).unsqueeze(-1)
        
#         # Normalize and apply sigmoid
#         gate_co = torch.sigmoid((gate_co + 1) / 2)
        
#         # Combine modulated hidden states as the final output
#         output = lstm_out * gate_co
        
#         return output

# ############################################################################


# class CGLSTMCellv0(nn.Module):
#     def __init__(self,n_latents, hidden_size, num_layers=1,dropout=0):
#         super(CGLSTMCellv0, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         # Linear mapping of input to hidden size for cosine similarity
#         self.input_mapped = nn.Linear(n_latents, hidden_size)
#         # self.input_mapped = nn.Embedding(n_latents, hidden_size)

#         # Basic LSTM layer
#         self.lstm = nn.LSTM(n_latents, hidden_size, batch_first=True,num_layers=num_layers,dropout=dropout)
        
#         # Weights for computing attention
#         self.weight_c = nn.Parameter(torch.Tensor(hidden_size, n_latents + hidden_size))
#         self.bias_c = nn.Parameter(torch.Tensor(hidden_size))
        
#         self.weight_hat_h = nn.Parameter(torch.Tensor(hidden_size, n_latents + hidden_size))
#         self.bias_hat_h = nn.Parameter(torch.Tensor(hidden_size))
#         self.transformation_layer = nn.Linear(hidden_size * 2, hidden_size)

#         # Apply Xavier/Glorot initialization to LSTM weights and zero initialization to biases
#         for name, param in self.lstm.named_parameters():
#             if 'weight_ih' in name:
#                 nn.init.xavier_uniform_(param.data)
#             elif 'weight_hh' in name:
#                 nn.init.xavier_uniform_(param.data)
#             elif 'bias' in name:
#                 param.data.fill_(0)
        
#         # Reset parameters
#         self.reset_parameters()
    
#     def reset_parameters(self):
#         # Initialize weights and biases
#         nn.init.kaiming_uniform_(self.weight_c, a=math.sqrt(5))
#         nn.init.kaiming_uniform_(self.weight_hat_h, a=math.sqrt(5))
#         nn.init.zeros_(self.bias_c)
#         nn.init.zeros_(self.bias_hat_h)

#     def create_prv_output(self, output):
#         batch_size, seq_len, hidden_size = output.shape
#         zero_tensor = torch.zeros(batch_size, 1, hidden_size, device=output.device)
#         prv_output = torch.cat((zero_tensor, output[:, :-1, :]), dim=1)
#         return prv_output



#     def forward(self, x):
#         # Map input for cosine similarity calculation
#         input_mapped = self.input_mapped(x).to(x.device)

#         output, (hn, cn)  = self.lstm(x)
#             # Replace the previous method with the new method to get prv_output
#         prv_output = self.create_prv_output(output)


#         # Concatenate x and hidden for computing attention
#         combined = torch.cat((x, prv_output), dim=2)  # This concatenates along the feature dimension
    
#         # Compute cosine similarity gates for input-cell average and input-hidden comparisons
#         gate_ic = F.cosine_similarity(input_mapped, output, dim=1, eps=1e-6).unsqueeze(1)
#         gate_co = F.cosine_similarity(prv_output, output, dim=1, eps=1e-6).unsqueeze(1)

#        # Normalize and apply sigmoid to similarity scores to modulate the final output
#         gate_ic = torch.sigmoid((gate_ic + 1) / 2)
#         gate_co = torch.sigmoid((gate_co + 1) / 2)

#         c_t = torch.tanh(F.linear(combined, self.weight_c, self.bias_c))
#         a_t = F.softmax(c_t, dim=2)
#         hat_h_t = torch.tanh(F.linear(combined, self.weight_hat_h, self.bias_hat_h))
#         hat_h_t = a_t * hat_h_t 

#         # Assume gate_weights are learned parameters or computed based on some function of gate_ic and gate_co
#         hat_h_t =  F.selu(gate_ic * hat_h_t + (1 - gate_co) * hat_h_t)

#         # Concatenation
#         combined = torch.cat([output, hat_h_t], dim=-1)

#         Foutput = self.transformation_layer(combined)

#         return Foutput


# class CGLSTMCellv0(nn.Module):
#     def __init__(self,n_latents, hidden_size, num_layers=1,dropout=0):
#         super(CGLSTMCellv0, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         # Linear mapping of input to hidden size for cosine similarity
#         self.input_mapped = nn.Linear(n_latents, hidden_size)

#         # Basic LSTM layer
#         self.lstm = nn.LSTM(n_latents, hidden_size, batch_first=True,num_layers=num_layers,dropout=dropout)
        
#         # Weights for computing attention
#         self.weight_c = nn.Parameter(torch.Tensor(hidden_size, n_latents + hidden_size+ hidden_size))
#         self.bias_c = nn.Parameter(torch.Tensor(hidden_size))
        
#         self.weight_hat_h = nn.Parameter(torch.Tensor(hidden_size, n_latents + hidden_size+ hidden_size))
#         self.bias_hat_h = nn.Parameter(torch.Tensor(hidden_size))
#         self.transformation_layer = nn.Linear(hidden_size * 2, hidden_size)


#         # Apply Xavier/Glorot initialization to LSTM weights and zero initialization to biases
#         for name, param in self.lstm.named_parameters():
#             if 'weight_ih' in name:
#                 nn.init.xavier_uniform_(param.data)
#             elif 'weight_hh' in name:
#                 nn.init.xavier_uniform_(param.data)
#             elif 'bias' in name:
#                 param.data.fill_(0)
        
#         # Reset parameters
#         self.reset_parameters()
    
#     def reset_parameters(self):
#         # Initialize weights and biases
#         nn.init.kaiming_uniform_(self.weight_c, a=math.sqrt(5))
#         nn.init.kaiming_uniform_(self.weight_hat_h, a=math.sqrt(5))
#         nn.init.zeros_(self.bias_c)
#         nn.init.zeros_(self.bias_hat_h)


#     def forward(self, x):
#         # Map input for cosine similarity calculation
#         input_mapped = self.input_mapped(x).to(x.device)

#         output, (hn, cn)  = self.lstm(x)

        # # Average the cell states over the layer dimension
        # cell_mapped_avg = cn.mean(dim=0)
        # cell_mapped = cell_mapped_avg.unsqueeze(1).expand(-1, x.size(1), -1)

#         # Compute cosine similarity gates for input-cell average and input-hidden comparisons
#         gate_ic = F.cosine_similarity(input_mapped, cell_mapped, dim=1, eps=1e-6).unsqueeze(1)
#         gate_co = F.cosine_similarity(input_mapped, output, dim=1, eps=1e-6).unsqueeze(1)

#        # Normalize and apply sigmoid to similarity scores to modulate the final output
#         gate_ic = torch.sigmoid((gate_ic + 1) / 2)
#         gate_co = torch.sigmoid((gate_co + 1) / 2)
#         # Apply learnable weights to the gates
#         gate_ic_weighted = gate_ic * input_mapped
#         gate_co_weighted = gate_co * output

#         # Concatenate x and hidden for computing attention
#         combined = torch.cat((x, gate_ic_weighted,gate_co_weighted), dim=2)  

#         gate_ic = torch.tanh(F.linear(combined, self.weight_c, self.bias_c))
#         a_t = F.softmax(gate_ic, dim=2)
#         gate_co = torch.tanh(F.linear(combined, self.weight_hat_h, self.bias_hat_h))


#         hat_h_t = F.relu(a_t * gate_co)

#         # Concatenation
#         combined = torch.cat([output, hat_h_t], dim=-1)

#         Foutput = self.transformation_layer(combined)

#         return Foutput

# class CGLSTMCellv1(nn.Module):
#     def __init__(self, input_size, hidden_size,num_layers=1,dropout=0):
#         super(CGLSTMCellv1, self).__init__()
#         # Initialize GRU layer
#         self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True,num_layers=num_layers,dropout=dropout)
#         # Linear mapping of input to hidden size for cosine similarity
#         self.input_mapped = nn.Linear(input_size, hidden_size)

#         # Weights for computing attention
#         self.weight_c = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size+ hidden_size))
#         self.bias_c = nn.Parameter(torch.Tensor(hidden_size))
        
#         self.weight_hat_h = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size+ hidden_size))
#         self.bias_hat_h = nn.Parameter(torch.Tensor(hidden_size))
#         self.transformation_layer = nn.Linear(hidden_size * 2, hidden_size)

#         # Reset parameters
#         self.reset_parameters()
    
#     def reset_parameters(self):
#         # Initialize weights and biases
#         nn.init.kaiming_uniform_(self.weight_c, a=math.sqrt(5))
#         nn.init.kaiming_uniform_(self.weight_hat_h, a=math.sqrt(5))
#         nn.init.zeros_(self.bias_c)
#         nn.init.zeros_(self.bias_hat_h)

#     def forward(self, x):
#         # Map input for cosine similarity calculation
#         input_mapped = self.input_mapped(x)
#         output, cn = self.gru(input_mapped)
#         # Average the cell states over the layer dimension
#         cell_mapped_avg = cn.mean(dim=0)
#         cell_mapped = cell_mapped_avg.unsqueeze(1).expand(-1, x.size(1), -1)
#         # Compute cosine similarity gates for input-cell average and input-hidden comparisons
#         gate_ic = F.cosine_similarity(input_mapped, cell_mapped, dim=1, eps=1e-6).unsqueeze(1)
#         gate_co = F.cosine_similarity(input_mapped, output, dim=1, eps=1e-6).unsqueeze(1)
#        # Normalize and apply sigmoid to similarity scores to modulate the final output
#         gate_ic = torch.sigmoid((gate_ic + 1) / 2)
#         gate_co = torch.sigmoid((gate_co + 1) / 2)
#         # Apply learnable weights to the gates
#         gate_ic_weighted = gate_ic * input_mapped
#         gate_co_weighted = gate_co * output
#         # Concatenate x and hidden for computing attention
#         combined = torch.cat((x, gate_ic_weighted,gate_co_weighted), dim=2)  
#         gate_ic = torch.tanh(F.linear(combined, self.weight_c, self.bias_c))
#         a_t = F.softmax(gate_ic, dim=2)
#         gate_co = torch.tanh(F.linear(combined, self.weight_hat_h, self.bias_hat_h))
#         hat_h_t = F.relu(a_t * gate_co)
#         # Concatenation
#         combined = torch.cat([output, hat_h_t], dim=-1)
#         Foutput = self.transformation_layer(combined)
#         return Foutput


#################

####################################################################
# class TransformerModel(nn.Module):
#     def __init__(self, input, hidden_size, num_layers, dropout=0.0):
#         super(TransformerModel, self).__init__()
#         self.pos_encoder = PositionalEncoding(hidden_size)

#         self.input_linear = nn.Linear(input, hidden_size)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1, dropout=dropout)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
#         self.init_weights()

#     def init_weights(self):
#         initrange = 0.1
#         self.input_linear.bias.data.zero_()
#         self.input_linear.weight.data.uniform_(-initrange, initrange)

#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask

#     def forward(self, src):
#         device = src.device
#         src = self.input_linear(src)
#         src = src.permute(1, 0, 2)
#         src = self.pos_encoder(src)
#         mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
#         output = self.transformer_encoder(src, mask)
#         output = output.permute(1, 0, 2)
#         return output

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=10_000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return x
