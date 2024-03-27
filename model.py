
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import torch.nn.init as init



class CGLSTM(nn.Module):
    def __init__(self,n_latents, n_actions, hidden_size, num_layers=1, dropout=0):
        super(CGLSTM, self).__init__()
        self.lstm = nn.LSTM(n_latents+ n_actions, hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.input_mapped = nn.Linear(n_latents+ n_actions, hidden_size)
        self.out_mapped = nn.Linear(hidden_size + hidden_size , hidden_size)
        self.output = nn.Linear(hidden_size, n_latents) 
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
        at_co = F.cosine_similarity(input_mapped, output, dim=1, eps=1e-12).unsqueeze(1)
        ht = ((input_mapped*gate_ic) + output )* at_co 
        ht = self.out_mapped(torch.cat((ht, output), dim=2)) * at_co
        ht = self.output(ht) 
        return  ht





#################################  Baseline  #####################################

class LSTMmodel(nn.Module):
    def __init__(self,n_latents, n_actions, hidden_size, num_layers):
        super(LSTMmodel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Basic LSTM layer
        self.lstm = nn.LSTM(n_latents+n_actions, hidden_size, num_layers, batch_first=True)
        
        # Apply Xavier/Glorot initialization to LSTM weights and zero initialization to biases
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        self.output = nn.Linear(hidden_size, n_latents)  # Map LSTM output directly to desired output size
        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.fill_(0)

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        output = self.output(lstm_out)
        return output

    def infer(self, x, hidden=None):
        if hidden is None:
            # Initialize hidden and cell states
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            hidden = (h, c)
        
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.output(lstm_out)
        
        return output, hidden

#######################################   GRU   ##################################################


class GRU(nn.Module):
    def __init__(self, n_latents, n_actions, hidden_size, num_layers=1):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # GRU layer with dropout
        self.gru = nn.GRU(n_latents+n_actions, hidden_size, num_layers, batch_first=True)
        
        # Apply Xavier/Glorot initialization to GRU weights and zero initialization to biases
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        self.output = nn.Linear(hidden_size, n_latents)  # Map GRU output directly to desired output size
        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.fill_(0)

    def forward(self, x):
        # GRU forward pass
        gru_out, _ = self.gru(x)
        # gru_out = gru_out[:, -1, :]  # Get the last output of the sequence

        output = self.output(gru_out)
        return output

    def infer(self, x, hidden=None):
        if hidden is None:
            # Initialize hidden state for GRU
            hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        gru_out, hidden = self.gru(x, hidden)
        output = self.output(gru_out)
        
        return output, hidden


##################################   TransformerModel  ##########################################################################

import math

class TransformerModel(nn.Module):
    def __init__(self, n_latents, n_actions, hidden_size, num_layers, dropout=0.0):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(hidden_size)

        self.input_linear = nn.Linear(n_latents + n_actions, hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(hidden_size, n_latents)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_linear.bias.data.zero_()
        self.input_linear.weight.data.uniform_(-initrange, initrange)
        self.output_linear.bias.data.zero_()
        self.output_linear.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        device = src.device
        src = self.input_linear(src)
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
        output = self.transformer_encoder(src, mask)
        output = self.output_linear(output.permute(1, 0, 2))
        # output = self.output_linear(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=120):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x








# class CGLSTM(nn.Module):
#     def __init__(self,n_latents, n_actions, hidden_size, num_layers):
#         super(CGLSTM, self).__init__()
        
#         self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
#         self._init_weights(self.lstm)
        
#         self.linear_mapping_input = nn.Linear(n_latents+n_actions, hidden_size)
#         self._init_weights(self.linear_mapping_input)
        
#         self.linear_mapping = nn.Linear(hidden_size, hidden_size)
#         self._init_weights(self.linear_mapping)
        
#         self.output = nn.Linear(hidden_size * 2, n_latents)
#         self._init_weights(self.output)

#         # Learnable weights for gates
#         self.gate_ic_weight = nn.Parameter(torch.Tensor(1, 1, hidden_size))
#         self.gate_co_weight = nn.Parameter(torch.Tensor(1, 1, hidden_size))

#         nn.init.xavier_uniform_(self.gate_ic_weight)
#         nn.init.xavier_uniform_(self.gate_co_weight)

#     def forward(self, x):
#         input_mapped = self.linear_mapping_input(x)
#         lstm_out, (h, c) = self.lstm(input_mapped)

#         # Average the cell states over the layer dimension
#         cell_mapped_avg = c.mean(dim=0)
#         cell_mapped_avg = cell_mapped_avg.unsqueeze(1).expand(-1, x.size(1), -1)
#         cell_mapped = self.linear_mapping(cell_mapped_avg)

#         # Calculate the cosine similarities
#         similarity_input_cell = F.cosine_similarity(input_mapped, cell_mapped, dim=-1, eps=1e-6).unsqueeze(2)
#         similarity_output_cell = F.cosine_similarity(lstm_out, cell_mapped, dim=-1, eps=1e-6).unsqueeze(2)

#         # Using the sigmoid function to squash the similarity values between 0 and 1
#         gate_ic = torch.sigmoid(similarity_input_cell)
#         gate_co = torch.sigmoid(similarity_output_cell)

#         # Apply learnable weights to the gates
#         gate_ic_weighted = gate_ic * self.gate_ic_weight
#         gate_co_weighted = gate_co * self.gate_co_weight

#         # Multiplicative combination
#         combined_features = lstm_out * gate_ic_weighted * gate_co_weighted
#         combined_feat = torch.cat((lstm_out, combined_features), dim=2)

#         output = self.output(combined_feat)
#         return output

#     def _init_weights(self, layer):
#         if isinstance(layer, nn.Linear):
#             nn.init.xavier_uniform_(layer.weight)
#             if layer.bias is not None:
#                 nn.init.constant_(layer.bias, 0)
#         elif isinstance(layer, nn.LSTM):
#             for name, param in layer.named_parameters():
#                 if 'weight_ih' in name:
#                     nn.init.xavier_uniform_(param.data)
#                 elif 'weight_hh' in name:
#                     nn.init.orthogonal_(param.data)
#                 elif 'bias' in name:
#                     nn.init.constant_(param.data, 0)

#     def infer(self, state, hidden_state):
#         # Assuming state is the input and hidden_state is (h, c) from an LSTM
#         # hidden_state should be a tuple of (h_0, c_0) for the LSTM
#         # The shape of h_0 and c_0 should be (num_layers, batch, hidden_size)
#         input_mapped = self.linear_mapping_input(state)
#         lstm_out, (h, c) = self.lstm(input_mapped, hidden_state)

#         # Average the cell states over the layer dimension
#         cell_mapped_avg = c.mean(dim=0).unsqueeze(1).expand(-1, state.size(1), -1)
#         cell_mapped = self.linear_mapping(cell_mapped_avg)

#         # Calculate the cosine similarities
#         similarity_input_cell = F.cosine_similarity(input_mapped, cell_mapped, dim=-1, eps=1e-6).unsqueeze(2)
#         similarity_output_cell = F.cosine_similarity(lstm_out, cell_mapped, dim=-1, eps=1e-6).unsqueeze(2)

#         # Using the sigmoid function to squash the similarity values between 0 and 1
#         gate_ic = torch.sigmoid(similarity_input_cell * self.gate_ic_weight)
#         gate_co = torch.sigmoid(similarity_output_cell * self.gate_co_weight)

#         # Multiplicative combination
#         combined_features = lstm_out * gate_ic * gate_co
#         combined_feat = torch.cat((lstm_out, combined_features), dim=2)

#         output = self.output(combined_feat)
#         new_hidden_state = (h, c)

#         return output, new_hidden_state


# class ParallelMultiHeadSimilarityLSTM(nn.Module):
#     def __init__(self, n_latents, n_actions, hidden_size, num_layers, num_heads=4):
#         super(ParallelMultiHeadSimilarityLSTM, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = hidden_size // num_heads
#         assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

#         self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
#         self._init_weights(self.lstm)

#         self.linear_mapping_input = nn.Linear(n_latents + n_actions, hidden_size)
#         self._init_weights(self.linear_mapping_input)

#         self.linear_mapping = nn.Linear(hidden_size, hidden_size)
#         self._init_weights(self.linear_mapping)

#         self.output = nn.Linear(hidden_size * 2, n_latents)
#         self._init_weights(self.output)

#         # Learnable weights for gates, one per head
#         self.gate_weights = nn.Parameter(torch.Tensor(num_heads, 1, 1, self.head_dim))
#         nn.init.xavier_uniform_(self.gate_weights)

#     def forward(self, x):
#         input_mapped = self.linear_mapping_input(x)
#         lstm_out, (h, c) = self.lstm(input_mapped)

#         # Prepare for multi-head computation
#         input_mapped_split = input_mapped.view(input_mapped.shape[0], input_mapped.shape[1], self.num_heads, self.head_dim)
#         lstm_out_split = lstm_out.view(lstm_out.shape[0], lstm_out.shape[1], self.num_heads, self.head_dim)

#         # Average the cell states over the layer dimension and prepare for multi-head
#         cell_mapped_avg = c.mean(dim=0).unsqueeze(1).expand(-1, x.size(1), -1)
#         cell_mapped_split = self.linear_mapping(cell_mapped_avg).view(cell_mapped_avg.shape[0], cell_mapped_avg.shape[1], self.num_heads, self.head_dim)

#         # Parallel computation across heads
#         combined_features = self.parallel_head_processing(input_mapped_split, cell_mapped_split)

#         # Reshape and apply final linear layer
#         combined_features = combined_features.view(combined_features.shape[0], combined_features.shape[1], -1)
#         output = self.output(torch.cat((lstm_out, combined_features), dim=2))
#         return output

#     def parallel_head_processing(self, input_mapped_split, cell_mapped_split):
#         # Process each head in parallel
#         combined_features = []
#         for i in range(self.num_heads):
#             combined_features_head = self.head_processing(input_mapped_split[:,:,i,:], cell_mapped_split[:,:,i,:], self.gate_weights[i])
#             combined_features.append(combined_features_head)

#         # Concatenate the features from all heads
#         combined_features = torch.cat(combined_features, dim=2)
#         return combined_features

#     def head_processing(self, input_mapped_head, cell_mapped_head, gate_weight):
#         # Calculate similarities within the head
#         cosine_similarity = F.cosine_similarity(input_mapped_head, cell_mapped_head, dim=-1, eps=1e-6).unsqueeze(2)
#         euclidean_similarity = self.euclidean_similarity(input_mapped_head, cell_mapped_head).unsqueeze(2)
#         combined_similarity = cosine_similarity * euclidean_similarity

#         # Apply gate using sigmoid function
#         gate = torch.sigmoid(combined_similarity * gate_weight)

#         # Multiplicative combination for the head
#         combined_features_head = input_mapped_head * gate
#         return combined_features_head

#     def euclidean_similarity(self, x1, x2):
#         distance = torch.norm(x1 - x2, dim=-1)
#         similarity = 2 * (1.0 / (1 + distance)) - 1
#         return similarity

#     def _init_weights(self, layer):
#         if isinstance(layer, nn.Linear):
#             nn.init.xavier_uniform_(layer.weight)
#             if layer.bias is not None:
#                 nn.init.constant_(layer.bias, 0)
#         elif isinstance(layer, nn.LSTM):
#             for name, param in layer.named_parameters():
#                 if 'weight_ih' in name:
#                     nn.init.xavier_uniform_(param.data)
#                 elif 'weight_hh' in name:
#                     nn.init.orthogonal_(param.data)
#                 elif 'bias' in name:
#                     nn.init.constant_(param.data, 0)

# class EGLSTM(nn.Module):
#     def __init__(self,n_latents, n_actions, hidden_size, num_layers):
#         super(EGLSTM, self).__init__()
        
#         self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
#         self._init_weights(self.lstm)
        
#         self.linear_mapping_input = nn.Linear(n_latents+n_actions, hidden_size)
#         self._init_weights(self.linear_mapping_input)
        
#         self.linear_mapping = nn.Linear(hidden_size, hidden_size)
#         self._init_weights(self.linear_mapping)
        
#         self.output = nn.Linear(hidden_size * 2, n_latents)
#         self._init_weights(self.output)

#         # Learnable weights for gates
#         self.gate_ic_weight = nn.Parameter(torch.Tensor(1, 1, hidden_size))
#         self.gate_co_weight = nn.Parameter(torch.Tensor(1, 1, hidden_size))

#         nn.init.xavier_uniform_(self.gate_ic_weight)
#         nn.init.xavier_uniform_(self.gate_co_weight)

#     def forward(self, x):
#         input_mapped = self.linear_mapping_input(x)
#         lstm_out, (h, c) = self.lstm(input_mapped)

#         # Average the cell states over the layer dimension
#         cell_mapped_avg = c.mean(dim=0)
#         cell_mapped_avg = cell_mapped_avg.unsqueeze(1).expand(-1, x.size(1), -1)
#         # cell_mapped = self.linear_mapping(cell_mapped_avg)

#         # Calculate the cosine similarities
#         similarity_input_cell = self.euclidean_similarity(input_mapped, cell_mapped_avg)
#         similarity_output_cell = self.euclidean_similarity(lstm_out, cell_mapped_avg)

#         # Using the sigmoid function to squash the similarity values between 0 and 1
#         gate_ic = torch.sigmoid(similarity_input_cell)
#         gate_co = torch.sigmoid(similarity_output_cell)

#         # Apply learnable weights to the gates
#         gate_ic_weighted = gate_ic * self.gate_ic_weight
#         gate_co_weighted = gate_co * self.gate_co_weight

#         # Multiplicative combination
#         combined_features = lstm_out * gate_ic_weighted * gate_co_weighted
#         combined_feat = torch.cat((lstm_out, combined_features), dim=2)

#         output = self.output(combined_feat)
#         return output

#     def euclidean_similarity(self, x1, x2):
#         distance = torch.norm(x1 - x2, dim=-1)
#         similarity = 2 * (1.0 / (1 + distance)) - 1
#         return similarity.unsqueeze(-1)

#     def _init_weights(self, layer):
#         if isinstance(layer, nn.Linear):
#             nn.init.xavier_uniform_(layer.weight)
#             if layer.bias is not None:
#                 nn.init.constant_(layer.bias, 0)
#         elif isinstance(layer, nn.LSTM):
#             for name, param in layer.named_parameters():
#                 if 'weight_ih' in name:
#                     nn.init.xavier_uniform_(param.data)
#                 elif 'weight_hh' in name:
#                     nn.init.orthogonal_(param.data)
#                 elif 'bias' in name:
#                     nn.init.constant_(param.data, 0)




