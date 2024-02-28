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
    """
    Recurrent Attention Unit (RAU) cell.

    This class implements a variation of the GRU cell, incorporating an additional
    attention mechanism. The RAU cell computes attention-based hidden states 
    alongside the standard reset and update gates of a GRU.
    """
    def __init__(self, input_size, hidden_size):
        """
        Initializes the RAUCell.

        Args:
            input_size (int): The number of expected features in the input `x`.
            hidden_size (int): The number of features in the hidden state.
        """
        super(RAUCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # Weights for computing reset and update gates
        self.weight_xr = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_r = nn.Parameter(torch.Tensor(hidden_size))
        
        self.weight_xz = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_z = nn.Parameter(torch.Tensor(hidden_size))
        
        # Weights for computing candidate hidden state
        self.weight_xh = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_h = nn.Parameter(torch.Tensor(hidden_size))
        
        # Weights for computing attention
        self.weight_c = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.bias_c = nn.Parameter(torch.Tensor(hidden_size))
        
        self.weight_hat_h = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.bias_hat_h = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes weights using a uniform distribution with range based on hidden size.
        """
        stdv = 1.0 / self.hidden_size ** 0.5
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, hidden):
        """
        Defines the forward pass of the RAUCell.

        Args:
            x (Tensor): The input tensor at the current time step.
            hidden (Tensor): The hidden state tensor from the previous time step.

        Returns:
            Tensor: The updated hidden state.
        """
        # Concatenate x and hidden for computing attention
        combined = torch.cat((x, hidden), 1)
        
        # Compute attention weights
        c_t = F.linear(combined, self.weight_c, self.bias_c)
        a_t = F.softmax(c_t, dim=1)
        
        # Compute attention-based hidden state
        hat_h_t = F.relu(F.linear(combined, self.weight_hat_h, self.bias_hat_h))
        hat_h_t = a_t * hat_h_t
        
        # Compute reset and update gates
        r_t = torch.sigmoid(F.linear(x, self.weight_xr, self.bias_r) + F.linear(hidden, self.weight_hr))
        z_t = torch.sigmoid(F.linear(x, self.weight_xz, self.bias_z) + F.linear(hidden, self.weight_hz))
        
        # Compute candidate hidden state
        h_tilde = torch.tanh(F.linear(x, self.weight_xh, self.bias_h) + F.linear(r_t * hidden, self.weight_hh))
        
        # Compute the final hidden state
        h_t = (1 - z_t) * h_tilde + z_t * hidden + hat_h_t

        return h_t



class LSTMCell(nn.Module):
    """
    An LSTM cell with Layer Normalization.

    This LSTM cell version applies LayerNorm after the projection of the inputs and before the gate activations,
    which can help stabilize the learning process by normalizing the inputs to have zero mean and unit variance.

    Args:
        input_size (int): The input size.
        hidden_size (int): The hidden state size.
        bias (bool, optional): Whether to apply a bias to the input projection. Defaults to True.
        batch_first (bool, optional): Whether the first dimension represents the batch dimension. Defaults to False.
        layer_norm (bool, optional): Whether to apply LayerNorm after the input projection. Defaults to True.
    """
    def __init__(
        self, input_size: int, hidden_size: int, bias: bool = True, batch_first: bool = False, layer_norm: bool = True
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        # LSTM uses 4 * hidden_size because it has 4 gates (input, forget, cell, and output)
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)
        
        if layer_norm:
            # Applying LayerNorm separately for each gate
            self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
        else:
            self.layer_norm = nn.ModuleList([nn.Identity() for _ in range(4)])

    def forward(self, input: Tensor, states: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        hx, cx = states if states is not None else (None, None)
        is_3d = input.dim() == 3
        if is_3d:
            if self.batch_first:
                input = input.squeeze(1)
            else:
                input = input.squeeze(0)

        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        if cx is None:
            cx = torch.zeros_like(hx)

        gates = self.linear(torch.cat([input, hx], dim=1))

        # Split the gates
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)
        
        # Apply Layer Normalization to each gate separately
        i_gate = self.layer_norm[0](i_gate)
        f_gate = self.layer_norm[1](f_gate)
        g_gate = self.layer_norm[2](g_gate)
        o_gate = self.layer_norm[3](o_gate)

        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)

        cx = f_gate * cx + i_gate * g_gate
        hx = o_gate * torch.tanh(cx)

        return hx, cx



class CGLSTMCellv0(nn.Module):
    """
    A custom LSTM cell implementation that integrates cosine similarity-based gating mechanisms,
    with the option to apply layer normalization for each gate within the cell. This cell is designed
    to process sequences of data by taking into account the similarity between different states and inputs,
    potentially improving the model's capacity to learn from complex temporal dynamics.

    Parameters:
    - input_size (int): The number of expected features in the input `x`
    - hidden_size (int): The number of features in the hidden state `h`
    - bias (bool, optional): If `False`, the layer will not use bias weights. Default is `True`.
    - batch_first (bool, optional): If `True`, the input and output tensors are provided
      as (batch, seq, feature) instead of (seq, batch, feature). Default is `False`.
    - layer_norm (bool, optional): If `True`, applies layer normalization to the gates,
      helping to stabilize the learning process. Default is `True`.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, batch_first: bool = False, layer_norm: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        
        # Linear layer to transform concatenated input and hidden state into gate values
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)
        
        # Conditional initialization of layer normalization modules or identity modules
        # for each gate based on the layer_norm flag
        if layer_norm:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
        else:
            self.layer_norm = nn.ModuleList([nn.Identity() for _ in range(4)])
        
        # Linear layer to map input to hidden size, facilitating cosine similarity comparisons
        self.input_mapped = nn.Linear(input_size, hidden_size)

    def forward(self, input: Tensor, states: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        """
        Defines the computation performed at each call, processing the input tensor and
        previous states to produce the next hidden state and cell state.

        Parameters:
        - input (Tensor): The input sequence to the LSTM cell.
        - states (Optional[Tuple[Tensor, Tensor]]): A tuple containing the initial hidden state
          and the initial cell state. If `None`, both states are initialized to zeros.

        Returns:
        - Tuple[Tensor, Tensor]: The next hidden state and cell state as a tuple.
        """
        # Unpack or initialize hidden and cell states
        hx, cx = states if states is not None else (None, None)
        
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        if cx is None:
            cx = torch.zeros_like(hx)
        
        # Map input for use in cosine similarity calculations
        input_mapped = self.input_mapped(input)
        # Concatenate input and hidden state, then apply linear transformation for gate computations
        gates = self.linear(torch.cat([input, hx], dim=1))

        # Split the result into four parts for the LSTM gates and apply layer normalization
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)
        i_gate = self.layer_norm[0](i_gate)
        f_gate = self.layer_norm[1](f_gate)
        g_gate = self.layer_norm[2](g_gate)
        o_gate = self.layer_norm[3](o_gate)

        # Apply sigmoid or tanh activations to gate outputs
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)

        # Update the cell state and hidden state
        cx = f_gate * cx + i_gate * g_gate
        hx = o_gate * torch.tanh(cx)
        
        # Calculate the average cell state for use in cosine similarity
        cell_mapped_avg = cx.mean(dim=0, keepdim=True).expand_as(cx)
        
        # Compute cosine similarity gates for input-cell average and input-hidden comparisons
        gate_ic = F.cosine_similarity(input_mapped, cell_mapped_avg, dim=1, eps=1e-6).unsqueeze(1)
        gate_co = F.cosine_similarity(input_mapped, hx, dim=1, eps=1e-6).unsqueeze(1)

        # Normalize and apply sigmoid to similarity scores to modulate the final output
        gate_ic = torch.sigmoid((gate_ic + 1) / 2)
        gate_co = torch.sigmoid((gate_co + 1) / 2)

        # Combine modulated hidden states as the final output
        hx_modulated = hx * gate_ic + hx * gate_co

        return hx_modulated, cx



class CGLSTMCellv1(nn.Module):
    """
    Custom LSTM Cell with Cosine Gate and Layer Normalization.

    This LSTM cell introduces a cosine similarity-based gating mechanism to modulate the input and hidden states' interaction. Additionally, it incorporates layer normalization for stabilizing the training process.

    Parameters:
    - input_size: The number of expected features in the input `x`
    - hidden_size: The number of features in the hidden state `h`
    - bias: If `False`, then the layer does not use bias weights `b_ih` and `b_hh`. Default: `True`
    - batch_first: If `True`, then the input and output tensors are provided as (batch, seq, feature). Default: `False`
    - layer_norm: If `True`, applies layer normalization to each gate. Default: `True`
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, batch_first: bool = False, layer_norm: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        # Linear transformation combining input and hidden state
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)
        
        # Conditional layer normalization or identity based on `layer_norm` flag
        if layer_norm:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
        else:
            self.layer_norm = nn.ModuleList([nn.Identity() for _ in range(4)])
        
        # Linear mapping of input to hidden size for cosine similarity
        self.input_mapped = nn.Linear(input_size, hidden_size)

    def forward(self, input: Tensor, states: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        """
        Defines the computation performed at every call.

        Applies the LSTM cell operation with an additional cosine similarity-based gating mechanism to modulate the inputs and the hidden state output.

        Parameters:
        - input: The input tensor containing features
        - states: The tuple of tensors containing the initial hidden state and the initial cell state. If `None`, both `hx` and `cx` are initialized as zeros.

        Returns:
        - Tuple containing the output hidden state and the new cell state.
        """
        hx, cx = states if states is not None else (None, None)
        
        # Initialize hidden and cell states if not provided
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        if cx is None:
            cx = torch.zeros_like(hx)
        
        # Map input for cosine similarity calculation
        input_mapped = self.input_mapped(input)

        # Calculate attention weights using cosine similarity between mapped input and hidden state
        gate_ic = F.cosine_similarity(input_mapped, hx, dim=1, eps=1e-6).unsqueeze(1)
        attention_weights = torch.sigmoid(gate_ic)
        # Modulate input with attention weights
        input = input + (attention_weights * input)

        # Compute gate values
        gates = self.linear(torch.cat([input, hx], dim=1))

        # Split the concatenated gates and apply layer normalization
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)
        i_gate = self.layer_norm[0](i_gate)
        f_gate = self.layer_norm[1](f_gate)
        g_gate = self.layer_norm[2](g_gate)
        o_gate = self.layer_norm[3](o_gate)

        # Apply non-linear activations
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)

        # Update cell state and hidden state
        cx = f_gate * cx + i_gate * g_gate
        hx = o_gate * torch.tanh(cx)

        # Cosine similarity for modulating the final output
        gate_ic = F.cosine_similarity(hx, cx, dim=1, eps=1e-6).unsqueeze(1)
        gate_co = torch.sigmoid((gate_ic + 1) / 2)
        hx_modulated = hx + (hx * gate_co)

        return hx_modulated, cx


class GRUCell(nn.Module):
    """
    Recurrent Attention Unit (RAU) cell.

    This class implements a variation of the GRU cell, incorporating an additional
    attention mechanism. The RAU cell computes attention-based hidden states 
    alongside the standard reset and update gates of a GRU.
    """
    def __init__(self, input_size, hidden_size):
        """
        Initializes the GRU.

        Args:
            input_size (int): The number of expected features in the input `x`.
            hidden_size (int): The number of features in the hidden state.
        """
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # Weights for computing reset and update gates
        self.weight_xr = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_r = nn.Parameter(torch.Tensor(hidden_size))
        
        self.weight_xz = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_z = nn.Parameter(torch.Tensor(hidden_size))
        
        # Weights for computing candidate hidden state
        self.weight_xh = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_h = nn.Parameter(torch.Tensor(hidden_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes weights using a uniform distribution with range based on hidden size.
        """
        stdv = 1.0 / self.hidden_size ** 0.5
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, hidden):
        """
        Defines the forward pass of the GRU.

        Args:
            x (Tensor): The input tensor at the current time step.
            hidden (Tensor): The hidden state tensor from the previous time step.

        Returns:
            Tensor: The updated hidden state.
        """

        # Compute reset and update gates
        r_t = torch.sigmoid(F.linear(x, self.weight_xr, self.bias_r) + F.linear(hidden, self.weight_hr))
        z_t = torch.sigmoid(F.linear(x, self.weight_xz, self.bias_z) + F.linear(hidden, self.weight_hz))
        
        # Compute candidate hidden state
        h_tilde = torch.tanh(F.linear(x, self.weight_xh, self.bias_h) + F.linear(r_t * hidden, self.weight_hh))
        
        # Compute the final hidden state
        h_t = (1 - z_t) * h_tilde + z_t * hidden 

        return h_t



######################################################################Transformer #############################################

from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class TransformerModel(nn.Module):
    def __init__(self, input, hidden_size, num_layers, dropout=0.0):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(hidden_size)

        self.input_linear = nn.Linear(input , hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_linear.bias.data.zero_()
        self.input_linear.weight.data.uniform_(-initrange, initrange)

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
        output = output.permute(1, 0, 2)

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
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

####################################################     ADDING PROBLEM  #######################################################################


class AP_RecurrentModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model_type='GRUCell'):
        super(AP_RecurrentModel, self).__init__()
        self.hidden_size = hidden_size
        self.model_type = model_type
        if model_type == 'GRUCell':
            self.recurrent_layer = GRUCell(input_size, hidden_size)
        elif model_type == 'LSTMCell':
            self.recurrent_layer = LSTMCell(input_size, hidden_size)
        elif model_type == 'RAUCell':
            self.recurrent_layer = RAUCell(input_size, hidden_size)
        elif model_type == 'CGLSTMCellv0':
            self.recurrent_layer = CGLSTMCellv0(input_size, hidden_size)
        elif model_type == 'CGLSTMCellv1':
            self.recurrent_layer = CGLSTMCellv1(input_size, hidden_size)
        elif model_type == 'Transformer':
            self.recurrent_layer = TransformerModel(input_size, hidden_size, num_layers=1, dropout=0.1)
        else:
            raise ValueError("Invalid model type. Choose 'GRUCell', 'LSTMCell', 'RAUCell', 'CGLSTMCellv0', 'CGLSTMCellv1', or 'Transformer'.")

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if self.model_type in ['GRUCell', 'RAUCell', 'CGLSTMCellv0', 'CGLSTMCellv1']:
            h = torch.zeros(x.size(0), self.hidden_size).to(x.device)
            for t in range(x.size(1)):
                if self.model_type in ['CGLSTMCellv0', 'CGLSTMCellv1']:
                    h, _ = self.recurrent_layer(x[:, t, :], (h, torch.zeros_like(h)))
                else:
                    h = self.recurrent_layer(x[:, t, :], h)
            last_output = h
        elif self.model_type == 'LSTMCell':
            h, c = torch.zeros(x.size(0), self.hidden_size).to(x.device), torch.zeros(x.size(0), self.hidden_size).to(x.device)
            for t in range(x.size(1)):
                h, c = self.recurrent_layer(x[:, t, :], (h, c))
            last_output = h
        elif self.model_type == 'Transformer':
            output = self.recurrent_layer(x)
            last_output = output[:, -1, :]  # Take the output of the last token for classification/regression tasks
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

        # Initialize your recurrent layers here based on model_type
        if model_type == 'GRUCell':
            self.recurrent_layer = nn.GRUCell(input_size, hidden_size)
        elif model_type == 'LSTMCell':
            self.recurrent_layer = nn.LSTMCell(input_size, hidden_size)
        elif model_type == 'RAUCell':
            self.recurrent_layer = RAUCell(input_size, hidden_size)
        elif model_type == 'CGLSTMCellv0':
            self.recurrent_layer = CGLSTMCellv0(input_size, hidden_size)
        elif model_type == 'CGLSTMCellv1':
            self.recurrent_layer = CGLSTMCellv1(input_size, hidden_size)
        elif model_type == 'Transformer':
            self.transformer = TransformerModel(input_size, hidden_size, num_layers=1, dropout=0)
        else:
            raise ValueError("Invalid model type. Choose 'GRUCell', 'LSTMCell', 'RAUCell', 'CGLSTMCellv0', 'CGLSTMCellv1', or 'Transformer'.")

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Adjust handling for LSTM and CGLSTM variants
        if self.model_type in ['GRUCell', 'RAUCell']:
            h = torch.zeros(x.size(0), self.hidden_size).to(x.device)
            for t in range(x.size(1)):
                h = self.recurrent_layer(x[:, t, :], h)
            last_output = h
        elif self.model_type in ['LSTMCell', 'CGLSTMCellv0', 'CGLSTMCellv1']:
            h, c = (torch.zeros(x.size(0), self.hidden_size).to(x.device) for _ in range(2))
            for t in range(x.size(1)):
                h, c = self.recurrent_layer(x[:, t, :], (h, c))
            last_output = h
        elif self.model_type == 'Transformer':
            output = self.transformer(x)
            last_output = output[:, -1, :]  # Take the output of the last token for classification/regression tasks
        else:
            raise ValueError("Unexpected model type encountered.")

        out = self.fc(last_output)
        return out



#############################################################ROW-WISE###################################################


class RowWise_RecurrentModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model_type='GRU'):
        super(RowWise_RecurrentModel, self).__init__()
        self.hidden_size = hidden_size
        self.model_type = model_type

        if model_type == 'GRUCell':
            self.recurrent_layer = GRUCell(input_size, hidden_size)
        elif model_type == 'LSTMCell':
            self.recurrent_layer = LSTMCell(input_size, hidden_size)
        elif model_type == 'RAUCell':
            self.recurrent_layer = RAUCell(input_size, hidden_size)
        elif model_type == 'CGLSTMCellv0':
            self.recurrent_layer = CGLSTMCellv0(input_size, hidden_size)
        elif model_type == 'CGLSTMCellv1':
            self.recurrent_layer = CGLSTMCellv1(input_size, hidden_size)
        elif model_type == 'Transformer':
            # Assuming num_layers is fixed for simplicity, adjust as needed
            self.recurrent_layer = TransformerModel(input_size, hidden_size, num_layers=1, dropout=0)
        else:
            raise ValueError("Invalid model type. Choose 'GRUCell', 'LSTMCell', 'RAUCell', 'CGLSTMCellv0', 'CGLSTMCellv1', or 'Transformer'.")

        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        if self.model_type in ['GRUCell', 'RAUCell', 'LSTMCell', 'CGLSTMCellv0', 'CGLSTMCellv1']:
            # RNN-based models are handled here as before
            if self.model_type in ['LSTMCell', 'CGLSTMCellv0', 'CGLSTMCellv1']:
                h = torch.zeros(x.size(0), self.hidden_size).to(x.device)
                c = torch.zeros_like(h)
                for t in range(x.size(1)):
                    h, c = self.recurrent_layer(x[:, t, :], (h, c))
                last_output = h
            else: # GRUCell, RAUCell
                h = torch.zeros(x.size(0), self.hidden_size).to(x.device)
                for t in range(x.size(1)):
                    h = self.recurrent_layer(x[:, t, :], h)
                last_output = h
        elif self.model_type == 'Transformer':
            output = self.recurrent_layer(x)
            # Assuming we're interested in the last output for tasks like classification
            last_output = output[:, -1, :]  
        else:
            raise ValueError("Unexpected model type encountered.")

        out = self.fc(last_output)
        return out


################################################ Sentiment Analysis #############################################


class SA_RecurrentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, model_type='GRU', dropout=0.5):
        super(SA_RecurrentModel, self).__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if model_type == 'RAUCell':
            self.recurrent_layer = RAUCell(embedding_dim, hidden_size)
        elif model_type == 'LSTMCell':
            self.recurrent_layer = LSTMCell(embedding_dim, hidden_size)
        elif model_type == 'CGLSTMCellv0':
            self.recurrent_layer = CGLSTMCellv0(embedding_dim, hidden_size)
        elif model_type == 'CGLSTMCellv1':
            self.recurrent_layer = CGLSTMCellv1(embedding_dim, hidden_size)
        elif model_type == 'GRUCell':
            self.recurrent_layer = GRUCell(embedding_dim, hidden_size)
        elif model_type == 'Transformer':
            # Assuming num_layers is fixed for simplicity, adjust as needed
            self.recurrent_layer = TransformerModel(embedding_dim, hidden_size, num_layers=1, dropout=dropout)
        else:
            raise ValueError("Invalid model type. Choose among 'GRUCell', 'LSTMCell', 'RAUCell', 'CGLSTMCellv0', 'CGLSTMCellv1', or 'Transformer'.")

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Sigmoid activation function for binary classification
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        embeds = self.embedding(x)

        if self.model_type in ['RAUCell', 'GRUCell', 'LSTMCell', 'CGLSTMCellv0', 'CGLSTMCellv1']:
            # Initialize hidden state (and cell state for LSTM variants)
            h = torch.zeros(x.size(0), self.hidden_size).to(x.device)
            if self.model_type in ['LSTMCell', 'CGLSTMCellv0', 'CGLSTMCellv1']:
                c = torch.zeros_like(h)
            rnn_outs = []
            for t in range(x.size(1)):
                if self.model_type in ['LSTMCell', 'CGLSTMCellv0', 'CGLSTMCellv1']:
                    h, c = self.recurrent_layer(embeds[:, t], (h, c))
                else:  # RAUCell, GRUCell
                    h = self.recurrent_layer(embeds[:, t], h)
                rnn_outs.append(h.unsqueeze(1))
            output = torch.cat(rnn_outs, dim=1)
        elif self.model_type == 'Transformer':
            output = self.recurrent_layer(embeds)
        else:
            raise ValueError("Unexpected model type encountered.")
        last_output = output[:, -1]

        out = self.sigmoid(self.fc(last_output))
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
        elif model_type == 'LSTMCell':
            self.rnn = LSTMCell(embedding_dim, hidden_dim)
        elif model_type == 'GRUCell':
            self.rnn = GRUCell(embedding_dim, hidden_dim)
        elif model_type == 'CGLSTMCellv0':
            self.rnn = CGLSTMCellv0(embedding_dim, hidden_dim)
        elif model_type == 'CGLSTMCellv1':
            self.rnn = CGLSTMCellv1(embedding_dim, hidden_dim)
        elif model_type == 'Transformer':
            # For the Transformer, hidden_dim is used as d_model
            self.transformer = TransformerModel(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout_rate)
        else:
            raise ValueError("Invalid model type. Choose 'GRUCell', 'LSTMCell', 'RAUCell', 'CGLSTMCellv0', 'CGLSTMCellv1', or 'Transformer'.")

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))

        if self.model_type == 'Transformer':
            if embedded.dim() == 2:
                embedded = embedded.unsqueeze(1)  # Add a sequence length of 1
            output = self.transformer(embedded)
            batch_size, seq_len, _ = output.size()  
        else:
            if embedded.dim() == 2:
                embedded = embedded.unsqueeze(1)  # Add a sequence length of 1
            batch_size, seq_len, _ = embedded.size()

            if isinstance(self.rnn, (RAUCell, GRUCell)):
                hidden = torch.zeros(batch_size, self.rnn.hidden_size, device=text.device)
            elif isinstance(self.rnn, (LSTMCell, CGLSTMCellv0, CGLSTMCellv1)):
                hx = torch.zeros(batch_size, self.rnn.hidden_size, device=text.device)
                cx = torch.zeros(batch_size, self.rnn.hidden_size, device=text.device)

            output = []
            for timestep in range(seq_len):
                if isinstance(self.rnn, (RAUCell, GRUCell)):
                    hidden = self.rnn(embedded[:, timestep, :], hidden)
                    output.append(hidden.unsqueeze(1))
                elif isinstance(self.rnn, (LSTMCell, CGLSTMCellv0, CGLSTMCellv1)):
                    hx, cx = self.rnn(embedded[:, timestep, :], (hx, cx))
                    output.append(hx.unsqueeze(1))
            if self.model_type != 'Transformer':
                output = torch.cat(output, dim=1)
                output = self.dropout(output)

        # Apply the fully connected layer to the output
        decoded = self.fc(output.reshape(-1, self.hidden_dim if self.model_type == 'Transformer' else self.rnn.hidden_size))
        return decoded.view(-1 if self.model_type == 'Transformer' else batch_size, seq_len, self.vocab_size)
