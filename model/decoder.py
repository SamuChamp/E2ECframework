import torch
import torch.nn as nn
import copy


class Decoder(nn.Module):
    def __init__(self, embd_dim, output_dim, max_length=32):
        super(Decoder, self).__init__()
        self.embd_dim = embd_dim
        self.output_dim = output_dim
        self.max_length = max_length
        
        # one-to-one embd for each bit position.
        # each position has 3 embds: one for 0, one for 1, one for -1 (invalid/truncated).
        self.embds = nn.ModuleList([
            nn.Embedding(3, embd_dim) for _ in range(max_length)
        ])
        
        # classifier.
        self.classifier = nn.Sequential(
            nn.Linear(embd_dim, 256),
            nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, output_dim)
        )
        
        # classifier for continuous counterpart.
        self.reg_net = copy.deepcopy(self.classifier)
        
        # initialize weights properly.
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.1)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.1)
    
    def decode_c(self, z_c):
        return self.reg_net(z_c)
    
    def decode(self, z_h):
        batch_size = z_h.size(0)
        
        # initialize the integrated representation.
        x_h = torch.zeros(batch_size, self.embd_dim, device=z_h.device)
        
        # process each bit position.
        for i in range(self.max_length):
            bit_i = z_h[:, i]
            
            # convert bit values to embd indices:
            # -1 -> index 0, 0 -> index 1, 1 -> index 2
            embd_indices = (bit_i + 1).long()
            
            # embd for all positions (including -1s).
            # each position i has 3 embds: 0 for -1, 1 for bit 0, 2 for bit 1.
            embds = self.embds[i](embd_indices)
            
            # let -1 embds be zero.
            mask = (bit_i == -1)
            embds[mask] = 0.0
            
            # repr via inverse transform.
            x_h += embds
            
        return self.classifier(x_h), x_h
    
    def importance_eval(self, z_h, zero_block_idx=None, block_size=4):
        """
        decode with specific bit-block embeddings zeroed out for importance evaluation.
        
        Args:
            z_h: input binary code [batch_size, max_length]
            zero_block_idx: index of the block to zero out (None means no zeroing)
            block_size: number of bits per block (default: 4)
            
        Returns:
            classifier output and representation
        """
        batch_size = z_h.size(0)
        
        # initialize the integrated representation.
        x_h = torch.zeros(batch_size, self.embd_dim, device=z_h.device)
        
        # calculate bit range to zero out
        zero_start = None
        zero_end = None
        if zero_block_idx is not None:
            zero_start = zero_block_idx * block_size
            zero_end = min((zero_block_idx + 1) * block_size, self.max_length)
        
        # process each bit position.
        for i in range(self.max_length):
            bit_i = z_h[:, i]
            
            # convert bit values to embd indices:
            # -1 -> index 0, 0 -> index 1, 1 -> index 2
            embd_indices = (bit_i + 1).long()
            
            # embd for all positions (including -1s).
            # each position i has 3 embds: 0 for -1, 1 for bit 0, 2 for bit 1.
            embds = self.embds[i](embd_indices)
            
            # let -1 embds be zero.
            mask = (bit_i == -1)
            embds[mask] = 0.0
            
            # zero out embds for the specified block
            if zero_start is not None and zero_end is not None and zero_start <= i < zero_end:
                embds = torch.zeros_like(embds)
            
            # repr via inverse transform.
            x_h += embds
            
        return self.classifier(x_h), x_h
    
    def param(self):
        return list(self.parameters())
