import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = embed_size // heads
        
        assert (self.heads_dim * heads == embed_size), 'Embed size should be divisible by number of heads'
        
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]
        # Get the Lengths
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # (N, value_len, embed_size)
        values = self.values(values)
        # (N, key_len, embed_size)
        keys = self.keys(keys)
        # (N, query_len, embed_size)
        query = self.queries(query)
        
        # Split into head number of pieces
        values = values.reshape(N, value_len, self.heads, self.heads_dim)
        keys = keys.reshape(N, key_len, self.heads, self.heads_dim)
        queries = query.reshape(N, query_len, self.heads, self.heads_dim)
        
        # query shape: (N, query_len, heads, heads_dim)
        # key shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum('nqhd, nkhd->nhqk', [queries, keys])
        
        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))
        
        attention = torch.softmax(energy/(self.embed_size**0.5), dim = 3)
        # After einsum(): (N, query_len, heads, heads_dim) then flatten last 2-dim
        
        out = torch.einsum('nhql, nlhd->nqhd', [attention,values]).reshape(
            N, query_len, self.heads*self.heads_dim
        )
        # (N, query_len, embed_size)
        out = self.fc_out(out)
        
        return out
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_size,
        heads,
        dropout,
        forward_expansion
    ):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, values, keys, query, mask):
        attention = self.attention(values, keys, query, mask)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        # Passing through FFN, applying dropout and normalizing
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        
        return out
    
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_len
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_len, embed_size)
        
        self.layers = nn.ModuleList(
            [
            TransformerBlock(
                embed_size,
                heads,
                dropout,
                forward_expansion
            )
            for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        # Adding word embedding and positional embedding
        out = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))
        # Passing through each encoder layer
        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        return out
    
class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_size,
        heads,
        forward_expansion,
        dropout,
        device
        ):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size,
            heads,
            dropout,
            forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, value, key, src_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        
        return out
    
class Decoder(nn.Module):
    def __init__(
        self,
        target_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length
        ):
        super(Decoder, self).__init__()
        self.device = device 
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            DecoderBlock(
                embed_size,
                heads,
                forward_expansion,
                dropout,
                device
            )
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout()
    
    def forward(self, x, enc_out, src_mask, target_mask):
        N, seq_len = x.shape
        postion = torch.arange(0,seq_len).expand(N,seq_len).to(self.device)
        # Adding word embedding and positional embedding
        x = self.dropout(self.word_embedding(x)+self.positional_embedding(postion))
        # Passing through each decoder layer
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, target_mask)
        # Passing through FFN
        out = self.fc_out(x)
        
        return out
    
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size = 256,
        num_layers = 6,
        forward_expansion = 4,
        heads = 8,
        dropout = 0,
        device = 'cuda',
        max_len = 100,
        ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_len
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_len
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        # (N,1,1,src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N,1,trg_len,trg_len)
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        # Making masks
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        # Encoding
        enc_src = self.encoder(src, src_mask)
        # Decoding
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        
        return out
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)