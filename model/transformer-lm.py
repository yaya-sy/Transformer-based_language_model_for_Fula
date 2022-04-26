import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
from utils.text_generator import *

class TransformerLayer(nn.Module) :
    """
    """

    def __init__(self, num_heads, embed_dim, ff_size, dropout=0.07) :    
        super(TransformerLayer, self).__init__()
        self.multiheads_attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                                        num_heads=num_heads,
                                                        dropout=dropout,
                                                        batch_first=True)
        self.dropout_mh           = nn.Dropout(p=dropout)
        self.layer_norm1          = nn.LayerNorm(embed_dim)
        self.ff                   = nn.Sequential(nn.Linear(embed_dim, ff_size),
                                                nn.ReLU(),
                                                nn.Linear(ff_size, embed_dim))
        self.layer_norm2          = nn.LayerNorm(embed_dim)
        self.dropout_ff           = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None) :
        # multi head attention
        h, attn = self.multiheads_attention(query=query,
                                    key=key,
                                    value=value,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask)
        h = self.dropout_mh(h)
        # add and norm
        hz  = self.layer_norm1(h + query)
        # Feedforward and dropout
        f   = self.dropout_ff(self.ff(hz))
        # add and norm
        h   = self.layer_norm2(f + hz)

        return h, attn


class TransformersDecoder(nn.Module) :

    def __init__(self,
                 padding_idx,
                 device,
                 embed_dim=256,
                 num_heads=4,
                 ff_size=512,
                 dropout=0.07
                 ) :
        super(TransformersDecoder, self).__init__()
        self.device = device
        self.padding_idx = padding_idx

        self.transformer_dec1 = TransformerLayer(num_heads=num_heads,
                                                 embed_dim=embed_dim,
                                                 ff_size=ff_size,
                                                 dropout=dropout).to(device)

        self.transformer_dec2 = TransformerLayer(num_heads=num_heads,
                                                 embed_dim=embed_dim,
                                                 ff_size=ff_size,
                                                 dropout=dropout).to(device)
        self.transformer_dec3 = TransformerLayer(num_heads=num_heads,
                                                 embed_dim=embed_dim,
                                                 ff_size=ff_size,
                                                 dropout=dropout).to(device)
        self.transformer_dec4 = TransformerLayer(num_heads=num_heads,
                                                 embed_dim=embed_dim,
                                                 ff_size=ff_size,
                                                 dropout=dropout).to(device)

    def forward(self, x, self_attn_mask, padding_mask) :
        # tranformer with self attention between
        h, _ = self.transformer_dec1(query=x,
                                     key=x,
                                     value=x,
                                     attn_mask=self_attn_mask,
                                     key_padding_mask=padding_mask)
        
        h, _ = self.transformer_dec2(query=h,
                                     key=h,
                                     value=h,
                                     attn_mask=self_attn_mask,
                                     key_padding_mask=padding_mask)
        
        h, _ = self.transformer_dec3(query=h,
                                     key=h,
                                     value=h,
                                     attn_mask=self_attn_mask,
                                     key_padding_mask=padding_mask)
        h, _ = self.transformer_dec4(query=h,
                                     key=h,
                                     value=h,
                                     attn_mask=self_attn_mask,
                                     key_padding_mask=padding_mask)
        return h

class TransformersLM(nn.Module) :
    """
    """

    def __init__(self,
                padding_idx, 
                vocab_size,
                device,
                embed_dim=256, 
                num_heads=4,
                ff_size=512,
                dropout=0.07,
                max_len=5_000) :

        super(TransformersLM, self).__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.ff_size = ff_size
        self.dropout = dropout
        self.device = device
        self.pos_padding = max_len

        self.tok_embedding = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embed_dim,
                                            padding_idx=padding_idx).to(device)
        self.pos_embedding = nn.Embedding(num_embeddings=max_len + 1,
                                            embedding_dim=embed_dim,
                                            padding_idx=self.pos_padding).to(device)

        self.decoder = TransformersDecoder(
            padding_idx=padding_idx,
            vocab_size=vocab_size,
            device=device,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_size=ff_size,
            dropout=dropout
        )
      
  
    def forward(self, x) :
        batch_size, seq_len = x.shape
        padding_mask = (x == self.padding_idx).to(self.device)

        tok_embedd = self.tok_embedding(x)
        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        pos[padding_mask] = self.pos_padding
        pos_embed = self.pos_embedding(pos)
        embed = tok_embedd + pos_embed

        # mask attention to tokens in the future
        self_attn_mask = torch.triu(torch.ones(seq_len, seq_len), 1).bool().to(self.device)
        # decoder
        decoded = self.decoder(embed, self_attn_mask, padding_mask)

        return decoded @ self.tok_embedding.weight.T

    def loss_function(self, logits, golds) :
        return nn.functional.cross_entropy(input=logits,
                                            target=golds,
                                            ignore_index=self.padding_idx,
                                            reduction="mean")

    def train(self, datagenerator, validgenerator, batch_size=6, epochs=256, lr=0.00084) :
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, factor=0.7, mode="min", threshold=0.01)
        best_loss = float("Inf")
        nb_batchs = sum(1 for _ in datagenerator(batch_size))
        epochs_losses = []
        for epoch in range(epochs) :
            loss_sum = 0
            total = 0
            for X, Y, _ in tqdm(datagenerator(batch_size=batch_size, shuffling=True), total=nb_batchs) :
                self.zero_grad()
                X = torch.tensor(X).to(self.device)
                Y = torch.tensor(Y).to(self.device)
                O = self(X) # outshape = [batch_size, len_sequence, vocab_size]
                batch_len, len_seq, _ = O.shape
                Y = Y.view(-1) # transform shape [batch_size, seq_len] to [batch_size * seq_len]. This enable to flatt Y and see it as gold characters vector : the i^th character is the gold character to the i-1^th character of X.
                O = O.view(batch_len * len_seq, -1) # transform shape [batch_size, seq_len, vocab_size] to [batch_size * seq_len, vocab_size] : this enable to see ouputs as matrix where the i^th vector line is the probabilities distribution predicted for the i+1^th character
                loss = self.loss_function(O, Y) # O.shape[0] and Y.shape[0] must be the same
                loss.backward() # backprobagation in order to compute the gradients of the loss function wrt parameters
                optimizer.step() # update parameters
                loss_sum += loss.item()
                total += 1
            train_loss = loss_sum / total
            scheduler.step(train_loss)
            epochs_losses.append(train_loss)
            print()
            print("-----" * 20)
            print(f"epoch={epoch}, train loss={train_loss}, train ppl={torch.exp(torch.tensor(train_loss))} lr={optimizer.param_groups[0]['lr']}")
            prompted = validgenerator.prompt()
            print(f"prompted : {validgenerator.decode(prompted)}")
            print(f"generated : {nucleus_sampling(self, self.device, validgenerator, prompted)}")
            # print("Sampling")
            # self.generate(datagenerator, prompted, sample=True)
            if train_loss < best_loss :
                best_loss = train_loss
                self.set_best()
        return epochs_losses
    
    def set_best(self):
        self.best_state_dict = copy.deepcopy(self.state_dict())

    def recover_best(self):
        self.load_state_dict(self.best_state_dict)