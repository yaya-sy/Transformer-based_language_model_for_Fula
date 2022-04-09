import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import copy

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
        self.mlp                  = nn.Sequential(nn.Linear(embed_dim, ff_size), nn.ReLU())
        self.linear               = nn.Linear(ff_size, embed_dim)
        self.layer_norm2          = nn.LayerNorm(embed_dim)
        self.dropout_ff           = nn.Dropout(p=dropout)
    
    def forward(self, x, attn_mask, key_padding_mask) :
        h, _ = self.multiheads_attention(query=x,
                                    key=x,
                                    value=x,
                                    attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask)
        hz  = self.layer_norm1(h + x)
        f   = self.dropout_ff(self.linear(self.mlp(hz)))
        h   = self.layer_norm2(f + hz)

        return h

class TransformersLM(nn.Module) :
    """
    """

    def __init__(self,
                padding_idx, 
                vocab_size,
                device,
                embed_dim=128, 
                num_layers=4,
                num_heads=4,
                ff_size=256,
                dropout=0.07) :
        super(TransformersLM, self).__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.ff_size = ff_size
        self.dropout = dropout
        self.device = device

        self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=embed_dim,
                                        padding_idx=padding_idx).to(device)
        
        self.transformer1 = TransformerLayer(num_heads=num_heads,
                                            embed_dim=embed_dim,
                                            ff_size=ff_size,
                                            dropout=dropout).to(device)
        self.transformer2 = TransformerLayer(num_heads=num_heads,
                                            embed_dim=embed_dim,
                                            ff_size=ff_size,
                                            dropout=dropout).to(device)
        self.transformer3 = TransformerLayer(num_heads=num_heads,
                                            embed_dim=embed_dim,
                                            ff_size=ff_size,
                                            dropout=dropout).to(device)
        self.transformer4 = TransformerLayer(num_heads=num_heads,
                                            embed_dim=embed_dim,
                                            ff_size=ff_size,
                                            dropout=dropout).to(device)
    
    def forward(self, x) :
        _, seq_len = x.shape
        # get the mask matrix in order to not attend to the future to tokens
        attn_mask = torch.triu(torch.ones(seq_len, seq_len), 1).bool().to(self.device)
        # in order to no attend to pad tokens
        key_padding_mask = (x == self.padding_idx)
        e = self.embeddings(x)
        h = self.transformer1(e, attn_mask, key_padding_mask)
        h = self.transformer2(h, attn_mask, key_padding_mask)
        h = self.transformer3(h, attn_mask, key_padding_mask)
        h = self.transformer4(h, attn_mask, key_padding_mask)

        return h @ self.embeddings.weight.T

    def loss_function(self, logits, golds) :
        return nn.functional.cross_entropy(input=logits,
                                            target=golds,
                                            ignore_index=self.padding_idx,
                                            reduction="mean")
    def top_k_top_p_filtering(self, logits, top_k=15, top_p=0.0, filter_value=-float('Inf')) :
        """
        @source : https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits
    
    def nucleus_sampling(self, data_generator, prompted, temperature=0.93, top_k=15, top_p=0.95, max_predicted_units=64) :
        with torch.inference_mode() :
            gen_idxs = []
            gen_idxs += prompted
            end = False
            for _ in range(max_predicted_units) :
                if end : continue
                inputs = torch.tensor(gen_idxs).view(1, -1).to(self.device)
                # forward
                logits = self(inputs)
                # get the last token
                logits = logits[0, -1, :]
                logits = logits / temperature
                filtered_logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probabilities = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
                if next_token == data_generator.bpe_model.eos_id() : end = True
                gen_idxs.append(next_token.item())
            print("generated: ", data_generator.decode(gen_idxs))

    def train(self, datagenerator, validgenerator, batch_size=32, epochs=256, lr=0.00084) :
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, factor=0.7, mode="min", threshold=0.01)
        best_loss = float("Inf")
        nb_batchs = sum(1 for _ in datagenerator(batch_size))
        epochs_losses = []
        for epoch in range(epochs) :
            loss_sum = 0
            total = 0
            for X, Y, lengths in tqdm(datagenerator(batch_size=batch_size, shuffling=True), total=nb_batchs) :
                self.zero_grad()
                X = torch.tensor(X).to(self.device)
                Y = torch.tensor(Y).to(self.device)
                lengths = torch.tensor(lengths)
                O = self(X) # out.shape = [batch_size, len_sequence, vocab_size]
                batch_len, len_seq, _ = O.shape
                Y = Y.view(-1) # transform shape [batch_size, seq_len] to [batch_size * seq_len]. Enable to flatt Y and see it as gold characters vector : the i^th character is the gold character to the i-1^th character of X.
                O = O.view(batch_len * len_seq, -1) # transform shape [batch_size, seq_len, vocab_size] to [batch_size * seq_len, vocab_size] : this enable to see ouputs as matrix where the i^th vector line is the probabilities distribution predicted for the i+1^th character
                loss = self.loss_function(O, Y) # O.shape[0] and Y.shape[0] must be same
                loss.backward() # backprobagation in order to compute the gradients of the loss function wrt parameters
                optimizer.step() # update parameters
                loss_sum += loss.item()
                total += 1
            train_loss = loss_sum / total
            scheduler.step(train_loss)
            epochs_losses.append(train_loss)
            print("-----" * 20)
            print(f"epoch={epoch}, train loss={train_loss}, train ppl={torch.exp(torch.tensor(train_loss))} lr={optimizer.param_groups[0]['lr']}")
            prompted = validgenerator.prompt()
            print(f"prompted : {datagenerator.decode(prompted)}")
            print("N-Sampling ")
            self.nucleus_sampling(datagenerator, prompted)
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