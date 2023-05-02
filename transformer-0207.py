#!/usr/bin/env python
# coding: utf-8

# # Transformerモデル

# https://blog.octopt.com/transformer/

#  https://qiita.com/gensal/items/e1c4a34dbfd0d7449099

# https://www.dskomei.com/entry/2021/05/24/165158

# In[ ]:


from pathlib import Path
import pandas as pd
import numpy as np
import random
from typing import Tuple
import math
import time
import tqdm
from nltk.translate import bleu_score
from nltk.translate.bleu_score import corpus_bleu

from torch.nn import Transformer
from torch.nn import (
    TransformerEncoder, TransformerDecoder,
    TransformerEncoderLayer, TransformerDecoderLayer
)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
from livelossplot import PlotLosses


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_dir_path = Path('model')
if not model_dir_path.exists():
    model_dir_path.mkdir(parents=True)


# In[ ]:


get_ipython().system('pip install spacy')


# In[ ]:


get_ipython().system('python -m spacy download en')
get_ipython().system('python -m spacy download de')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


SRC = Field(tokenize = "spacy",
            tokenizer_language="de_core_news_sm" ,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

TRG = Field(tokenize = "spacy",
            tokenizer_language="en_core_web_sm",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))


# In[ ]:


SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)


# In[ ]:


def create_mask(src, tgt, PAD_IDX):
    
    seq_len_src = src.shape[0]
    seq_len_tgt = tgt.shape[0]

    mask_tgt = generate_square_subsequent_mask(seq_len_tgt,PAD_IDX)
    mask_src = torch.zeros((seq_len_src, seq_len_src), device=device).type(torch.bool)

    padding_mask_src = (src == PAD_IDX).transpose(0, 1)
    padding_mask_tgt = (tgt == PAD_IDX).transpose(0, 1)
    
    return mask_src, mask_tgt, padding_mask_src, padding_mask_tgt


def generate_square_subsequent_mask(seq_len, PAD_IDX):
    mask = (torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == PAD_IDX, float(0.0))
    return mask


# In[ ]:


class TokenEmbedding(nn.Module):
    
    def __init__(self, vocab_size, embedding_size):
        
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size
        
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.embedding_size)


# In[ ]:


class PositionalEncoding(nn.Module):
    
    def __init__(self, embedding_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        den = torch.exp(-torch.arange(0, embedding_size, 2) * math.log(10000) / embedding_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        embedding_pos = torch.zeros((maxlen, embedding_size))
        embedding_pos[:, 0::2] = torch.sin(pos * den)
        embedding_pos[:, 1::2] = torch.cos(pos * den)
        embedding_pos = embedding_pos.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('embedding_pos', embedding_pos)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.embedding_pos[: token_embedding.size(0), :])


# In[ ]:


class Seq2SeqTransformer(nn.Module):
    
    def __init__(
        self, num_encoder_layers: int, num_decoder_layers: int,
        embedding_size: int, vocab_size_src: int, vocab_size_tgt: int,
        dim_feedforward:int = 512, dropout:float = 0.1, nhead:int = 8
    ):
        #num_encoder_layers:エンコーダーの層の数
        #num_decoder_layers:デコーダーの層の数
        #embedding_size: 埋め込みサイズ
        #vocab_size_src: sourceの語彙数
        #vocab_size_tgt: targetの語彙数
        #dim_feedforward : ffの次元数
        #dropout: 過学習抑制
        #nhead: Multihead Attentionのヘッド数
        
        super().__init__()

        self.token_embedding_src = TokenEmbedding(vocab_size_src, embedding_size)#埋め込み
        self.positional_encoding = PositionalEncoding(embedding_size, dropout=dropout)#位置エンコーディング
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size, nhead=nhead, dim_feedforward=dim_feedforward
        )#エンコーダーレイヤー
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        #エンコーダーの設計
        self.token_embedding_tgt = TokenEmbedding(vocab_size_tgt, embedding_size)#埋め込み
        decoder_layer = TransformerDecoderLayer(
            d_model=embedding_size, nhead=nhead, dim_feedforward=dim_feedforward
        )#デコーダーレイヤー
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        #デコーダーの設計
        self.output = nn.Linear(embedding_size, vocab_size_tgt)#アウトプットレイヤー

    def forward(
        self, src: Tensor, tgt: Tensor,
        mask_src: Tensor, mask_tgt: Tensor,
        padding_mask_src: Tensor, padding_mask_tgt: Tensor,
        memory_key_padding_mask: Tensor
    ):
        
        embedding_src = self.positional_encoding(self.token_embedding_src(src))
        memory = self.transformer_encoder(embedding_src, mask_src, padding_mask_src)
        embedding_tgt = self.positional_encoding(self.token_embedding_tgt(tgt))
        outs = self.transformer_decoder(
            embedding_tgt, memory, mask_tgt, None,
            padding_mask_tgt, memory_key_padding_mask
        )
        return self.output(outs)

    def encode(self, src: Tensor, mask_src: Tensor):
        return self.transformer_encoder(self.positional_encoding(self.token_embedding_src(src)), mask_src)

    def decode(self, tgt: Tensor, memory: Tensor, mask_tgt: Tensor):
        return self.transformer_decoder(self.positional_encoding(self.token_embedding_tgt(tgt)), memory, mask_tgt)


# In[ ]:


batch = iter(train_iterator).__next__()
src = batch.src
trg = batch.trg

src.size()
trg.size()

src.shape[0] #== 29

len(SRC.vocab.stoi)


# In[ ]:


PAD_IDX = TRG.vocab.stoi['<pad>']


# In[ ]:


vocab_size_src = len(SRC.vocab)
vocab_size_tgt = len(TRG.vocab)
embedding_size = 512
nhead = 8
dim_feedforward = 512
num_encoder_layers = 2
num_decoder_layers = 2
dropout = 0.1

model = Seq2SeqTransformer(
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    embedding_size=embedding_size,
    vocab_size_src=vocab_size_src, vocab_size_tgt=vocab_size_tgt,
    dim_feedforward=dim_feedforward,
    dropout=dropout, nhead=nhead
)

model = model.to(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)


# In[ ]:


def train(model: nn.Module,
          iterator: BucketIterator,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()
    

    epoch_loss = 0
    epoch_ave_bleu = 0
    for _, batch in enumerate(iterator):

        src = batch.src # [seq_len,batch_size]
        trg = batch.trg # [seq_len,batch_size]
        
        optimizer.zero_grad()
        
        mask_src, mask_tgt, padding_mask_src, padding_mask_tgt = create_mask(src, trg, PAD_IDX)
        
        output = model(src,trg,mask_src,mask_tgt,padding_mask_src,padding_mask_tgt,padding_mask_src)
        
        bleu_score = caculate_bleu(trg,output)
        epoch_ave_bleu += bleu_score
        
        
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()        

    return (epoch_loss / len(iterator)),(epoch_ave_bleu/len(iterator))


# In[ ]:


def evaluate(model: nn.Module,
             iterator: BucketIterator,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0
    epoch_ave_bleu = 0

    with torch.no_grad():

        for _, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg
            mask_src, mask_tgt, padding_mask_src, padding_mask_tgt = create_mask(src, trg, PAD_IDX)

            output = model(src,trg,mask_src,mask_tgt,padding_mask_src,padding_mask_tgt,padding_mask_src)
            bleu_score = caculate_bleu(trg,output)
            epoch_ave_bleu += bleu_score

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return (epoch_loss / len(iterator)),(epoch_ave_bleu/len(iterator))


# In[ ]:


def delete_index(array):
    array = [s for s in array if s != '<unk>']
    array = [s for s in array if s != '<sos>']
    array = [s for s in array if s != '<eos>']
    array = [s for s in array if s != '<pad>']
    
    return array

def caculate_bleu(correct,answer):
    batch_bleu = 0
    target_corpus = []
    answer_corpus= []
    
    _,answer_array = torch.max(answer,dim=2)
    
    answer_array = answer_array.transpose(1,0)
    correct = correct.transpose(1,0)
    
    for en in answer_array:
        answer_corpus.append([TRG.vocab.itos[e] for e in en])
    for en in correct:
        target_corpus.append([TRG.vocab.itos[e] for e in en])
        
    answer_corpus = np.array(answer_corpus)
    target_corpus = np.array(answer_corpus)
    
    for i in range(answer_corpus.shape[0]):
        myans__ = answer_corpus[i]
        correct__ = target_corpus[i]
        myans__ = delete_index(myans__)
        correct__ = delete_index(correct__)
        
        batch_bleu += corpus_bleu([correct__],[myans__],smoothing_function=bleu_score.SmoothingFunction().method2)
    
    return (batch_bleu)/BATCH_SIZE


# In[ ]:


def __vocab_stoi__(src,correct,answer):
    batch_bleu = 0
    sorce_corpus = []
    target_corpus = []
    answer_corpus= []

    _,answer_array = torch.max(answer,dim=2)
    
    src = src.transpose(1,0)
    answer_array = answer_array.transpose(1,0)
    correct = correct.transpose(1,0)
    
    for de in src:
        sorce_corpus.append([SRC.vocab.itos[e] for e in de])
    for en in answer_array:
        answer_corpus.append([TRG.vocab.itos[e] for e in en])
    for en in correct:
        target_corpus.append([TRG.vocab.itos[e] for e in en])
        
    sorce_corpus = np.array(sorce_corpus)
    answer_corpus = np.array(answer_corpus)
    target_corpus = np.array(target_corpus)
    
        
    for i in range(answer_corpus.shape[0]):
        i_bleu = 0
        src_1 = delete_index(sorce_corpus[i])
        ans_1 = delete_index(answer_corpus[i])
        trg_1 = delete_index(target_corpus[i])
        ans_1 = ans_1[:][1:]
        print("sorce_sentence is",src_1)
        print("answer_sentence is",ans_1)
        print("target_sentence is",trg_1)
        
        i_bleu = corpus_bleu([trg_1],[ans_1],smoothing_function=bleu_score.SmoothingFunction().method2)
        print(i_bleu*100)
        batch_bleu += i_bleu

    return (batch_bleu)/BATCH_SIZE

def evaluate_test(model: nn.Module,
             iterator: BucketIterator,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0
    epoch_ave_bleu = 0

    with torch.no_grad():

        for _, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            mask_src, mask_tgt, padding_mask_src, padding_mask_tgt = create_mask(src, trg, PAD_IDX)

            output = model(src,trg,mask_src,mask_tgt,padding_mask_src,padding_mask_tgt,padding_mask_src)
            epoch_ave_bleu += __vocab_stoi__(src,trg,output)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return (epoch_loss / len(iterator)),(epoch_ave_bleu/len(iterator))


# In[ ]:


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[ ]:


N_EPOCHS = 1
CLIP = 1

loss = []
val_loss = []
bleu_train = []
bleu_valid = []

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    print("{} epoch is training ...".format(epoch+1))

    start_time = time.time()

    train_loss,train_bleu = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss,valid_bleu = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    loss.append(train_loss)
    val_loss.append(valid_loss)
    bleu_train.append(train_bleu*100)
    bleu_valid.append(valid_bleu*100)
    

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    print("Train bleu: {}".format(train_bleu*100))
    print("Valid bleu: {}".format(valid_bleu*100))
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model_transformer.pth')


# In[ ]:


liveloss = PlotLosses()
for n in range(N_EPOCHS):
    logs = {}
    logs['loss'] = loss[n]
    logs['val_loss'] = val_loss[n]

    liveloss.update(logs)
    liveloss.send()


# In[ ]:


liveloss = PlotLosses()
for n in range(N_EPOCHS):
    logs = {}
    logs['bleu_train'] = bleu_train[n]
    logs['bleu_valid'] = bleu_valid[n]

    liveloss.update(logs)
    liveloss.send()


# In[ ]:


test_loss,test_bleu = evaluate_test(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
print("Test bleu: {}".format(test_bleu*100))

