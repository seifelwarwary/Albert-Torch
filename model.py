

import torch.nn as nn
import torch

from transformers import AutoTokenizer



class AlbertEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(30000,128)
        self.pos_embedding=nn.Embedding(512,128)
        self.token_type_embedding=nn.Embedding(2,128)
        self.layer_norm=nn.LayerNorm(128)
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def forward(self,inputs):
        input_ids,attention_mask,token_type_ids=inputs
        
        
        
        input_embedding=self.embedding(input_ids)
        p=torch.arange(input_ids.size(1),device=self.device).unsqueeze(0)
        
        pos_embedding=self.pos_embedding(p)
        token_type_embedding=self.token_type_embedding(token_type_ids)
        embedding=self.layer_norm(input_embedding+pos_embedding+token_type_embedding)
        embedding=embedding.masked_fill_(attention_mask.unsqueeze(-1)==0,0)
        
        return embedding    

class AlbertNSPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(768,2)
        

        self.act_fn=nn.LogSoftmax(-1)
    def forward(self,inputs):
        x=self.linear(inputs)
        
        x=self.act_fn(x)
        return x

class AlbertMLMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(768,30000)
        self.act_fn=nn.LogSoftmax(-1)
    def forward(self,inputs):
        x=self.linear(inputs)
        
        x=self.act_fn(x)
        return x

class AlbertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=AlbertEmbedding()
        self.linear=nn.Linear(128,768)
        self.layer_norm=nn.LayerNorm((768,))
        
        self.encoder=nn.TransformerEncoderLayer(768,12,768,0)
        self.ffn=nn.Linear(768,3072)
        self.ffn_output=nn.Linear(3072,768)
        self.act_fn=nn.GELU()
        self.pooler=nn.Linear(768,768)        
        self.relu=nn.ReLU()
        self.mlm_head=AlbertMLMHead()
        self.nsp_head=AlbertNSPHead()
        
    def forward(self,inputs):
        x=self.embedding(inputs)
        x=self.linear(x)
        for _ in range(12):
            x=self.layer_norm(x)
            x=self.encoder(x)
            x=self.ffn(x)
            x=self.ffn_output(x)
            x=self.act_fn(x)
        
        x=self.pooler(x)
        x=self.act_fn(x)
        y1=self.mlm_head(x)
        y2=self.nsp_head(x[:,0,:])


        return y1,y2


def test():
    tokenizer=AutoTokenizer.from_pretrained("albert/albert-base-v2")

    raw_inputs='[CLS] SEQUENCE_A [SEP] SEQUENCE_B [SEP]'*100
    # raw_inputs=[raw_inputs]*2
    inputs = tokenizer(raw_inputs,raw_inputs,max_length=512, padding=True, truncation=True, return_tensors="pt")
    albert_embeddings=AlbertEmbedding()
    albert_embeddings(inputs).size()

    l1=nn.Linear(128,768)(albert_embeddings(inputs))
    nn.TransformerEncoderLayer(768,12,768,0)(l1).size()
    model_test=AlbertModel()
    model_test(inputs)





test()
    