import torch.nn.functional as F
from datasets import load_dataset
import torch.nn as nn
import torch.nn.functional as F
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
        embedding=torch.masked_fill(embedding,attention_mask.unsqueeze(-1)==0,0)
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



''' 

AlbertModel(
  (embeddings): AlbertEmbeddings(
    (word_embeddings): Embedding(30000, 128, padding_idx=0)
    (position_embeddings): Embedding(512, 128)
    (token_type_embeddings): Embedding(2, 128)
    (LayerNorm): LayerNorm((128,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0, inplace=False)
  )
  (encoder): AlbertTransformer(
    (embedding_hidden_mapping_in): Linear(in_features=128, out_features=768, bias=True)
    (albert_layer_groups): ModuleList(
      (0): AlbertLayerGroup(
        (albert_layers): ModuleList(
          (0): AlbertLayer(
            (full_layer_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (attention): AlbertAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (attention_dropout): Dropout(p=0, inplace=False)
              (output_dropout): Dropout(p=0, inplace=False)
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            )
            (ffn): Linear(in_features=768, out_features=3072, bias=True)
            (ffn_output): Linear(in_features=3072, out_features=768, bias=True)
            (activation): NewGELUActivation()
            (dropout): Dropout(p=0, inplace=False)
          )
        )
      )
    )
  )
  (pooler): Linear(in_features=768, out_features=768, bias=True)
  (pooler_activation): Tanh()
)

'''


# test()
    