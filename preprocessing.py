from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
import torch

from transformers import AutoTokenizer
from datasets import load_dataset




wikipedia=load_dataset("wikimedia/wikipedia",'20231101.en',split="train")
books=load_dataset("bookcorpus",split="train")
tokenizer=AutoTokenizer.from_pretrained("albert/albert-base-v2")


b=books.to_list()
bs=[ba['text'] for ba in b ]
books=[[(bs[i],bs[i+1])] for i in range(0,len(bs),2)]
a=len(books)
books.extend([[(bs[i],bs[i+4])] for i in range(0,len(bs)-4,2)])




class TextDataset(Dataset):
    def __init__(self, text,tokenizer):
        self.text = text
        self.tokenizer=tokenizer
        self.nsp_labels=torch.LongTensor([1]*a+[0]*(len(books)-a)).reshape(-1,1)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        data=self.tokenizer(self.text[idx],padding='max_length',max_length=512, truncation=True, return_tensors="pt")
        d={ key:val.squeeze() for (key,val) in data.items() }
        d['label']=self.nsp_labels[idx]
        
        return d



text_dataset = TextDataset(books,tokenizer)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)


text_dataloader = DataLoader(text_dataset, batch_size=16, shuffle=True, collate_fn=data_collator,num_workers=7)





