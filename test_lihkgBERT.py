#%%
from transformers import pipeline


fill_mask = pipeline("fill-mask", model="./models/lihkgBERT", tokenizer="./models/lihkgBERT")
base_fill_mask = pipeline("fill-mask", model="bert-base-chinese", tokenizer="bert-base-chinese")
#%%
a = '可唔可以咁講美國d商家已經共識[MASK]'
b = '大量因素都抗爭咗成年算係[MASK]啦'

#%%
re_a = fill_mask(b)

# %%
re_b = base_fill_mask(b)
# %%
[(i['token_str'], i['score']) for i in re_a]
# %%
[(i['token_str'], i['score']) for i in re_b]

# %%
from transformers import pipeline, BertModel, BertTokenizerFast

# %%
model = BertModel.from_pretrained('models/lihkgBERT')
tk = BertTokenizerFast.from_pretrained('models/lihkgBERT')
# %%
tokens = tk([a,b],padding=True, return_tensors='pt')
# %%
model.eval()
tmp = model(**tokens)

# %%
