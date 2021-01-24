#%% INIT
import json

from utils import split_text, generate_class_vector

#%%
with open("data/cleaned_1202_v5.json", 'r') as f:
    df = json.loads(f.read())

#%%
# output_name = ["base", "electra", "wwm"]
# pretrained_models = ["bert-base-chinese", "hfl/chinese-electra-180g-base-discriminator", "hfl/chinese-roberta-wwm-ext"]
output_name = ["wwm"]
pretrained_models = ["hfl/chinese-roberta-wwm-ext"]
# generate_class_vector(df, model_name="hfl/chinese-roberta-wwm-ext", output='output/wwm_output_1202.pt')
for i in range(len(pretrained_models)):
    generate_class_vector(
        df,
        model_name=pretrained_models[i],
        output_name=f'output/{output_name[i]}_output_1202_v5.pt',
        full=False)

# %%
