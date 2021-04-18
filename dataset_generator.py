#%%
import json
import postgres
import re

#%%
date = '2021-01-05'
data = postgres.query(f"""
SELECT cleaned_content FROM comment
WHERE tcreated::date = '{date}'
""", return_1d_array=True)
root = 'data'

# %%
def actual_length(sequence):
    tmp = re.sub(r"[A-Za-z']+", "_", sequence)
    return len(tmp)
data = [i.lower() for i in data if actual_length(i) >= 5]

# %%
with open(f'{root}/{date}_comments.json', 'w') as f:
    f.write(json.dumps(data))
# %%
data = postgres.query("""
SELECT cleaned_content FROM comment
WHERE cluster_uid IN (30, 36, 27, 4, 10, 25, 28, 11, 37)
ORDER BY comment_id
""", return_1d_array=True)
with open(f'{root}/validation.json', 'w') as f:
    f.write(json.dumps(data))
# %%
