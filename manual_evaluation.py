#%%
import json
import pandas as pd
import matplotlib.pyplot as plt

#%% Read result
with open("./report/KMeans.json", 'r', encoding='utf-8') as f:
    result = json.loads(f.read())
    print(result.keys())
    print(len(result['clusterLabel']))
number_of_cluster = int(result['numberOfCluster'])
#%% Read dataset

with open("./data/cleaned_1202_v5.json", 'r', encoding='utf-8') as f:
    data = json.loads(f.read())
    print(len(data))

#with open("post_1202_v1.json", 'r', encoding='utf-8') as f:
#    data = json.loads(f.read())
#    print(len(data))

#with open("randomNumber_3000.json", 'r', encoding='utf-8') as f:
#    sample_idx = json.loads(f.read())
#    data = [data[idx] for idx in sample_idx]
#    print(len(data))

#%% combine into Dataframe
df = zip(data, result['clusterLabel'])
df = pd.DataFrame(df)
df.columns = ['content', 'cluster']
print(df.dtypes)
df.head()
df.to_excel("./report/data_label.xlsx")

#%% plot # of comment in each cluster
fig = plt.figure()
df.cluster.value_counts().sort_index().plot.bar()
plt.title('distribution')
plt.xlabel('cluster no.')
plt.ylabel('number of elements')
plt.show()
fig.savefig("./report/distri.png")

# %%
all_freq = []
for cluster_idx in range(1, number_of_cluster + 1):
    tmp = {}
    for content in df[df.cluster == cluster_idx]['content'].values:
        for character in content:
            rubbish = [' ', '\n', ',','，', '：', '-', '.', '!', '！', '。', ':', '?', '？', '1', '2', '3', '4', '5', '6', '7', '8', 
                    '9', '0', '一', '二', '係', '你', '我', '唔', '撚', '好', '咁', '佢', '死', '屌', '啦', '嘅', '會', '就', '同', 
                    '呢', '以', '的', '和', '不', '人', '被', '既', '個', '都', '在', '了', '上', '啲', '及', '時', '是', '冇',
                    '年', '月', '日', '王', '又', '仲', '話', '屌', '母', '咪', '咩', '要', '啊', '推', '回', '得', '睇', '自', 
                    '到', '多', '講']
            if character in rubbish:
                continue
            if character in tmp:
                tmp[character] += 1
            else:
                tmp[character] = 1

    # c4_freq = dict(sorted(c4_freq.items(), key=lambda item: item[1], reverse=True)))
    tmp_freq = []
    for key, value in tmp.items():
        tmp_freq.append((value, key))
    tmp_freq.sort(reverse=True)
    all_freq.append(tmp_freq)

# %%
with open("./report/frequentWord.txt", 'w', encoding='utf-8') as f:

    for i, freq in enumerate(all_freq):
        f.write(f'cluster: {i+1}')
        for n, character in freq[:10]:
            f.write(f'   {character}: {n}')
        #f.write('-'*50)
        f.write('\n')

    for k in range(1, number_of_cluster + 1):
        cluster_idx = k
        sample_size = 5
        sample = df[df.cluster == cluster_idx]['content'].sample(sample_size).values
        for i in sample:
            f.write(i)
            f.write('\n')
            f.write('-')
            f.write('\n')
        f.write('\n')
        f.write('-'*60)
        f.write('\n')
