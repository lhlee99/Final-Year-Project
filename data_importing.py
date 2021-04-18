"""Read csv with invalid syntax and upload to database for organization"""
#%%
import pandas as pd
import re
import postgres
import re
import json
import contextlib
import csv

# %%
# df = pd.read_csv("post_content/cleaned.csv").dropna()
with open('post_content/error.log', 'w') as log:
    with contextlib.redirect_stderr(log):
        df = pd.read_csv("post_content/post_content_201908+.csv",
                                warn_bad_lines=True, error_bad_lines=False)
df = df.dropna()

#%%
df['idauthor'] = df['idauthor'].astype('int64')
df['idauthor.1'] = df['idauthor.1'].astype('int64')
df['idtopic'] = df['idtopic'].astype('int64')

# %%
author_df_1 = df.iloc[:,1:3].drop_duplicates()
author_df_2 = df.iloc[:,6:8].drop_duplicates()
author_df_2.columns = author_df_1.columns
author_df = pd.concat([author_df_1, author_df_2]).drop_duplicates()
author_df.sort_values(by=['idauthor'])

# %%
author_df.to_csv("post_content/author.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
# %%
from sqlalchemy import create_engine
config = configparser.ConfigParser()
config.read('db.ini')
conn = psycopg2.connect( host=config['postgres']['host'],
                        user=config['postgres']['user'],
                        password=config['postgres']['passwd'],
                        dbname=config['postgres']['db'],
                        # charset='utf8mb4',
                        )
engine = create_engine(f'postgresql://postgres:{config['postgres']['passwd']}@{config['postgres']['host']}:5432/{config['postgres']['passwd']}')
author_df.columns = ['author_id', 'name']
author_df.to_sql('author', engine, index=False, if_exists='append')
# %%
post_df = df[['tcreated', 'idauthor', 'idtopic', 'title']].drop_duplicates()
post_df.columns = ['tcreated', 'author_id', 'post_id', 'title']
post_df.to_sql('post', engine, index=False, if_exists='append')
# %%
comment_df = df[['idauthor.1', 'tcreated.1', 'idtopic', 'content']]
comment_df.columns = ['author_id', 'tcreated', 'post_id', 'content']
comment_df.to_sql('comment', engine, index=False, if_exists='append')
# %%
comment_df.to_csv("post_content/comment.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

# %%
