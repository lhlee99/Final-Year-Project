import configparser
import psycopg2
import json
from tqdm import tqdm, trange

def query(q, output=True, return_dict=False, return_1d_array=False):
    config = configparser.ConfigParser()
    config.read('db.ini')
    conn = psycopg2.connect( host=config['postgres']['host'],
                            user=config['postgres']['user'],
                            password=config['postgres']['passwd'],
                            dbname=config['postgres']['db'],
                            # charset='utf8mb4',
                            )
    cursor = conn.cursor()
    cursor.execute(q)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    print(f'{len(result)} rows returned')
    if output:
        print(f"Returned {len(result)} rows.")
    if return_dict:
        columns = q[q.find("SELECT") + len("SELECT"):q.find("FROM")].strip()
        columns = columns.split(',')
        columns = [i.strip() for i in columns]
        result_dict = []
        for item in result:
            tmp_dict = {}
            for i, column in enumerate(columns):
                tmp_dict[column] = item[i]
            result_dict.append(tmp_dict)
        print(f"Columns: {', '.join(columns)}")
        try:
            print(json.dumps(result_dict[0], indent=2, ensure_ascii=False) + "\n...")
        except TypeError:
            print(result_dict[0])
            print("...")
        return result_dict
    if return_1d_array:
        result = [i[0] for i in result]
    return result

def dict_to_list(d):
    """Convert dictionary values into 2 lists (key, value)."""
    l = []
    for key in d.keys():
        l.append(d[key])
    return d.keys(), l

def executemany(method, table_name, data, pk='id'):
    if method.lower() == 'delete' and type(data) == list:
        data = [{ pk: i } for i in data]
    data_length = len(data)
    length = len(data[0].keys())
    print(f"{table_name} {method} {data_length} rows x {length} column")
    config = configparser.ConfigParser()
    config.read('db.ini')
    conn = psycopg2.connect( host=config['postgres']['host'],
                            user=config['postgres']['user'],
                            password=config['postgres']['passwd'],
                            dbname=config['postgres']['db'],
                            # charset='utf8mb4',
                            )
    cursor = conn.cursor()

    for d in tqdm(data, desc=method):
        keys, values = list(d.keys()), list(d.values())

        if method.lower() == 'update':
            # remove pk from the dictionary
            pk_value = d.pop(pk)
            keys, values = list(d.keys()), list(d.values())
            keys = [f'"{key}" = %s' for key in keys]
            command = f"UPDATE {table_name} SET {', '.join(keys)} WHERE \"{pk}\" = {pk_value}"

        elif method.lower() == 'insert':
            keys = [f'"{key}"' for key in keys]
            command = f"INSERT INTO {table_name}({', '.join(keys)}) VALUES ({', '.join(['%s'] * length)})"

        elif method.lower() == 'delete':
            command = f"DELETE FROM {table_name} WHERE \"{pk}\" = {d[pk]}"
        else:
            raise Exception(f"Method {method} not handled...")


        if len(keys) != len(values):
            raise Exception('key, value length do not match', keys, values)

        try:
            cursor.execute(command, values)
        except Exception as e:
            print(values)
            print(keys)
            raise e

    conn.commit()
    print("Committed change, closing connection")
    cursor.close()
    conn.close()

def insert_many(table_name, data):
    """Insert many row to postgres from a list of dictionary

    Args:
        table_name(str): name of the target table
        data(list of dict): list of dictionary to be added to postgres
    """
    executemany("insert", table_name, data)

def update_many(table_name, data, pk='id'):
    """Insert many row to postgres from a list of dictionary

    Args:
        table_name(str): name of the target table
        data(list of dict): list of dictionary to be added to postgres
        pk(str): name of the primary key. Default to dis
    """
    executemany('update', table_name, data, pk=pk)

def delete_many(table_name, data, pk='id'):
    executemany('delete', table_name, data, pk=pk)

def read_csv_as_dict(filename):
    df = pd.read_csv(filename)
    data = json.loads(df.to_json(orient="records"))
    assert len(data) == len(df), f'Length does not equal. df: {len(df)}, data: {len(data)}'
    print(f'Returned {len(data)} items.')
    return data