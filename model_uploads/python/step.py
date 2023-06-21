import pandas as pd

def wallaroo_json(data: pd.DataFrame):
    print(data)
    return [{"output": [data["dense_2"].to_list()[0]]}]