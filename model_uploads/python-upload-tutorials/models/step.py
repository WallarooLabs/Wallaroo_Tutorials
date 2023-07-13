import pandas as pd

# take a dataframe output of the house price model, and reformat the `dense_2`
# column as `output`
def wallaroo_json(data: pd.DataFrame):
    print(data)
    return [{"output": [data["dense_2"].to_list()[0][0]]}]