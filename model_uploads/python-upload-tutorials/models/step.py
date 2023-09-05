import pandas as pd

# take a dataframe output of the house price model, and reformat the `dense_2`
# column as `output`
def wallaroo_json(data: pd.DataFrame):
    dense_list = data['dense_2'].to_list()

    df = pd.DataFrame({
        'output': dense_list
    })

    return df.to_dict(orient="records")

    # return [{
    #         'output': dense_list
    # }]



# import pandas as pd

# # take a dataframe output of the house price model, and reformat the `dense_2`
# # column as `output`
# def wallaroo_json(data: pd.DataFrame):
#     print(data)
#     return [{"output": [data["dense_2"].to_list()]}]