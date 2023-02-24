import pandas as pd
import pyarrow as pa
import json


def convert_to_pa_dtype(dtype):
    """
    Checks the python data type and returns the closest matching pyarrow data type
    """
    if isinstance(dtype, float):
        return  pa.float32()
    elif isinstance(dtype, int):
        return pa.int64()
    elif isinstance(dtype, int):
        return pa.string()
    elif isinstance(dtype, bool):
        return pa._bool()
    return None

def convert_pandas_to_arrow(data, data_type_dict):
    """
    Converts passed in pandas dataframe `data` into pyarrow table with each column as fixed_size_list arrays
    with proper data types
    
    create a dictionary with column names as key and data type as the value and pass it as a param in place of `data_type_dict`. 
    If not, the `convert_to_pa_dtype` function will try and guesss the equivalent pyarrow data type and use it (this may or may not work as intended).
    """
    data_table = pa.Table.from_pandas(data)
    fields = []
    data_type = None
    for i in data_table.column_names:
        if pa.types.is_fixed_size_list(data_table[i].type):
            fields.append(pa.field(i, data_table[i].type))
        else:
            inner_size = len(data_table[i][0])
            tensor_type = {"shape": [inner_size]}
            tensor_meta_type = {"tensor_type": json.dumps(tensor_type)}
            if data_type_dict is not None:
                data_type = data_type_dict[i]
            if data_type is None:
                data_type=convert_to_pa_dtype(data[i][0][0])
            tensor_arrow_type = pa.list_(data_type, inner_size)
            fields.append(pa.field(i, tensor_arrow_type, metadata=tensor_meta_type))
    schema = pa.schema(fields)
    final_table = pa.Table.from_pandas(data, schema=schema).cast(target_schema=schema)
    return final_table