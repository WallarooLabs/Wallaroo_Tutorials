The following guide on using inference data inputs from Wallaroo proprietary JSON to either Pandas DataFrame or Apache Arrow downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/tools/convert_wallaroo_data_to_pandas_arrow).

## Introduction

The following guide is to help users transition from using Wallaroo Proprietary JSON to [Pandas DataFrame](https://pandas.pydata.org/docs/reference/frame.html) and [Apache Arrow](https://arrow.apache.org/).  The latter two formats allow data scientists to work natively with DataFrames, and when ready convert those into Arrow table files which provides greater file size efficiency and overall speed.

This guide will demonstrate the following:

* Converting from Wallaroo Proprietary JSON to Pandas DataFrame used for inferences in the Wallaroo Engine.
* Converting from Pandas DataFrame to Apache Arrow used for inferences in the Wallaroo Engine.
* Converting from a flattened Apache Arrow format to multi-dimensional Pandas DataFrame.

### Prerequisites

The demonstration code assumes a Wallaroo instance with Arrow support enabled and provides the following:

* `ccfraud.onnx`: Sample trained ML Model trained to detect credit card fraud transactions.
* `data/high_fraud.json`: Sample inference input formatted in the Wallaroo proprietary JSON format.

The following demonstrates how to convert Wallaroo Proprietary JSON to Pandas DataFrame.  This example data and models are taken from the [Wallaroo 101](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-101), which uses the CCFraud model examples.

## Initial Connection to Wallaroo

The following establishes a connection to a Wallaroo instance for testing and sample inferences.  These steps can be skipped if no sample inferences are required, and are used to demonstrate how inference inputs and output work with Pandas DataFrame and Apache Arrow.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
from IPython.display import display

# used to display dataframe information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)
```

```python
# Login through local Wallaroo instance

# wl = wallaroo.Client()

# SSO login through keycloak

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR PREFIX"

wallarooPrefix = "doc-test"
wallarooSuffix = "wallaroocommunity.ninja"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

```python
workspace_name = 'inferencedataexamplesworkspace'
pipeline_name = 'inferencedataexamplespipeline'
model_name = 'ccfraud'
model_file_name = './ccfraud.onnx'

def get_workspace(name):
    workspace = None
    for ws in wl.list_workspaces():
        if ws.name() == name:
            workspace= ws
    if(workspace == None):
        workspace = wl.create_workspace(name)
    return workspace

def get_pipeline(name):
    try:
        pipeline = wl.pipelines_by_name(pipeline_name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(pipeline_name)
    return pipeline

# Create the workspace

workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

# upload the model
model = wl.upload_model(model_name, model_file_name).configure()

# Create the pipeline then deploy it
pipeline = get_pipeline(pipeline_name)
pipeline.add_model_step(model).deploy()
```

<table><tr><th>name</th> <td>inferencedataexamplespipeline</td></tr><tr><th>created</th> <td>2023-02-28 17:42:19.166319+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-28 17:42:19.871068+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>0ae102ab-082b-4991-9eb2-f43b44d29ed7, 5a90c841-63c6-45c0-b6e6-042e5d213146</td></tr><tr><th>steps</th> <td>ccfraud</td></tr></table>

### Enable Arrow SDK Support

The following is only required when the environment `"ARROW_ENABLED"` has not been set locally.  This environment variable is required in the current release of the SDK to enable Arrow support.

```python
import os
# Only set the below to make the OS environment ARROW_ENABLED to TRUE.  Otherwise, leave as is.
os.environ["ARROW_ENABLED"]="True"
```

## Convert Wallaroo Proprietary JSON to Pandas DataFrame

The following demonstrates how to convert Wallaroo Proprietary JSON to Pandas DataFrame.

### Load Libraries

The following libraries are used as part of the conversion process.

```python
import pandas as pd
import pyarrow as pa
import json
import datetime
import numpy as np
```

### Load Wallaroo Data

The Wallaroo data will be saved to a variable.  This sample input when run through the trained model as an inference returns a high probability of fraud.

```python
# Start with the single example

high_fraud_data = {
    "tensor": [
        [1.0678324729342086,
        18.155556397512136,
        -1.658955105843852,
        5.2111788045436445,
        2.345247064454334,
        10.467083577773014,
        5.0925820522419745,
        12.829515363712181,
        4.953677046849403,
        2.3934736228338225,
        23.912131817957253,
        1.7599568310350207,
        0.8561037518143335,
        1.1656456468728567,
        0.5395988813934498,
        0.7784221343010385,
        6.75806107274245,
        3.927411847659908,
        12.462178276650056,
        12.307538216518655,
        13.787951906620115,
        1.4588397511627804,
        3.681834686805714,
        1.7539143660379741,
        8.484355003656184,
        14.6454097666836,
        26.852377436250144,
        2.716529237720336,
        3.061195706890285]
    ]
}
```

### Convert to DataFrame

The Wallaroo proprietary JSON file will now be converted into Pandas DataFrame.

```python
high_fraud_dataframe =  pd.DataFrame.from_records(high_fraud_data)
display(high_fraud_dataframe)
```

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[1.0678324729342086, 18.155556397512136, -1.658955105843852, 5.2111788045436445, 2.345247064454334, 10.467083577773014, 5.0925820522419745, 12.829515363712181, 4.953677046849403, 2.3934736228338225, 23.912131817957253, 1.7599568310350207, 0.8561037518143335, 1.1656456468728567, 0.5395988813934498, 0.7784221343010385, 6.75806107274245, 3.927411847659908, 12.462178276650056, 12.307538216518655, 13.787951906620115, 1.4588397511627804, 3.681834686805714, 1.7539143660379741, 8.484355003656184, 14.6454097666836, 26.852377436250144, 2.716529237720336, 3.061195706890285]</td>
    </tr>
  </tbody>
</table>

### DataFrame for Inferences

Once converted, the DataFrame version of the data can be used for inferences in an Arrow enabled Wallaroo instance.

```python
# Use this dataframe to infer
result = pipeline.infer(high_fraud_dataframe)
display(result)
```

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-02-28 17:43:17.778</td>
      <td>[1.0678324729, 18.1555563975, -1.6589551058, 5.2111788045, 2.3452470645, 10.4670835778, 5.0925820522, 12.8295153637, 4.9536770468, 2.3934736228, 23.912131818, 1.759956831, 0.8561037518, 1.1656456469, 0.5395988814, 0.7784221343, 6.7580610727, 3.9274118477, 12.4621782767, 12.3075382165, 13.7879519066, 1.4588397512, 3.6818346868, 1.753914366, 8.4843550037, 14.6454097667, 26.8523774363, 2.7165292377, 3.0611957069]</td>
      <td>[0.981199]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Pandas JSON to Pandas DataFrame

For JSON data that is in the Pandas DataFrame format, the data can be turned into a Pandas DataFrame object through the same method.  Note that the original variable is **JSON**, which could have come from a file, to a DataFrame object.

```python
high_fraud_dataframe_json = [
    {
        "tensor":[
            1.0678324729,
            18.1555563975,
            -1.6589551058,
            5.2111788045,
            2.3452470645,
            10.4670835778,
            5.0925820522,
            12.8295153637,
            4.9536770468,
            2.3934736228,
            23.912131818,
            1.759956831,
            0.8561037518,
            1.1656456469,
            0.5395988814,
            0.7784221343,
            6.7580610727,
            3.9274118477,
            12.4621782767,
            12.3075382165,
            13.7879519066,
            1.4588397512,
            3.6818346868,
            1.753914366,
            8.4843550037,
            14.6454097667,
            26.8523774363,
            2.7165292377,
            3.0611957069
        ]
    }
]
```

```python
# Infer from the JSON
high_fraud_from_dataframe_json =  pd.DataFrame.from_records(high_fraud_dataframe_json)
display(high_fraud_from_dataframe_json)
```

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[1.0678324729, 18.1555563975, -1.6589551058, 5.2111788045, 2.3452470645, 10.4670835778, 5.0925820522, 12.8295153637, 4.9536770468, 2.3934736228, 23.912131818, 1.759956831, 0.8561037518, 1.1656456469, 0.5395988814, 0.7784221343, 6.7580610727, 3.9274118477, 12.4621782767, 12.3075382165, 13.7879519066, 1.4588397512, 3.6818346868, 1.753914366, 8.4843550037, 14.6454097667, 26.8523774363, 2.7165292377, 3.0611957069]</td>
    </tr>
  </tbody>
</table>

```python
# Use this dataframe to infer
results = pipeline.infer(high_fraud_from_dataframe_json)
display(results)
```

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-02-28 17:43:23.159</td>
      <td>[1.0678324729, 18.1555563975, -1.6589551058, 5.2111788045, 2.3452470645, 10.4670835778, 5.0925820522, 12.8295153637, 4.9536770468, 2.3934736228, 23.912131818, 1.759956831, 0.8561037518, 1.1656456469, 0.5395988814, 0.7784221343, 6.7580610727, 3.9274118477, 12.4621782767, 12.3075382165, 13.7879519066, 1.4588397512, 3.6818346868, 1.753914366, 8.4843550037, 14.6454097667, 26.8523774363, 2.7165292377, 3.0611957069]</td>
      <td>[0.981199]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

## Convert Wallaroo JSON File to Pandas DataFrame

When working with files containing Wallaroo JSON data, these can be imported from their original JSON, then converted to a Pandas DataFrame object with the pandas method `read_json`.

```python
high_fraud_filename = "./data/high_fraud.json"
high_fraud_data_from_file =  pd.read_json(high_fraud_filename, orient="records")
display(high_fraud_data_from_file)
```

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[1.067832472934208, 18.155556397512136, -1.658955105843852, 5.2111788045436445, 2.345247064454334, 10.467083577773014, 5.092582052241974, 12.829515363712181, 4.953677046849403, 2.393473622833822, 23.912131817957253, 1.7599568310350202, 0.8561037518143331, 1.165645646872856, 0.539598881393449, 0.778422134301038, 6.75806107274245, 3.927411847659908, 12.462178276650056, 12.307538216518655, 13.787951906620115, 1.45883975116278, 3.681834686805714, 1.7539143660379741, 8.484355003656184, 14.6454097666836, 26.852377436250144, 2.7165292377203363, 3.061195706890285]</td>
    </tr>
  </tbody>
</table>

The data can be used in an inference either with the `infer` method on the DataFrame object, or directly from the file.  Note that in either case, the returned object is a DataFrame.

```python
# Use this dataframe to infer
result =  pipeline.infer(high_fraud_data_from_file)
display(result)
```

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-02-28 17:43:44.683</td>
      <td>[1.0678324729, 18.1555563975, -1.6589551058, 5.2111788045, 2.3452470645, 10.4670835778, 5.0925820522, 12.8295153637, 4.9536770468, 2.3934736228, 23.912131818, 1.759956831, 0.8561037518, 1.1656456469, 0.5395988814, 0.7784221343, 6.7580610727, 3.9274118477, 12.4621782767, 12.3075382165, 13.7879519066, 1.4588397512, 3.6818346868, 1.753914366, 8.4843550037, 14.6454097667, 26.8523774363, 2.7165292377, 3.0611957069]</td>
      <td>[0.981199]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
# Infer from file - it is read as a Pandas DataFrame object from the DataFrame JSON
result = pipeline.infer_from_file(high_fraud_filename)
display(result)
```

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-02-28 17:43:45.938</td>
      <td>[1.0678324729, 18.1555563975, -1.6589551058, 5.2111788045, 2.3452470645, 10.4670835778, 5.0925820522, 12.8295153637, 4.9536770468, 2.3934736228, 23.912131818, 1.759956831, 0.8561037518, 1.1656456469, 0.5395988814, 0.7784221343, 6.7580610727, 3.9274118477, 12.4621782767, 12.3075382165, 13.7879519066, 1.4588397512, 3.6818346868, 1.753914366, 8.4843550037, 14.6454097667, 26.8523774363, 2.7165292377, 3.0611957069]</td>
      <td>[0.981199]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

## Convert Pandas DataFrame to Arrow Table

The helper file `convert_wallaroo_data.py` is used to convert from Pandas DataFrame to an Arrow Table with the following caveats:

Arrow requires the user to specify the exact datatypes of the array elements before passing the data to the engine. If you are aware of what data type the model expects, create a dictionary with column names as key and data type as the value and pass it as a param in place of `data_type_dict`. If not, the `convert_to_pa_dtype` function will try and guess the equivalent pyarrow data type and use it (this may or may not work as intended).

```python
import convert_wallaroo_data
```

```python
data_type_dict = {"tensor": pa.float32()}
```

```python
pa_table = convert_wallaroo_data.convert_pandas_to_arrow(high_fraud_dataframe, data_type_dict)
```

```python
pa_table
```

    pyarrow.Table
    tensor: fixed_size_list<item: float>[29]
      child 0, item: float
    ----
    tensor: [[[1.0678325,18.155556,-1.6589551,5.211179,2.345247,...,8.484355,14.64541,26.852377,2.7165291,3.0611956]]]

An inference can be done using the arrow table.  The following shows the code sample and result.  Note that when submitting an Arrow table to `infer`, that the returned object is an Arrow table.

```python
# use the arrow table for infer:
result = pipeline.infer(pa_table)
display(result)
```

    pyarrow.Table
    time: timestamp[ms]
    in.tensor: list<item: float> not null
      child 0, item: float
    out.dense_1: list<inner: float not null> not null
      child 0, inner: float not null
    check_failures: int8
    ----
    time: [[2023-02-28 17:44:28.402]]
    in.tensor: [[[1.0678325,18.155556,-1.6589551,5.211179,2.345247,...,8.484355,14.64541,26.852377,2.7165291,3.0611956]]]
    out.dense_1: [[[0.981199]]]
    check_failures: [[0]]

### Save Arrow Table to Arrow File

The converted Arrow table can be saved using the `pyarrow` library.

```python
arrow_file_name = "./data/high_fraud.arrow"
```

```python
with pa.OSFile(arrow_file_name, 'wb') as sink:
    with pa.ipc.new_file(sink, pa_table.schema) as arrow_ipc:
        arrow_ipc.write(pa_table)
        arrow_ipc.close()
```

`infer_from_file` can be performed using the new `arrow` file.  Note again that when submitting an inference with an Arrow object, the returning value is an Arrow object.

```python
result = pipeline.infer_from_file(arrow_file_name)
display(result)
```

    pyarrow.Table
    time: timestamp[ms]
    in.tensor: list<item: float> not null
      child 0, item: float
    out.dense_1: list<inner: float not null> not null
      child 0, inner: float not null
    check_failures: int8
    ----
    time: [[2023-02-28 17:45:54.049]]
    in.tensor: [[[1.0678325,18.155556,-1.6589551,5.211179,2.345247,...,8.484355,14.64541,26.852377,2.7165291,3.0611956]]]
    out.dense_1: [[[0.981199]]]
    check_failures: [[0]]

## Read Arrow File to DataFrame

The data can go the opposite direction - reading from an Arrow file, and turning the data into either an Arrow table with the Arrow `read_all` method, or just the data into a DataFrame with the Arrow `read_pandas` method.

```python
with pa.ipc.open_file(arrow_file_name) as source:
            table = source.read_all() # to get pyarrow table
            table_df = source.read_pandas() # to get pandas dataframe
            display(table)
            display(table_df)
```

    pyarrow.Table
    tensor: fixed_size_list<item: float>[29]
      child 0, item: float
    ----
    tensor: [[[1.0678325,18.155556,-1.6589551,5.211179,2.345247,...,8.484355,14.64541,26.852377,2.7165291,3.0611956]]]

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[1.0678325, 18.155556, -1.6589551, 5.211179, 2.345247, 10.467084, 5.092582, 12.829515, 4.953677, 2.3934736, 23.912132, 1.7599568, 0.8561038, 1.1656456, 0.5395989, 0.7784221, 6.758061, 3.9274118, 12.462178, 12.307538, 13.787951, 1.4588398, 3.6818347, 1.7539144, 8.484355, 14.64541, 26.852377, 2.7165291, 3.0611956]</td>
    </tr>
  </tbody>
</table>

### Convert Arrow Infer to DataFrame

When an infer result is returned as an Arrow object, it can be converted to a DataFrame for easy viewing.

```python
result = pipeline.infer_from_file(arrow_file_name)
display(result.to_pandas())
```

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-02-28 17:46:31.778</td>
      <td>[1.0678325, 18.155556, -1.6589551, 5.211179, 2.345247, 10.467084, 5.092582, 12.829515, 4.953677, 2.3934736, 23.912132, 1.7599568, 0.8561038, 1.1656456, 0.5395989, 0.7784221, 6.758061, 3.9274118, 12.462178, 12.307538, 13.787951, 1.4588398, 3.6818347, 1.7539144, 8.484355, 14.64541, 26.852377, 2.7165291, 3.0611956]</td>
      <td>[0.981199]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

## Convert Flattened Arrow Table to Multi-Dimensional Pandas DataFrame

Currently, arrow doesn't give us a multi-dimensional result back nor does it provide the shape of the returned data. If you happen to know the shape of the data returned from the model, you could use it to reshape the flattened table to desired multi-dimensional pandas dataframe.

Here is a sample infer result data in arrow Table format.

```python
time_array = pa.array([datetime.datetime(2023, 2 , 22, 22, 14)])
in_tensor_array = pa.array([[1.5997242884551583,-0.72885535293112,-0.8464381472712799,-0.48041787186839674,0.8211244519635765,0.999086254697715,-1.365979802921807,0.36611200379560294,1.27093766309002,0.4895466723195178]])
out_array = pa.array([[1.8749652,-0.94025564,-1.0790397,-0.72123086,0.90895796,1.092086,-1.2834015,0.340406,1.2441622,0.57471186]])
check_failures_array = pa.array([0])
names = ["time", "in.tensor", "out.reshape", "check_failures"]
flattened_2d_table = pa.Table.from_arrays([time_array, in_tensor_array, out_array, check_failures_array], names = names)
flattened_2d_table
```

    pyarrow.Table
    time: timestamp[us]
    in.tensor: list<item: double>
      child 0, item: double
    out.reshape: list<item: double>
      child 0, item: double
    check_failures: int64
    ----
    time: [[2023-02-22 22:14:00.000000]]
    in.tensor: [[[1.5997242884551583,-0.72885535293112,-0.8464381472712799,-0.48041787186839674,0.8211244519635765,0.999086254697715,-1.365979802921807,0.36611200379560294,1.27093766309002,0.4895466723195178]]]
    out.reshape: [[[1.8749652,-0.94025564,-1.0790397,-0.72123086,0.90895796,1.092086,-1.2834015,0.340406,1.2441622,0.57471186]]]
    check_failures: [[0]]

```python
flattened_2d_table["out.reshape"]
```

    <pyarrow.lib.ChunkedArray object at 0x28903b040>
    [
      [
        [
          1.8749652,
          -0.94025564,
          -1.0790397,
          -0.72123086,
          0.90895796,
          1.092086,
          -1.2834015,
          0.340406,
          1.2441622,
          0.57471186
        ]
      ]
    ]

### Verify the Shape

Let's suppose the shape of the output that natively comes out of the model is [2,5].  We can use that to make sure the shape is correct when translating from the 1 dimensional Arrow table.

```python
tensor_type = {"shape": [2, 5]}
```

```python
output_df = flattened_2d_table.to_pandas()['out.reshape'] 
```

```python
# numpy array, shape [N, 2, 5] 
# In this case N = 1
output_list = [elt.reshape(tensor_type['shape']) for elt in output_df]
output_tensor = np.stack(output_list)
```

```python
output_tensor
```

    array([[[ 1.8749652 , -0.94025564, -1.0790397 , -0.72123086,
              0.90895796],
            [ 1.092086  , -1.2834015 ,  0.340406  ,  1.2441622 ,
              0.57471186]]])

```python
output_2d_df = pd.DataFrame(output_tensor.tolist())
```

```python
output_2d_df
```

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[1.8749652, -0.94025564, -1.0790397, -0.72123086, 0.90895796]</td>
      <td>[1.092086, -1.2834015, 0.340406, 1.2441622, 0.57471186]</td>
    </tr>
  </tbody>
</table>

## Undeploy Pipeline

The pipeline will now be undeployed to return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>inferencedataexamplespipeline</td></tr><tr><th>created</th> <td>2023-02-28 17:42:19.166319+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-28 17:42:19.871068+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>0ae102ab-082b-4991-9eb2-f43b44d29ed7, 5a90c841-63c6-45c0-b6e6-042e5d213146</td></tr><tr><th>steps</th> <td>ccfraud</td></tr></table>

