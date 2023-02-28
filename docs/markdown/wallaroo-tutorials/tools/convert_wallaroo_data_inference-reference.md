The following guide on using inference data inputs from Wallaroo proprietary JSON to either Pandas DataFrame or Apache Arrow downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/quickstartguide_samples/blob/20230207_arrow_versions/tools/convert_wallaroo_data_to_pandas_arrow).

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

## Convert Wallaroo Proprietary JSON to Pandas DataFrame

The following demonstrates how to convert Wallaroo Proprietary JSON to Pandas DataFrame.  This example data and models are taken from the [Wallaroo 101](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-101), which uses the CCFraud model examples.
```

### Load Libraries

The following libraries are used as part of the conversion process.


```python
import pandas as pd
import pyarrow as pa
import json
import datetime
import numpy as np

pd.set_option('display.max_colwidth', None)
```

### Load Wallaroo Data

The Wallaroo data will be saved to a variable.

```python
# Start with the single example

high_fraud_data = {"tensor": [[1.0678324729342086,
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
    3.061195706890285]]
}
```

### Convert to DataFrame

The Wallaroo proprietary JSON file will now be converted into Pandas DataFrame.


```python
high_fraud_dataframe =  pd.DataFrame.from_records(high_fraud_data)
display(high_fraud_dataframe)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
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
</div>


### DataFrame for Inferences

Once converted, the DataFrame version of the data can be used for inferences in an Arrow enabled Wallaroo instance.


```python
# Use this dataframe to infer
# pipeline.infer(high_fraud_dataframe)
```

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
high_fraud_from_dataframe_json =  pd.DataFrame.from_records(high_fraud_dataframe_json)
display(high_fraud_from_dataframe_json)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
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
</div>



```python
# Use this dataframe to infer
# pipeline.infer(high_fraud_from_dataframe_json)
```

### Convert Wallaroo JSON File to Pandas DataFrame

When working with files containing Wallaroo JSON data, these can be imported from their original JSON, then converted to a Pandas DataFrame object with the pandas method `read_json`.


```python
high_fraud_filename = "./data/high_fraud.json"
high_fraud_data_from_file =  pd.read_json(filename, orient="records")
display(high_fraud_data_from_file)
```

The data can be used in an inference either with the `infer` method on the DataFrame object, or directly from the file.  Note that in either case, the returned object is a DataFrame.

```python
# Use this dataframe to infer
# result_dataframe =  pipeline.infer(data)
# display(result_dataframe)
```

DataFrame Result:

```
	time	in.tensor	out.dense_1	check_failures
0	2023-02-23 16:01:46.486	[1.0678324729, 18.1555563975, -1.6589551058, 5.2111788045, 2.3452470645, 10.4670835778, 5.0925820522, 12.8295153637, 4.9536770468, 2.3934736228, 23.912131818, 1.759956831, 0.8561037518, 1.1656456469, 0.5395988814, 0.7784221343, 6.7580610727, 3.9274118477, 12.4621782767, 12.3075382165, 13.7879519066, 1.4588397512, 3.6818346868, 1.753914366, 8.4843550037, 14.6454097667, 26.8523774363, 2.7165292377, 3.0611957069]	[0.981199]	0
```

```python
# # or to use the file directly,
result_json = pipeline.infer_from_file(filename)
display(result_dataframe)
```

JSON Result:

```
	time	in.tensor	out.dense_1	check_failures
0	2023-02-24 21:26:22.916	[1.0678324729, 18.1555563975, -1.6589551058, 5.2111788045, 2.3452470645, 10.4670835778, 5.0925820522, 12.8295153637, 4.9536770468, 2.3934736228, 23.912131818, 1.759956831, 0.8561037518, 1.1656456469, 0.5395988814, 0.7784221343, 6.7580610727, 3.9274118477, 12.4621782767, 12.3075382165, 13.7879519066, 1.4588397512, 3.6818346868, 1.753914366, 8.4843550037, 14.6454097667, 26.8523774363, 2.7165292377, 3.0611957069]	[0.981199]	0
```





### Convert Pandas DataFrame to Arrow Table

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
result = ccfraud_pipeline.infer(pa_table)
display(result)

pyarrow.Table
time: timestamp[ms]
in.tensor: list<item: float> not null
  child 0, item: float
out.dense_1: list<inner: float not null> not null
  child 0, inner: float not null
check_failures: int8
----
time: [[2023-02-24 21:40:26.752]]
in.tensor: [[[1.0678325,18.155556,-1.6589551,5.211179,2.345247,...,8.484355,14.64541,26.852377,2.7165291,3.0611956]]]
out.dense_1: [[[0.981199]]]
check_failures: [[0]]
```



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
result = ccfraud_pipeline.infer_from_file('./data/high_fraud.arrow')
display(result)

pyarrow.Table
time: timestamp[ms]
in.tensor: list<item: float> not null
  child 0, item: float
out.dense_1: list<inner: float not null> not null
  child 0, inner: float not null
check_failures: int8
----
time: [[2023-02-24 21:49:04.323]]
in.tensor: [[[1.0678325,18.155556,-1.6589551,5.211179,2.345247,...,8.484355,14.64541,26.852377,2.7165291,3.0611956]]]
out.dense_1: [[[0.981199]]]
check_failures: [[0]]
```

### Read Arrow File to DataFrame

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



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
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
</div>


### Convert Arrow Infer to DataFrame

When an infer result is returned as an Arrow object, it can be converted to a DataFrame for easy viewing.

```python
result = ccfraud_pipeline.infer_from_file('./data/high_fraud.arrow')
display(result.to_pandas())
	time	in.tensor	out.dense_1	check_failures
0	2023-02-24 21:49:04.323	[1.0678325, 18.155556, -1.6589551, 5.211179, 2.345247, 10.467084, 5.092582, 12.829515, 4.953677, 2.3934736, 23.912132, 1.7599568, 0.8561038, 1.1656456, 0.5395989, 0.7784221, 6.758061, 3.9274118, 12.462178, 12.307538, 13.787951, 1.4588398, 3.6818347, 1.7539144, 8.484355, 14.64541, 26.852377, 2.7165291, 3.0611956]	[0.981199]	0
```

### Convert flattened arrow table to multi-dimensional pandas dataframe

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




    <pyarrow.lib.ChunkedArray object at 0x11e1ef630>
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



#### Let's suppose the shape of the output that natively comes out of the model is [2,5]


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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
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
</div>


