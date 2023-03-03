This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_conversion/xgboost-to-onnx).

## How to Convert XGBoost to ONNX

The following tutorial is a brief example of how to convert a [XGBoost](https://xgboost.readthedocs.io/en/stable/index.html) ML model to the [ONNX](https://onnx.ai/ ) standard.  This allows organizations that have trained XGBoost models to convert them and use them with Wallaroo.

This tutorial assumes that you have a Wallaroo instance and are running this Notebook from the Wallaroo Jupyter Hub service.

This tutorial provides the following:

* `housing_model_xgb.pkl`: A pretrained model used as part of the [Notebooks in Production tutorial](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/notebooks_in_prod).  This model has a total of 18 columns.

## Conversion Process

### Libraries

The first step is to import our libraries we will be using.


```python
import onnx
import pickle
from onnxmltools.convert import convert_xgboost

from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
```

### Set Variables

The following variables are required to be known before the process can be started:

* **number of columns**:  The number of columns used by the model.
* **TARGET_OPSET**: Verify the TARGET_OPSET value taht will be used in the conversion process matches the current [Wallaroo model uploads requirements](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-model-management/#upload-models-to-a-workspace).


```python
# set the number of columns
ncols = 18
TARGET_OPSET = 15
```

### Load the XGBoost Model

Next we will load our model that has been saved in the `pickle` format and unpickle it.


```python
# load the xgboost model
with open("housing_model_xgb.pkl", "rb") as f:
    xgboost_model = pickle.load(f)
```

### Conversion Inputs

The `convert_xgboost` method has the following format and requires the following inputs:

```
convert_xgboost({XGBoost Model}, 
                {XGBoost Model Type},
                [
                    ('input', 
                    {Tensor Data Type}([None, {ncols}]))
                ],
                target_opset={TARGET_OPSET})
```
    
1. **XGBoost Model**:  The XGBoost Model to convert.
1. **XGBoost Model Type**: The type of XGBoost model.  In this example is it a `tree-based classifier`.
1. **Tensor Data Type**:  Either `FloatTensorType` or `DoubleTensorType` from the `skl2onnx.common.data_types` library.
1. **ncols**:  Number of columns in the model.
1. **TARGET_OPSET**:  The target opset which can be derived in code showed below.

## Convert the Model

With all of our data in place we can now convert our XBBoost model to ONNX using the `convert_xgboost` method.


```python
onnx_model_converted = convert_xgboost(xgboost_model, 'tree-based classifier',
                             [('input', FloatTensorType([None, ncols]))],
                             target_opset=TARGET_OPSET)
```

## Save the Model

With the model converted to ONNX, we can now save it and use it in a Wallaroo pipeline.


```python
onnx.save_model(onnx_model_converted, "housing_model_xgb.onnx")
```


```python

```
