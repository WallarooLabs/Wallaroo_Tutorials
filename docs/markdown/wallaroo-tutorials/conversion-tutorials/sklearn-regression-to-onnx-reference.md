This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_conversion/sklearn-regression-to-onnx).

## How to Convert sk-learn Regression Model to ONNX

The following tutorial is a brief example of how to convert a [scikit-learn](https://scikit-learn.org/stable/) (aka sk-learn) regression ML model to the [ONNX](https://onnx.ai/ ).

This tutorial assumes that you have a Wallaroo instance and are running this Notebook from the Wallaroo Jupyter Hub service.

This tutorial provides the following:

* `demand_curve.pickle`: a demand curve trained sk-learn model.  Once this file is converted to ONNX format, it can be used as part of the [Demand Curve Pipeline Tutorial](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-model-cookbooks/demand_curve).

    This model contains 3 columns: `UnitPrice`, `cust_known`, and `UnitPriceXcust_known`.

## Prerequisites

* An installed Wallaroo instance.
* The following Python libraries installed:
  * `pickle`
  * [`skl2onnx`](https://pypi.org/project/skl2onnx/)
  * [`onnxmltools`](https://pypi.org/project/onnxmltools/)
  * [`onnx`](https://pypi.org/project/onnx/)
  * `warnings`

## Conversion Process

### Libraries

The first step is to import our libraries we will be using.

```python
# Used to load the sk-learn model
import pickle

# Used for the conversion process
import onnx, skl2onnx, onnxmltools
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.data_types import DoubleTensorType

# ignoring warnings for demonstration
import warnings
warnings.filterwarnings('ignore')
```

Next let's define our `model_to_onnx` method for converting a sk-learn model to ONNX. This has the following inputs:

* `model`:  The sk-learn model we're converting.
* `cols`: The number of inputs the model expects
* `input_type`: Determines how to manage float values, which can either be `DoubleTensorType` or `FloatTensorType`.

```python
# convert model to ONNX

def model_to_onnx(model, cols, *, input_type='Double'):
    input_type_lower=input_type.lower()
    # How to manage float values
    if input_type=='Double':
        tensor_type=DoubleTensorType
    elif input_type=='Float':
        tensor_type=FloatTensorType
    else:
        raise ValueError("bad input type")
    tensor_size=cols
    initial_type=[(f'{input_type_lower}_input', tensor_type([None, tensor_size]))]
    onnx_model=onnxmltools.convert_sklearn(model,initial_types=initial_type)
    return onnx_model
```

With our method defined, now it's time to convert.  Let's load our sk-learn model and save it into the variable `sklearn_model`.

```python
# pickle the model, so I can try the Wallaroo converter on it

sklearn_model = pickle.load(open('./demand_curve.pickle', 'rb'))
```

Now we'll convert our `sklearn-model` into the variable `onnx_model` using our `model_to_onnx` method.  Recall that our `sklearn-model` has 3 columns.

```python
onnx_model_converted = model_to_onnx(sklearn_model, 3)
```

Now we can save our model to a `onnx` file.

```python
onnx.save_model(onnx_model_converted, "demand_curve.onnx")
```
