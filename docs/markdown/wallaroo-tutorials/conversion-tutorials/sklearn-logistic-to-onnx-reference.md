This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository]https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_conversion/sklearn-classification-to-onnx).

## sk-learn Logistic Model to ONNX Outside Wallaroo

The following tutorial is a brief example of how to convert a [scikit-learn](https://scikit-learn.org/stable/) (aka sk-learn) **Classification ML model** to the [ONNX](https://onnx.ai/ ) format for use with Wallaroo.

This tutorial assumes that you have a Wallaroo instance and are running this Notebook from the Wallaroo Jupyter Hub service.

This tutorial provides the following:

* `isolet_logistic_model_numclass.pickle`: a logistic trained sk-learn model.
    This model contains 617 columns.
* `isolet_test_data.tsv`:  A test file that can be used to verify the output of the converted logistic model.
* `test-converted-sklearn-logistics-to-onnx.ipynb`: This Jupyter Notebook demonstrates the use of the converted sk-learn logistic ML model in ONNX with Wallaroo.

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
from skl2onnx import convert_sklearn
```

Now we can determine the correct ONNX Target Opset for our libraries.

```python
# figure out the correct opset

from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())
TARGET_OPSET
```

    13

With the `TARGET_OPSET` determined, we can convert our sklearn logistic model to onnx.

* **IMPORTANT NOTE**:  Note that for the conversion process, `zipmap` is **disabled**.

Load the model that we will be converting:

```python
# convert model to ONNX

# load the model

with open("./isolet_logistic_model_numclass.pickle", "rb") as f:
    logistic_model = pickle.load(f)
```

We already know the number of columns, so we'll set that variable in the next step.

```python
# Set the number of columns

ncols = 617
```

Next up is to set the options.  As a reminder **zipmap must be disabled**.

```python
## Set the options

initial_type = [('float_input', FloatTensorType([None, ncols]))]
options = {id(logistic_model): {'zipmap': False}} # here we turn off the zipmap
```

With everything ready, we can now convert the sk-learn Logistics model to ONNX, and store it in the variable `onnx_model_converted`.

```python
## Run the conversion

onnx_model_converted = convert_sklearn(logistic_model, initial_types=initial_type, options=options,
                       target_opset=TARGET_OPSET)
```

Now we can save our model to a `onnx` file.  Once complete, we can run it through the `Logistic Version of the Isolet Model Test in Wallaroo` available at the [Wallaroo Tutorials repository]https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_conversion/sklearn-classification-to-onnx) to verify it.

```python
# Export the model to a file
onnx.save_model(onnx_model_converted, "isolet_logistic_model_numclass.onnx")
```
