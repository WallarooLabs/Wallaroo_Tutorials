This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_conversion/pytorch-to-onnx).

## How to Convert PyTorch to ONNX

The following tutorial is a brief example of how to convert a [PyTorth](https://pytorch.org/) (aka sk-learn) ML model to [ONNX](https://onnx.ai/).  This allows organizations that have trained sk-learn models to convert them and use them with Wallaroo.

This tutorial assumes that you have a Wallaroo instance and are running this Notebook from the Wallaroo Jupyter Hub service.  This sample code is based on the guide [Convert your PyTorch model to ONNX](https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model).

This tutorial provides the following:

* `pytorchbikeshare.pt`: a RandomForestRegressor PyTorch model.  This model has a total of 58 inputs, and uses the class `BikeShareRegressor`.

## Conversion Process

### Libraries

The first step is to import our libraries we will be using.  For this example, the PyTorth `torch` library will be imported into this kernel.

```python
# the Pytorch libraries
# Import into this kernel

import sys
!{sys.executable} -m pip install torch

import torch
import torch.onnx 
```

## Load the Model

To load a PyTorch model into a variable, the model's `class` has to be defined.  For out example we are using the `BikeShareRegressor` class as defined below.

```python
class BikeShareRegressor(torch.nn.Module):
    def __init__(self):
        super(BikeShareRegressor, self).__init__()

        
        self.net = nn.Sequential(nn.Linear(input_size, l1),
                                 torch.nn.ReLU(),
                                 torch.nn.Dropout(p=dropout),
                                 nn.BatchNorm1d(l1),
                                 nn.Linear(l1, l2),
                                 torch.nn.ReLU(),
                                 torch.nn.Dropout(p=dropout),                                
                                 nn.BatchNorm1d(l2),                                                                                                   
                                 nn.Linear(l2, output_size))

    def forward(self, x):
        return self.net(x)
```

Now we will load the model into the variable `pytorch_tobe_converted`.

```python
# load the Pytorch model
model = torch.load("./pytorch_bikesharingmodel.pt")
```

### Convert_ONNX Inputs

Now we will define our method `Convert_ONNX()` which has the following inputs:
    
* **PyTorchModel**: the PyTorch we are converting.
* **modelInputs**: the model input or tuple for multiple inputs.
* **onnxPath**: The location to save the onnx file.

* **opset_version**: The ONNX version to export to.
* **input_names**: Array of the model's input names.
* **output_names**:  Array of the model's output names.
* **dynamic_axes**:  Sets variable length axes in the format, replacing the `batch_size` as necessary:
  `{'modelInput' : { 0 : 'batch_size'}, 'modelOutput' : {0 : 'batch_size'}}`
* **export_params**:  Whether to store the trained parameter weight inside the model file.  Defaults to `True`.
* **do_constant_folding**: Sets whether to execute constant folding for optimization.  Defaults to `True`.
  

```python
#Function to Convert to ONNX 
def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         pypath,       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=15,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes = {'modelInput' : {0 : 'batch_size'}, 'modelOutput' : {0 : 'batch_size'}} # variable length axes 
    ) 
    print(" ") 
    print('Model has been converted to ONNX') 
```

### Convert the Model

We'll now set our variables and run our conversion.  For out example, the `input_size` is known to be 58, and the `device` value we'll derive from `torch.cuda`.  We'll also set the ONNX version for exporting to 15.

```python
pypath = "pytorchbikeshare.onnx"

input_size = 58

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

onnx_opset_version = 15

# Set up some dummy input tensor for the model
dummy_input = torch.randn(1, input_size, requires_grad=True).to(device)

Convert_ONNX()
```

     
    Model has been converted to ONNX

## Conclusion

And now our conversion is complete.  Please feel free to use this sample code in your own projects.
