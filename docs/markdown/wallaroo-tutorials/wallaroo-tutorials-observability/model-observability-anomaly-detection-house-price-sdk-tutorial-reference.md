The following tutorials are available from the [Wallaroo Tutorials Repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/20240205-data-schemas/model-observability/model-observabilty-anomaly-detection-ccfraud-sdk-tutorial).

## Wallaroo Model Observability: Anomaly Detection with House Price Prediction

The following tutorial demonstrates the use case of detecting anomalies:  inference input or output data that does not match typical validations.

Wallaroo provides **validations** to detect anomalous data from inference inputs and outputs.  Validations are added to a Wallaroo pipeline with the `wallaroo.pipeline.add_validations` method.

Adding validations takes the format:

```python
pipeline.add_validations(
    validation_name_01 = polars.col(in|out.{column_name}) EXPRESSION,
    validation_name_02 = polars.col(in|out.{column_name}) EXPRESSION
    ...{additional rules}
)
```

* `validation_name`: The user provided name of the validation.  The names must match Python variable naming requirements.
  * **IMPORTANT NOTE**: Using the name `count` as a validation name **returns a warning**.  Any validation rules named `count` are dropped upon request and an warning returned.
* `polars.col(in|out.{column_name})`: Specifies the **input** or **output** for a specific field aka "column" in an inference result.  Wallaroo inference requests are in the format `in.{field_name}` for **inputs**, and `out.{field_name}` for **outputs**.
  * More than one field can be selected, as long as they follow the rules of the [polars 0.18 Expressions library](https://docs.pola.rs/docs/python/version/0.18/reference/expressions/index.html).
* `EXPRESSION`:  The expression to validate. When the expression returns **True**, that indicates an anomaly detected.

The [`polars` library version 0.18.5](https://docs.pola.rs/docs/python/version/0.18/index.html) is used to create the validation rule.  This is installed by default with the Wallaroo SDK.  This provides a powerful range of comparisons to organizations tracking anomalous data from their ML models.

When validations are added to a pipeline, inference request outputs return the following fields:

| Field | Type | Description |
|---|---|---|
| **anomaly.count** | **Integer** | The total of all validations that returned **True**. |
| **anomaly.{validation name}** | **Bool** | The output of the validation `{validation_name}`. |

When validation returns `True`, **an anomaly is detected**.

For example, adding the validation `fraud` to the following pipeline returns `anomaly.count` of `1` when the validation `fraud` returns `True`.  The validation `fraud` returns `True` when the **output** field **dense_1** at index **0** is greater than 0.9.

```python
sample_pipeline = wallaroo.client.build_pipeline("sample-pipeline")
sample_pipeline.add_model_step(model)

# add the validation
sample_pipeline.add_validations(
    fraud=pl.col("out.dense_1").list.get(0) > 0.9,
    )

# deploy the pipeline
sample_pipeline.deploy()

# sample inference
display(sample_pipeline.infer_from_file("dev_high_fraud.json", data_format='pandas-records'))
```

|&nbsp;|time|in.tensor|out.dense_1|anomaly.count|anomaly.fraud|
|---|---|---|---|---|---|
|0|2024-02-02 16:05:42.152|[1.0678324729, 18.1555563975, -1.6589551058, 5...]|[0.981199]|1|True|

### Detecting Anomalies from Inference Request Results

When an inference request is submitted to a Wallaroo pipeline with validations, the following fields are output:

| Field | Type | Description |
|---|---|---|
| **anomaly.count** | **Integer** | The total of all validations that returned **True**. |
| **anomaly.{validation name}** | **Bool** | The output of each pipeline validation `{validation_name}`. |

For example, adding the validation `fraud` to the following pipeline returns `anomaly.count` of `1` when the validation `fraud` returns `True`.

```python
sample_pipeline = wallaroo.client.build_pipeline("sample-pipeline")
sample_pipeline.add_model_step(model)

# add the validation
sample_pipeline.add_validations(
    fraud=pl.col("out.dense_1").list.get(0) > 0.9,
    )

# deploy the pipeline
sample_pipeline.deploy()

# sample inference
display(sample_pipeline.infer_from_file("dev_high_fraud.json", data_format='pandas-records'))
```

|&nbsp;|time|in.tensor|out.dense_1|anomaly.count|anomaly.fraud|
|---|---|---|---|---|---|
|0|2024-02-02 16:05:42.152|[1.0678324729, 18.1555563975, -1.6589551058, 5...]|[0.981199]|1|True|

## Anomaly Detection Demonstration

The following demonstrates how to:

* Upload a house price ML model trained to predict house prices based on a set of inputs.  This outputs the field `variable` as an float which is the predicted house price.
* Add the house price model as a pipeline step.
* Add the validation `too_high` to detect when a house price exceeds a certain value.
* Deploy the pipeline and performing sample inferences on it.
* Perform sample inferences to show when the `too_high` validation returns `True` and `False`.
* Perform sample inference with different datasets to show enable or disable certain fields from displaying in the inference results.

### Prerequisites

* Wallaroo version 2023.4.1 (@TODO: Verify) and above.
* [`polars` version 0.18.5](https://docs.pola.rs/docs/python/version/0.18/index.html).  This is installed by default with the Wallaroo SDK.

## Tutorial Steps

### Load Libraries

The first step is to import the libraries used in this notebook.

```python
import wallaroo
wallaroo.__version__

```

    '2023.4.1+379cb6b8a'

### Connect to the Wallaroo Instance through the User Interface

The next step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
wl = wallaroo.Client()
```

### Create a New Workspace

We'll use the SDK below to create our workspace then assign as our **current workspace**.  The current workspace is used by the Wallaroo SDK for where to upload models, create pipelines, etc.  We'll also set up variables for our models and pipelines down the road, so we have one spot to change names to whatever fits your organization's standards best.

Before starting, verify that the workspace name is unique in your Wallaroo instance.

```python
def get_workspace(name, client):
    workspace = None
    for ws in client.list_workspaces():
        if ws.name() == name:
            workspace= ws
    if(workspace == None):
        workspace = client.create_workspace(name)
    return workspace

workspace_name = 'validation-house-price-demonstration'
pipeline_name = 'validation-demo'
model_name = 'anomaly-housing-model'
model_file_name = './models/rf_model.onnx'
```

```python
workspace = get_workspace(workspace_name, wl)
wl.set_current_workspace(workspace)
```

    {'name': 'validation-house-price-demonstration', 'id': 25, 'archived': False, 'created_by': 'c97d480f-6064-4537-b18e-40fb1864b4cd', 'created_at': '2024-02-08T21:52:50.354176+00:00', 'models': [{'name': 'anomaly-housing-model', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 2, 8, 21, 52, 51, 671284, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 2, 8, 21, 52, 51, 671284, tzinfo=tzutc())}], 'pipelines': [{'name': 'validation-demo', 'create_time': datetime.datetime(2024, 2, 8, 21, 52, 52, 879885, tzinfo=tzutc()), 'definition': '[]'}]}

### Upload the Model

Upload the model to the Wallaroo workspace with the `wallaroo.client.upload_model` method.  Our house price ML model is a Wallaroo Default Runtime of type `ONNX`, so all we need is the model name, the model file path, and the framework type of `wallaroo.framework.Framework.ONNX`.

```python
model = (wl.upload_model(model_name, 
                                 model_file_name, 
                                 framework=wallaroo.framework.Framework.ONNX)
                )
```

### Build the Pipeline

Pipelines are build with the `wallaroo.client.build_pipeline` method, which takes the pipeline name.  This will create the pipeline in our default workspace.  Note that if there are any existing pipelines with the same name in this workspace, this method will retrieve that pipeline for this SDK session.

Once the pipeline is created, we add the ccfraud model as our pipeline step.

```python
sample_pipeline = wl.build_pipeline(pipeline_name)
sample_pipeline.clear()
sample_pipeline = sample_pipeline.add_model_step(model)

```

```python
import onnx

model = onnx.load(model_file_name)
output =[node.name for node in model.graph.output]

input_all = [node.name for node in model.graph.input]
input_initializer =  [node.name for node in model.graph.initializer]
net_feed_input = list(set(input_all)  - set(input_initializer))

print('Inputs: ', net_feed_input)
print('Outputs: ', output)
```

    Inputs:  ['float_input']
    Outputs:  ['variable']

### Add Validation

Now we add our validation to our new pipeline.  We will give it the following configuration.

* Validation Name: `too_high`
* Validation Field: `out.variable`
* Validation Field Index: `0`
* Validation Expression:  Values greater than `1000000.0`.

The `polars` library is required for creating the validation.  We will import the polars library, then add our validation to the pipeline.

* **IMPORTANT NOTE**:  Validation names must be unique **per pipeline**.  If a validation of the same name is added, both are included in the pipeline validations, but only **most recent validation with the same name** is displayed with the inference results.  Anomalies detected by multiple validations of the same name are added to the `anomaly.count` inference result field.

```python
import polars as pl

sample_pipeline = sample_pipeline.add_validations(
    too_high=pl.col("out.variable").list.get(0) > 1000000.0
)
```

### Display Pipeline And Validation Steps

The method `wallaroo.pipeline.steps()` shows the current pipeline steps. The added validations are in the `Check` field.  This is used for demonstration purposes to show the added validation to the pipeline.

```python
sample_pipeline.steps()
```

    [{'ModelInference': {'models': [{'name': 'anomaly-housing-model', 'version': '9a76a2cf-9ea3-4978-8fd5-005d0280e661', 'sha': 'e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6'}]}},
     {'Check': {'tree': ['{"Alias":[{"BinaryExpr":{"left":{"Function":{"input":[{"Column":"out.variable"},{"Literal":{"Int32":0}}],"function":{"ListExpr":"Get"},"options":{"collect_groups":"ApplyFlat","fmt_str":"","input_wildcard_expansion":false,"auto_explode":true,"cast_to_supertypes":false,"allow_rename":false,"pass_name_to_apply":false,"changes_length":false,"check_lengths":true,"allow_group_aware":true}}},"op":"Gt","right":{"Literal":{"Float64":1000000.0}}}},"too_high"]}']}}]

### Deploy Pipeline

With the pipeline steps set and the validations created, we deploy the pipeline.  Because of it's size, we will only allocate `0.1` cpu from the cluster for the pipeline's use.

```python
deploy_config = wallaroo.deployment_config.DeploymentConfigBuilder() \
    .cpus(0.25)\
    .build()

sample_pipeline.deploy(deployment_config=deploy_config)
```

    Waiting for undeployment - this will take up to 45s ..................................... ok
    Waiting for deployment - this will take up to 45s ......... ok

<table><tr><th>name</th> <td>validation-demo</td></tr><tr><th>created</th> <td>2024-02-08 21:52:52.879885+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-08 22:14:13.217863+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>2a8d204a-f359-4f02-b558-950dbab28dc6, 424c7f24-ca65-45af-825f-64d3e9f8e8c8, 190226c9-a536-4457-9851-a68ef968b6fc, e10707fd-75b8-4386-8466-e58dc13d2828, 53b87a42-8498-475a-bf30-73fdebbf85cc</td></tr><tr><th>steps</th> <td>anomaly-housing-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Sample Inferences

Two sample inferences are performed with the method `wallaroo.pipeline.infer_from_file` that takes either a pandas Record JSON file or an Apache Arrow table as the input.

For our demonstration, we will use the following pandas Record JSON file with the following sample data:

* `./data/houseprice_5000_data.json`: A sample sets of 5000 houses to generates a range of predicted values.

The inference request returns a pandas DataFrame.

Each of the inference outputs will include the following fields:

| Field | Type | Description |
|---|---|---|
| **time** | **DateTime** | The DateTime of the inference request. |
| **in.{input_field_name}** | Input Dependent | Each input field submitted is labeled as `in.{input_field_name}` in the inference request result.  For our example, this is `tensor`, so the input field in the returned inference request is `in.tensor`. |
| **out.{model_output_field_name}** | Output Dependent | Each field output by the ML model is labeled as `out.{model_output_field_name}` in the inference request result.  For our example, the ccfraud model returns `dense_1` as its output field, so the output field in the returned inference request is `out.dense_1`. |
| **anomaly.count**	| Integer | The total number of validations that returned `True`. |
| **anomaly.{validation_name} | Bool | Each validation added to the pipeline is returned as `anomaly.{validation_name}`, and returns either `True` if the validation returns `True`, indicating an anomaly is found, or `False` for an anomaly for the validation is not found.  For our example, we will have `anomaly.fraud` returned.

```python
results = sample_pipeline.infer_from_file('./data/test-1000.df.json')
# first 20 results
display(results.head(20))

# only results that trigger the anomaly too_high
results.loc[results['anomaly.too_high'] == True]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.float_input</th>
      <th>out.variable</th>
      <th>anomaly.count</th>
      <th>anomaly.too_high</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[718013.75]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0,...</td>
      <td>[615094.56]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, ...</td>
      <td>[448627.72]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[758714.2]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4....</td>
      <td>[513264.7]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 2.0, 2140.0, 4923.0, 1.0, 0.0, 0.0, 4.0,...</td>
      <td>[668288.0]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.5, 3590.0, 5334.0, 2.0, 0.0, 2.0, 3.0,...</td>
      <td>[1004846.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 2.0, 1280.0, 960.0, 2.0, 0.0, 0.0, 3.0, ...</td>
      <td>[684577.2]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 2.5, 2820.0, 15000.0, 2.0, 0.0, 0.0, 4.0...</td>
      <td>[727898.1]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 2.25, 1790.0, 11393.0, 1.0, 0.0, 0.0, 3....</td>
      <td>[559631.1]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 1.5, 1010.0, 7683.0, 1.5, 0.0, 0.0, 5.0,...</td>
      <td>[340764.53]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 2.0, 1270.0, 1323.0, 3.0, 0.0, 0.0, 3.0,...</td>
      <td>[442168.06]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 1.75, 2070.0, 9120.0, 1.0, 0.0, 0.0, 4.0...</td>
      <td>[630865.6]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 1.0, 1620.0, 4080.0, 1.5, 0.0, 0.0, 3.0,...</td>
      <td>[559631.1]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.25, 3990.0, 9786.0, 2.0, 0.0, 0.0, 3.0...</td>
      <td>[909441.1]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 2.0, 1780.0, 19843.0, 1.0, 0.0, 0.0, 3.0...</td>
      <td>[313096.0]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 2.5, 2130.0, 6003.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[404040.8]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 1.75, 1660.0, 10440.0, 1.0, 0.0, 0.0, 3....</td>
      <td>[292859.5]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 2.5, 2110.0, 4118.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[338357.88]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 2.25, 2200.0, 11250.0, 1.5, 0.0, 0.0, 5....</td>
      <td>[682284.6]</td>
      <td>0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.float_input</th>
      <th>out.variable</th>
      <th>anomaly.count</th>
      <th>anomaly.too_high</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.5, 3590.0, 5334.0, 2.0, 0.0, 2.0, 3.0,...</td>
      <td>[1004846.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0...</td>
      <td>[1514079.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 4.5, 5120.0, 41327.0, 2.0, 0.0, 0.0, 3.0...</td>
      <td>[1204324.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>63</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.0, 4040.0, 19700.0, 2.0, 0.0, 0.0, 3.0...</td>
      <td>[1028923.06]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>110</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 2.5, 3470.0, 20445.0, 2.0, 0.0, 0.0, 4.0...</td>
      <td>[1412215.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>130</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 2.75, 2620.0, 13777.0, 1.5, 0.0, 2.0, 4....</td>
      <td>[1223839.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>133</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[5.0, 2.25, 3320.0, 13138.0, 1.0, 0.0, 2.0, 4....</td>
      <td>[1108000.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>154</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 2.75, 3800.0, 9606.0, 2.0, 0.0, 0.0, 3.0...</td>
      <td>[1039781.25]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>160</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[5.0, 3.5, 4150.0, 13232.0, 2.0, 0.0, 0.0, 3.0...</td>
      <td>[1042119.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>210</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.5, 4300.0, 70407.0, 2.0, 0.0, 0.0, 3.0...</td>
      <td>[1115275.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>239</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.25, 5010.0, 49222.0, 2.0, 0.0, 0.0, 5....</td>
      <td>[1092274.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>248</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.75, 4410.0, 8112.0, 3.0, 0.0, 4.0, 3.0...</td>
      <td>[1967344.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>255</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.0, 4750.0, 21701.0, 1.5, 0.0, 0.0, 5.0...</td>
      <td>[2002393.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>271</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[5.0, 3.25, 5790.0, 13726.0, 2.0, 0.0, 3.0, 3....</td>
      <td>[1189654.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>281</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 3.0, 3570.0, 6250.0, 2.0, 0.0, 2.0, 3.0,...</td>
      <td>[1124493.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>282</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 2.75, 3170.0, 34850.0, 1.0, 0.0, 0.0, 5....</td>
      <td>[1227073.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>283</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 2.75, 3260.0, 19542.0, 1.0, 0.0, 0.0, 4....</td>
      <td>[1364650.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>285</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 2.75, 4020.0, 18745.0, 2.0, 0.0, 4.0, 4....</td>
      <td>[1322835.9]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>323</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 3.0, 2480.0, 5500.0, 2.0, 0.0, 3.0, 3.0,...</td>
      <td>[1100884.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>351</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[5.0, 4.0, 4660.0, 9900.0, 2.0, 0.0, 2.0, 4.0,...</td>
      <td>[1058105.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>360</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.5, 3770.0, 8501.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[1169643.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>398</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 2.25, 2390.0, 7875.0, 1.0, 0.0, 1.0, 3.0...</td>
      <td>[1364149.9]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>414</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[5.0, 3.5, 5430.0, 10327.0, 2.0, 0.0, 2.0, 3.0...</td>
      <td>[1207858.6]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>443</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[5.0, 4.0, 4360.0, 8030.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[1160512.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>497</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 2.5, 4090.0, 11225.0, 2.0, 0.0, 0.0, 3.0...</td>
      <td>[1048372.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>513</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.25, 3320.0, 8587.0, 3.0, 0.0, 0.0, 3.0...</td>
      <td>[1130661.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>520</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[5.0, 3.75, 4170.0, 8142.0, 2.0, 0.0, 2.0, 3.0...</td>
      <td>[1098628.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>530</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 4.25, 3500.0, 8750.0, 1.0, 0.0, 4.0, 5.0...</td>
      <td>[1140733.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>535</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.5, 4460.0, 16271.0, 2.0, 0.0, 2.0, 3.0...</td>
      <td>[1208638.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>556</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.5, 4285.0, 9567.0, 2.0, 0.0, 1.0, 5.0,...</td>
      <td>[1886959.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>623</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.25, 4240.0, 25639.0, 2.0, 0.0, 3.0, 3....</td>
      <td>[1156651.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>624</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.5, 3440.0, 9776.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[1124493.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>634</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.25, 4700.0, 38412.0, 2.0, 0.0, 0.0, 3....</td>
      <td>[1164589.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>651</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 3.0, 3920.0, 13085.0, 2.0, 1.0, 4.0, 4.0...</td>
      <td>[1452224.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>658</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 3.25, 3230.0, 7800.0, 2.0, 0.0, 3.0, 3.0...</td>
      <td>[1077279.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>671</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 3.5, 3080.0, 6495.0, 2.0, 0.0, 3.0, 3.0,...</td>
      <td>[1122811.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>685</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 2.5, 4200.0, 35267.0, 2.0, 0.0, 0.0, 3.0...</td>
      <td>[1181336.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>686</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.25, 4160.0, 47480.0, 2.0, 0.0, 0.0, 3....</td>
      <td>[1082353.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>698</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 4.5, 5770.0, 10050.0, 1.0, 0.0, 3.0, 5.0...</td>
      <td>[1689843.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>711</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 2.5, 5403.0, 24069.0, 2.0, 1.0, 4.0, 4.0...</td>
      <td>[1946437.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>720</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[5.0, 3.0, 3420.0, 18129.0, 2.0, 0.0, 0.0, 3.0...</td>
      <td>[1325961.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>722</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 3.25, 4560.0, 13363.0, 1.0, 0.0, 4.0, 3....</td>
      <td>[2005883.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>726</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[5.0, 3.5, 4200.0, 5400.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[1052898.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>737</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.25, 2980.0, 7000.0, 2.0, 0.0, 3.0, 3.0...</td>
      <td>[1156206.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>740</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 4.5, 6380.0, 88714.0, 2.0, 0.0, 0.0, 3.0...</td>
      <td>[1355747.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>782</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[5.0, 4.25, 4860.0, 9453.0, 1.5, 0.0, 1.0, 5.0...</td>
      <td>[1910823.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>798</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 2.5, 2790.0, 5450.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[1097757.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>818</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 4.0, 4620.0, 130208.0, 2.0, 0.0, 0.0, 3....</td>
      <td>[1164589.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>827</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 2.5, 3340.0, 10422.0, 2.0, 0.0, 0.0, 3.0...</td>
      <td>[1103101.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>828</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[5.0, 3.5, 3760.0, 10207.0, 2.0, 0.0, 0.0, 3.0...</td>
      <td>[1489624.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>901</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 2.25, 4470.0, 60373.0, 2.0, 0.0, 0.0, 3....</td>
      <td>[1208638.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>912</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[3.0, 2.25, 2960.0, 8330.0, 1.0, 0.0, 3.0, 4.0...</td>
      <td>[1178314.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>919</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.25, 5180.0, 19850.0, 2.0, 0.0, 3.0, 3....</td>
      <td>[1295531.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>941</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.75, 3770.0, 4000.0, 2.5, 0.0, 0.0, 5.0...</td>
      <td>[1182821.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>965</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[6.0, 4.0, 5310.0, 12741.0, 2.0, 0.0, 2.0, 3.0...</td>
      <td>[2016006.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>973</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[5.0, 2.0, 3540.0, 9970.0, 2.0, 0.0, 3.0, 3.0,...</td>
      <td>[1085835.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>997</th>
      <td>2024-02-08 22:14:47.075</td>
      <td>[4.0, 3.25, 2910.0, 1880.0, 2.0, 0.0, 3.0, 5.0...</td>
      <td>[1060847.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
  </tbody>
</table>

### Other Validation Examples

The following are additional examples of validations.

#### Multiple Validations

The following uses multiple validations to check for anomalies.  We still use `fraud` which detects outputs that are greater than `1000000.0`.  The second validation `too_low` triggers an anomaly when the `out.variable` is under `250000.0`.

After the validations are added, the pipeline is redeployed to "set" them.

```python
sample_pipeline = sample_pipeline.add_validations(
    too_low=pl.col("out.variable").list.get(0) < 250000.0
)

deploy_config = wallaroo.deployment_config.DeploymentConfigBuilder() \
    .cpus(0.1)\
    .build()
sample_pipeline.undeploy()
sample_pipeline.deploy(deployment_config=deploy_config)
```

    Waiting for undeployment - this will take up to 45s ..................................... ok
    Waiting for deployment - this will take up to 45s .............. ok

<table><tr><th>name</th> <td>validation-demo</td></tr><tr><th>created</th> <td>2024-02-08 21:52:52.879885+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-08 22:16:50.265231+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>053580e2-f73d-4b63-8c9c-0b5e06be96c2, 2a8d204a-f359-4f02-b558-950dbab28dc6, 424c7f24-ca65-45af-825f-64d3e9f8e8c8, 190226c9-a536-4457-9851-a68ef968b6fc, e10707fd-75b8-4386-8466-e58dc13d2828, 53b87a42-8498-475a-bf30-73fdebbf85cc</td></tr><tr><th>steps</th> <td>anomaly-housing-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
results = sample_pipeline.infer_from_file('./data/test-1000.df.json')
# first 20 results
display(results.head(20))

# only results that trigger the anomaly too_high
results.loc[results['anomaly.too_high'] == True]

# only results that trigger the anomaly too_low
results.loc[results['anomaly.too_low'] == True]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.float_input</th>
      <th>out.variable</th>
      <th>anomaly.count</th>
      <th>anomaly.too_high</th>
      <th>anomaly.too_low</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[718013.75]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0,...</td>
      <td>[615094.56]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, ...</td>
      <td>[448627.72]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[758714.2]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4....</td>
      <td>[513264.7]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[3.0, 2.0, 2140.0, 4923.0, 1.0, 0.0, 0.0, 4.0,...</td>
      <td>[668288.0]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[4.0, 3.5, 3590.0, 5334.0, 2.0, 0.0, 2.0, 3.0,...</td>
      <td>[1004846.5]</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[3.0, 2.0, 1280.0, 960.0, 2.0, 0.0, 0.0, 3.0, ...</td>
      <td>[684577.2]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[4.0, 2.5, 2820.0, 15000.0, 2.0, 0.0, 0.0, 4.0...</td>
      <td>[727898.1]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[3.0, 2.25, 1790.0, 11393.0, 1.0, 0.0, 0.0, 3....</td>
      <td>[559631.1]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[3.0, 1.5, 1010.0, 7683.0, 1.5, 0.0, 0.0, 5.0,...</td>
      <td>[340764.53]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[3.0, 2.0, 1270.0, 1323.0, 3.0, 0.0, 0.0, 3.0,...</td>
      <td>[442168.06]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[4.0, 1.75, 2070.0, 9120.0, 1.0, 0.0, 0.0, 4.0...</td>
      <td>[630865.6]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[4.0, 1.0, 1620.0, 4080.0, 1.5, 0.0, 0.0, 3.0,...</td>
      <td>[559631.1]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[4.0, 3.25, 3990.0, 9786.0, 2.0, 0.0, 0.0, 3.0...</td>
      <td>[909441.1]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[4.0, 2.0, 1780.0, 19843.0, 1.0, 0.0, 0.0, 3.0...</td>
      <td>[313096.0]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[4.0, 2.5, 2130.0, 6003.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[404040.8]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[3.0, 1.75, 1660.0, 10440.0, 1.0, 0.0, 0.0, 3....</td>
      <td>[292859.5]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[3.0, 2.5, 2110.0, 4118.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[338357.88]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[4.0, 2.25, 2200.0, 11250.0, 1.5, 0.0, 0.0, 5....</td>
      <td>[682284.6]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.float_input</th>
      <th>out.variable</th>
      <th>anomaly.count</th>
      <th>anomaly.too_high</th>
      <th>anomaly.too_low</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[2.0, 2.0, 1390.0, 1302.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[249227.8]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>69</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[3.0, 1.75, 1050.0, 9871.0, 1.0, 0.0, 0.0, 5.0...</td>
      <td>[236238.66]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[3.0, 1.75, 1070.0, 8100.0, 1.0, 0.0, 0.0, 4.0...</td>
      <td>[236238.66]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[3.0, 2.5, 1340.0, 3011.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[244380.27]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>124</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[4.0, 1.5, 1200.0, 10890.0, 1.0, 0.0, 0.0, 5.0...</td>
      <td>[241330.19]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>939</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[3.0, 1.0, 1150.0, 4800.0, 1.5, 0.0, 0.0, 4.0,...</td>
      <td>[240834.92]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>946</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[2.0, 1.0, 780.0, 6250.0, 1.0, 0.0, 0.0, 3.0, ...</td>
      <td>[236815.78]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>948</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[1.0, 1.0, 620.0, 8261.0, 1.0, 0.0, 0.0, 3.0, ...</td>
      <td>[236815.78]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>962</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[3.0, 1.0, 1190.0, 7500.0, 1.0, 0.0, 0.0, 5.0,...</td>
      <td>[241330.19]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>991</th>
      <td>2024-02-08 22:17:23.630</td>
      <td>[2.0, 1.0, 870.0, 8487.0, 1.0, 0.0, 0.0, 4.0, ...</td>
      <td>[236238.66]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>62 rows × 6 columns</p>

#### Compound Validations

The following combines multiple field checks into a single validation.  For this, we will check for values of `out.variable` that are between 500000 and 1000000.

Each expression is separated by `()`.  For example:

* Expression 1: `pl.col("out.variable").list.get(0) < 1000000.0`
* Expression 2: `pl.col("out.variable").list.get(0) > 500000.0`
* Compound Expression: `(pl.col("out.variable").list.get(0) < 1000000.0) & (pl.col("out.variable").list.get(0) > 500000.0)`

```python
sample_pipeline = sample_pipeline.add_validations(
    in_between=(pl.col("out.variable").list.get(0) < 1000000.0) & (pl.col("out.variable").list.get(0) > 500000.0)
)

deploy_config = wallaroo.deployment_config.DeploymentConfigBuilder() \
    .cpus(0.1)\
    .build()
sample_pipeline.undeploy()
sample_pipeline.deploy(deployment_config=deploy_config)
```

    Waiting for undeployment - this will take up to 45s .................................... ok
    Waiting for deployment - this will take up to 45s ............. ok

<table><tr><th>name</th> <td>validation-demo</td></tr><tr><th>created</th> <td>2024-02-08 21:52:52.879885+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-08 22:18:18.613710+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>016728d7-3948-467f-8ecc-e0594f406884, 053580e2-f73d-4b63-8c9c-0b5e06be96c2, 2a8d204a-f359-4f02-b558-950dbab28dc6, 424c7f24-ca65-45af-825f-64d3e9f8e8c8, 190226c9-a536-4457-9851-a68ef968b6fc, e10707fd-75b8-4386-8466-e58dc13d2828, 53b87a42-8498-475a-bf30-73fdebbf85cc</td></tr><tr><th>steps</th> <td>anomaly-housing-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
results = sample_pipeline.infer_from_file('./data/test-1000.df.json')

results.loc[results['anomaly.in_between'] == True] 
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.float_input</th>
      <th>out.variable</th>
      <th>anomaly.count</th>
      <th>anomaly.in_between</th>
      <th>anomaly.too_high</th>
      <th>anomaly.too_low</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-08 22:18:32.886</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[718013.75]</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02-08 22:18:32.886</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0,...</td>
      <td>[615094.56]</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-02-08 22:18:32.886</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[758714.2]</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-02-08 22:18:32.886</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4....</td>
      <td>[513264.7]</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2024-02-08 22:18:32.886</td>
      <td>[3.0, 2.0, 2140.0, 4923.0, 1.0, 0.0, 0.0, 4.0,...</td>
      <td>[668288.0]</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>989</th>
      <td>2024-02-08 22:18:32.886</td>
      <td>[4.0, 2.75, 2500.0, 4950.0, 2.0, 0.0, 0.0, 3.0...</td>
      <td>[700271.56]</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>993</th>
      <td>2024-02-08 22:18:32.886</td>
      <td>[3.0, 2.5, 2140.0, 8925.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[669645.5]</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>995</th>
      <td>2024-02-08 22:18:32.886</td>
      <td>[3.0, 2.5, 2900.0, 23550.0, 1.0, 0.0, 0.0, 3.0...</td>
      <td>[827411.0]</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>998</th>
      <td>2024-02-08 22:18:32.886</td>
      <td>[3.0, 1.75, 2910.0, 37461.0, 1.0, 0.0, 0.0, 4....</td>
      <td>[706823.56]</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>999</th>
      <td>2024-02-08 22:18:32.886</td>
      <td>[3.0, 2.0, 2005.0, 7000.0, 1.0, 0.0, 0.0, 3.0,...</td>
      <td>[581003.0]</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>395 rows × 7 columns</p>

### Specify Dataset

Wallaroo inference requests allow datasets to be excluded or included with the `dataset_exclude` and `dataset` parameters.

| Parameter | Type | Description|
|---|---|---|
| **dataset_exclude** | *List(String)* | The list of datasets to exclude.  Values include:  <ul><li>`metadata`: Returns inference time per model, last model used, and other parameters.</li><li>`anomaly`: The anomaly results of all validations added to the pipeline.</li></ul> |
| **dataset** | *List(String)* | The list of datasets and fields to include. |

For our example, we will **exclude** the `anomaly` dataset, but **include** the datasets `'time'`, `'in'`, `'out'`, `'anomaly.count'`.  Note that while we exclude `anomaly`, we override that with by setting the anomaly field `'anomaly.count'` in our `dataset` parameter.

```python
sample_pipeline.infer_from_file('./data/test-1000.df.json', 
                                dataset_exclude=['anomaly'], 
                                dataset=['time', 'in', 'out', 'anomaly.count']
                                )
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.float_input</th>
      <th>out.variable</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-08 22:19:04.558</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[718013.75]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02-08 22:19:04.558</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0,...</td>
      <td>[615094.56]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-02-08 22:19:04.558</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, ...</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-02-08 22:19:04.558</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0,...</td>
      <td>[758714.2]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-02-08 22:19:04.558</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4....</td>
      <td>[513264.7]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>2024-02-08 22:19:04.558</td>
      <td>[3.0, 2.5, 2900.0, 23550.0, 1.0, 0.0, 0.0, 3.0...</td>
      <td>[827411.0]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>996</th>
      <td>2024-02-08 22:19:04.558</td>
      <td>[4.0, 1.75, 2700.0, 7875.0, 1.5, 0.0, 0.0, 4.0...</td>
      <td>[441960.38]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>997</th>
      <td>2024-02-08 22:19:04.558</td>
      <td>[4.0, 3.25, 2910.0, 1880.0, 2.0, 0.0, 3.0, 5.0...</td>
      <td>[1060847.5]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>998</th>
      <td>2024-02-08 22:19:04.558</td>
      <td>[3.0, 1.75, 2910.0, 37461.0, 1.0, 0.0, 0.0, 4....</td>
      <td>[706823.56]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>999</th>
      <td>2024-02-08 22:19:04.558</td>
      <td>[3.0, 2.0, 2005.0, 7000.0, 1.0, 0.0, 0.0, 3.0,...</td>
      <td>[581003.0]</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 4 columns</p>

### Undeploy the Pipeline

With the demonstration complete, we undeploy the pipeline and return the resources back to the cluster.

```python
sample_pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .................................... ok

<table><tr><th>name</th> <td>validation-demo</td></tr><tr><th>created</th> <td>2024-02-08 21:52:52.879885+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-08 22:18:18.613710+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>016728d7-3948-467f-8ecc-e0594f406884, 053580e2-f73d-4b63-8c9c-0b5e06be96c2, 2a8d204a-f359-4f02-b558-950dbab28dc6, 424c7f24-ca65-45af-825f-64d3e9f8e8c8, 190226c9-a536-4457-9851-a68ef968b6fc, e10707fd-75b8-4386-8466-e58dc13d2828, 53b87a42-8498-475a-bf30-73fdebbf85cc</td></tr><tr><th>steps</th> <td>anomaly-housing-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

