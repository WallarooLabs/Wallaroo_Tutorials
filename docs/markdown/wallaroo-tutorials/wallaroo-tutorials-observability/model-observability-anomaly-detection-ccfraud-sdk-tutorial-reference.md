The following tutorials are available from the [Wallaroo Tutorials Repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/20240205-data-schemas/model-observability/model-observabilty-anomaly-detection-ccfraud-sdk-tutorial).

## Wallaroo Model Observability: Anomaly Detection with CCFraud

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
  * **IMPORTANT NOTE**: Using the name `count` as a validation name **returns an error**.  Any validation rules named `count` are dropped upon request and an error returned.
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
sample_pipeline.add_model_step(ccfraud_model)

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
sample_pipeline.add_model_step(ccfraud_model)

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

* Upload a ccfraud ML model trained to detect the likelihood of a transaction being fraudulent.  This outputs the field `dense_1` as an float where the closer to 1, the higher the likelihood of the transaction being fraudulent.
* Add the ccfraud model as a pipeline step.
* Add the validation `fraud` to detect when the output of `dense_1` at index 0 when the values are greater than `0.9`.
* Deploy the pipeline and performing sample inferences on it.
* Perform sample inferences to show when the `fraud` validation returns `True` and `False`.
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

workspace_name = 'validation-ccfraud-demonstration-jch'
pipeline_name = 'ccfraud-validation-demo'
model_name = 'ccfraud'
model_file_name = './models/ccfraud.onnx'
```

```python
workspace = get_workspace(workspace_name, wl)
wl.set_current_workspace(workspace)
```

    {'name': 'validation-ccfraud-demonstration-jch', 'id': 19, 'archived': False, 'created_by': 'c97d480f-6064-4537-b18e-40fb1864b4cd', 'created_at': '2024-02-08T16:57:29.044902+00:00', 'models': [{'name': 'ccfraud', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 2, 8, 17, 12, 40, 341069, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 2, 8, 16, 57, 31, 612826, tzinfo=tzutc())}], 'pipelines': [{'name': 'ccfraud-validation', 'create_time': datetime.datetime(2024, 2, 8, 16, 57, 32, 340043, tzinfo=tzutc()), 'definition': '[]'}]}

### Upload the Model

Upload the model to the Wallaroo workspace with the `wallaroo.client.upload_model` method.  Our ccfraud ML model is a Wallaroo Default Runtime of type `ONNX`, so all we need is the model name, the model file path, and the framework type of `wallaroo.framework.Framework.ONNX`.

```python
ccfraud_model = (wl.upload_model(model_name, 
                                 model_file_name, 
                                 framework=wallaroo.framework.Framework.ONNX)
                )
```

### Build the Pipeline

Pipelines are build with the `wallaroo.client.build_pipeline` method, which takes the pipeline name.  This will create the pipeline in our default workspace.  Note that if there are any existing pipelines with the same name in this workspace, this method will retrieve that pipeline for this SDK session.

Once the pipeline is created, we add the ccfraud model as our pipeline step.

```python
sample_pipeline = wl.build_pipeline(pipeline_name)
sample_pipeline = sample_pipeline.add_model_step(ccfraud_model)

```

### Add Validation

Now we add our validation to our new pipeline.  We will give it the following configuration.

* Validation Name: `fraud`
* Validation Field: `out.dense_1`
* Validation Field Index: `0`
* Validation Expression:  Values greater than `0.9`.

The `polars` library is required for creating the validation.  We will import the polars library, then add our validation to the pipeline.

* **IMPORTANT NOTE**:  Validation names must be unique **per pipeline**.  If a validation of the same name is added, both are included in the pipeline validations, but only **most recent validation with the same name** is displayed with the inference results.  Anomalies detected by multiple validations of the same name are added to the `anomaly.count` inference result field.

```python
import polars as pl

sample_pipeline = sample_pipeline.add_validations(
    fraud=pl.col("out.dense_1").list.get(0) > 0.9
)
```

### Display Pipeline And Validation Steps

The method `wallaroo.pipeline.steps()` shows the current pipeline steps. The added validations are in the `Check` field.  This is used for demonstration purposes to show the added validation to the pipeline.

```python
sample_pipeline.steps()
```

    [{'ModelInference': {'models': [{'name': 'ccfraud', 'version': 'f1f2ab86-a41d-4601-b14a-b594e3d86c6e', 'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507'}]}},
     {'Check': {'tree': ['{"Alias":[{"BinaryExpr":{"left":{"Function":{"input":[{"Column":"out.dense_1"},{"Literal":{"Int32":0}}],"function":{"ListExpr":"Get"},"options":{"collect_groups":"ApplyFlat","fmt_str":"","input_wildcard_expansion":false,"auto_explode":true,"cast_to_supertypes":false,"allow_rename":false,"pass_name_to_apply":false,"changes_length":false,"check_lengths":true,"allow_group_aware":true}}},"op":"Gt","right":{"Literal":{"Float64":0.9}}}},"fraud"]}']}}]

### Deploy Pipeline

With the pipeline steps set and the validations created, we deploy the pipeline.  Because of it's size, we will only allocate `0.1` cpu from the cluster for the pipeline's use.

```python
deploy_config = wallaroo.deployment_config.DeploymentConfigBuilder() \
    .cpus(0.1)\
    .build()

sample_pipeline.deploy(deployment_config=deploy_config)
```

    Waiting for deployment - this will take up to 45s ......... ok

<table><tr><th>name</th> <td>ccfraud-validation-demo</td></tr><tr><th>created</th> <td>2024-02-08 17:47:02.951799+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-08 17:47:03.074392+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>5d5a2272-5c80-4eb1-9712-0a342febb775, 7baa8bc8-2218-4b09-9436-00a40407a14d</td></tr><tr><th>steps</th> <td>ccfraud</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Sample Inferences

Two sample inferences are performed with the method `wallaroo.pipeline.infer_from_file` that takes either a pandas Record JSON file or an Apache Arrow table as the input.

For our demonstration, we will use the following pandas Record JSON files with the following sample data:

* `./data/dev_smoke_test.pandas.json`: A sample inference that generates a low (lower than 0.01) likelihood of fraud.
* `./data/dev_high_fraud.pandas.json`: A sample inference that generates a high (higher than 0.90) likelihood of fraud.

The inference request returns a pandas DataFrame.

Each of the inference outputs will include the following fields:

| Field | Type | Description |
| **time** | **DateTime** | The DateTime of the inference request. |
| **in.{input_field_name}** | Input Dependent | Each input field submitted is labeled as `in.{input_field_name}` in the inference request result.  For our example, this is `tensor`, so the input field in the returned inference request is `in.tensor`. |
| **out.{model_output_field_name}** | Output Dependent | Each field output by the ML model is labeled as `out.{model_output_field_name}` in the inference request result.  For our example, the ccfraud model returns `dense_1` as its output field, so the output field in the returned inference request is `out.dense_1`. |
| **anomaly.count**	| Integer | The total number of validations that returned `True`. |
| **anomaly.{validation_name} | Bool | Each validation added to the pipeline is returned as `anomaly.{validation_name}`, and returns either `True` if the validation returns `True`, indicating an anomaly is found, or `False` for an anomaly for the validation is not found.  For our example, we will have `anomaly.fraud` returned.

```python
sample_pipeline.infer_from_file("./data/dev_smoke_test.pandas.json")
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.dense_input</th>
      <th>out.dense_1</th>
      <th>anomaly.count</th>
      <th>anomaly.fraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-08 17:47:12.909</td>
      <td>[1.0678324729, 0.2177810266, -1.7115145262, 0....</td>
      <td>[0.0014974177]</td>
      <td>0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>

```python
sample_pipeline.infer_from_file("./data/dev_high_fraud.json")
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.dense_input</th>
      <th>out.dense_1</th>
      <th>anomaly.count</th>
      <th>anomaly.fraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-08 17:47:12.962</td>
      <td>[1.0678324729, 18.1555563975, -1.6589551058, 5...</td>
      <td>[0.981199]</td>
      <td>1</td>
      <td>True</td>
    </tr>
  </tbody>
</table>

### Other Validation Examples

The following are additional examples of validations.

#### Multiple Validations

The following uses multiple validations to check for anomalies.  We still use `fraud` which detects outputs that are greater than `0.9`.  The second validation `too_low` triggers an anomaly when the `out.dense_1` is under 0.05.

After the validations are added, the pipeline is redeployed to "set" them.

```python
sample_pipeline = sample_pipeline.add_validations(
    too_low=pl.col("out.dense_1").list.get(0) < 0.001
)

deploy_config = wallaroo.deployment_config.DeploymentConfigBuilder() \
    .cpus(0.1)\
    .build()
sample_pipeline.undeploy()
sample_pipeline.deploy(deployment_config=deploy_config)
```

    Waiting for undeployment - this will take up to 45s ..................................... ok
    Waiting for deployment - this will take up to 45s ......... ok

<table><tr><th>name</th> <td>ccfraud-validation-demo</td></tr><tr><th>created</th> <td>2024-02-08 17:47:02.951799+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-08 17:47:51.243530+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>54982a2e-180b-4da6-ab50-e2940ec14ff5, 5d5a2272-5c80-4eb1-9712-0a342febb775, 7baa8bc8-2218-4b09-9436-00a40407a14d</td></tr><tr><th>steps</th> <td>ccfraud</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
sample_pipeline.infer_from_file("./data/dev_smoke_test.pandas.json")
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.dense_input</th>
      <th>out.dense_1</th>
      <th>anomaly.count</th>
      <th>anomaly.fraud</th>
      <th>anomaly.too_low</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-08 17:48:00.996</td>
      <td>[1.0678324729, 0.2177810266, -1.7115145262, 0....</td>
      <td>[0.0014974177]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>

```python
sample_pipeline.infer_from_file("./data/dev_high_fraud.json")
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.dense_input</th>
      <th>out.dense_1</th>
      <th>anomaly.count</th>
      <th>anomaly.fraud</th>
      <th>anomaly.too_low</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-08 17:48:01.041</td>
      <td>[1.0678324729, 18.1555563975, -1.6589551058, 5...</td>
      <td>[0.981199]</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>

#### Compound Validations

The following combines multiple field checks into a single validation.  For this, we will check for values of `out.dense_1` that are between 0.05 and 0.9.

Each expression is separated by `()`.  For example:

* Expression 1: `pl.col("out.dense_1").list.get(0) < 0.9`
* Expression 2: `pl.col("out.dense_1").list.get(0) > 0.001`
* Compound Expression: `(pl.col("out.dense_1").list.get(0) < 0.9) & (pl.col("out.dense_1").list.get(0) > 0.001)`

```python
sample_pipeline = sample_pipeline.add_validations(
    in_between_2=(pl.col("out.dense_1").list.get(0) < 0.9) & (pl.col("out.dense_1").list.get(0) > 0.001)
)

deploy_config = wallaroo.deployment_config.DeploymentConfigBuilder() \
    .cpus(0.1)\
    .build()
sample_pipeline.undeploy()
sample_pipeline.deploy(deployment_config=deploy_config)
```

    Waiting for undeployment - this will take up to 45s ..................................... ok
    Waiting for deployment - this will take up to 45s ......... ok

<table><tr><th>name</th> <td>ccfraud-validation-demo</td></tr><tr><th>created</th> <td>2024-02-08 17:47:02.951799+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-08 17:48:39.387256+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>97fb284e-aa10-4192-a072-308a705b969c, 54982a2e-180b-4da6-ab50-e2940ec14ff5, 5d5a2272-5c80-4eb1-9712-0a342febb775, 7baa8bc8-2218-4b09-9436-00a40407a14d</td></tr><tr><th>steps</th> <td>ccfraud</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
results = sample_pipeline.infer_from_file("./data/cc_data_1k.df.json")

results.loc[results['anomaly.in_between_2'] == True] 
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.dense_input</th>
      <th>out.dense_1</th>
      <th>anomaly.count</th>
      <th>anomaly.fraud</th>
      <th>anomaly.in_between_2</th>
      <th>anomaly.too_low</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>2024-02-08 17:48:49.305</td>
      <td>[0.5817662108, 0.097881551, 0.1546819424, 0.47...</td>
      <td>[0.0010916889]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2024-02-08 17:48:49.305</td>
      <td>[1.0379636346, -0.152987302, -1.0912561862, -0...</td>
      <td>[0.0011294782]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2024-02-08 17:48:49.305</td>
      <td>[0.1517283662, 0.6589966337, -0.3323713647, 0....</td>
      <td>[0.0018743575]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2024-02-08 17:48:49.305</td>
      <td>[-0.1683100246, 0.7070470317, 0.1875234948, -0...</td>
      <td>[0.0011520088]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2024-02-08 17:48:49.305</td>
      <td>[0.6066235674, 0.0631839305, -0.0802961973, 0....</td>
      <td>[0.0016568303]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
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
      <th>982</th>
      <td>2024-02-08 17:48:49.305</td>
      <td>[-0.0932906169, 0.2837744937, -0.061094265, 0....</td>
      <td>[0.0010192394]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>983</th>
      <td>2024-02-08 17:48:49.305</td>
      <td>[0.0991458877, 0.5813808183, -0.3863062246, -0...</td>
      <td>[0.0020678043]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>992</th>
      <td>2024-02-08 17:48:49.305</td>
      <td>[1.0458395446, 0.2492453605, -1.5260449285, 0....</td>
      <td>[0.0013128221]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>998</th>
      <td>2024-02-08 17:48:49.305</td>
      <td>[1.0046377125, 0.0343666504, -1.3512533246, 0....</td>
      <td>[0.0011070371]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>2024-02-08 17:48:49.305</td>
      <td>[0.6118805301, 0.1726081102, 0.4310545502, 0.5...</td>
      <td>[0.0012498498]</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>179 rows × 7 columns</p>

### Specify Dataset

Wallaroo inference requests allow datasets to be excluded or included with the `dataset_exclude` and `dataset` parameters.

| Parameter | Type | Description|
|---|---|---|
| **dataset_exclude** | *List(String)* | The list of datasets to exclude.  Values include:  <ul><li>`metadata`: Returns inference time per model, last model used, and other parameters.</li><li>`anomaly`: The anomaly results of all validations added to the pipeline.</li></ul> |
| **dataset** | *List(String)* | The list of datasets and fields to include. |

For our example, we will **exclude** the `anomaly` dataset, but **include** the datasets `'time'`, `'in'`, `'out'`, `'anomaly.count'`.  Note that while we exclude `anomaly`, we override that with by setting the anomaly field `'anomaly.count'` in our `dataset` parameter.

```python
sample_pipeline.infer_from_file("./data/dev_high_fraud.json", 
                                dataset_exclude=['anomaly'], 
                                dataset=['time', 'in', 'out', 'anomaly.count']
                                )
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.dense_input</th>
      <th>out.dense_1</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-08 17:48:49.634</td>
      <td>[1.0678324729, 18.1555563975, -1.6589551058, 5...</td>
      <td>[0.981199]</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

### Undeploy the Pipeline

With the demonstration complete, we undeploy the pipeline and return the resources back to the cluster.

```python
sample_pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ..................................... ok

<table><tr><th>name</th> <td>ccfraud-validation-demo</td></tr><tr><th>created</th> <td>2024-02-08 17:47:02.951799+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-08 17:48:39.387256+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>97fb284e-aa10-4192-a072-308a705b969c, 54982a2e-180b-4da6-ab50-e2940ec14ff5, 5d5a2272-5c80-4eb1-9712-0a342febb775, 7baa8bc8-2218-4b09-9436-00a40407a14d</td></tr><tr><th>steps</th> <td>ccfraud</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python

```
