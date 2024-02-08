The following tutorials are available from the [Wallaroo Tutorials Repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/20240205-data-schemas/model-observability/model-observabilty-anomaly-detection-ccfraud-sdk-tutorial).

## Wallaroo Model Observability: Anomaly Detection with CCFraud

The following tutorial demonstrates the use case of detecting anomalies:  inference input or output data that does not match typical validations.

Wallaroo provides **validations** to detect anomalous data from inference inputs and outputs.  These validations typically take the format:

```python
validation_name = polars.col(in|out.{column_name}).list.get(index) EXPRESSION
```

For example, to the following validation `fraud` detects values for the **output** of an inference request for the field `dense_1` that are greater than 0.9, indicating a transaction has a high likelihood of fraud:

```python
fraud=pl.col("out.dense_1").list.get(0) > 0.9
```

When a validation expression is **True**, then we have detected an **anomaly**.

### Validation Rule Breakdown

The validation is broken down as so:

* `validation_name`: The user provided name of the validation.  The names must match Python variable naming requirements.  * **IMPORTANT NOTE**: Using the name `count` as a validation name will return an error.  Any validation rules named `count` are dropped upon request and an error returned.
* `polars.col(in|out.{column_name})`: Specifies the **input** or **output** for a specific field in an inference request.  Wallaroo inference requests are in the format `in.{field_name}` for **inputs**, and `out.{field_name}` for **outputs**.
* `list.get(index)`: The index for the field.
* `EXPRESSION`:  The expression to test against.

For example, if the following input is submitted, where the sales figures go from `[Monday-Sunday]`:

&nbsp;|week|site_id|sales_count|
---|---|---|---|---|---|---|---
0|[27]|[site0001]|[1240, 1551, 2324, 805, 1948, 2315, 1917]

To validate that sales figures on a Saturday do **not** go below 1000 units, the validation could be:

```python
saturday_sales=pl.col("in.sales_count").list.get(6) < 1000
```

The following input would set the `saturday_sales` validation as `True`, which returns that we have detected an anomaly.

&nbsp;|week|site_id|sales_count|
---|---|---|---|---|---|---|---
0|[28]|[site0001]|[1357, 1247, 1583, 1437, 952, 757, 1831]

### Adding Validations to a Wallaroo Pipeline

Validations are added to a Wallaroo pipeline with the `wallaroo.pipeline.add_validations` method, which supports one or more validations.  For example, the following adds two validations to a Wallaroo pipeline:

* `fraud`: Detects when an inference output for the field `dense_1` at index `0` is greater than 0.9, indicating fraud.
* `to_low`: Detects when ain inference output for the field `dense_1` at index `0` is lower than 0.05, indicating some data output error may have occurred.

```python
sample_pipeline = wallaroo.client.build_pipeline("sample-pipeline")
sample_pipeline.add_model_step(ccfraud_model)
sample_pipeline.add_validations(
    fraud=pl.col("out.dense_1").list.get(0) > 0.9,
    too_low=pl.col("out.dense_1").list.get(0) < 0.05
    )
```

### Detecting Anomalies from Inference Request Results

When an inference request is submitted to a Wallaroo pipeline with validations, the following fields are output:

| Field | Type | Description |
|---|---|---|
| **anomaly.count** | **Integer** | The total of all validations that returned **True**. |
| **anomaly.{validation name}** | **Bool** | The output of the validation `{validation_name}`. |

For example, adding the validation `fraud` to the following pipeline returns `anomaly.count` of `1` when the validation `fraud` returns true, indicating an anomaly.

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

&nbsp;|time|in.tensor|out.dense_1|anomaly.count|anomaly.fraud
|---|---|---|---|---|---|
0|2024-02-02 16:05:42.152|[1.0678324729, 18.1555563975, -1.6589551058, 5...|[0.981199]|1|True

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
* [`polars` version 0.18.5]().  This is installed by default with the Wallaroo SDK.

## Tutorial Steps

### Load Libraries

The first step is to import the libraries used in this notebook.

```python
import wallaroo

```

### Connect to the Wallaroo Instance through the User Interface

The next step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
wl = wallaroo.Client()
```

    Please log into the following URL in a web browser:
    
    	https://fluffy-couch-1563.keycloak.wallaroo.dev/auth/realms/master/device?user_code=WHIV-AFPV
    
    Login successful!

### Create a New Workspace

We'll use the SDK below to create our workspace then assign as our **current workspace**.  The current workspace is used by the Wallaroo SDK for where to upload models, create pipelines, etc.  We'll also set up variables for our models and pipelines down the road, so we have one spot to change names to whatever fits your organization's standards best.

Before starting, verify that the workspace name is unique in your Wallaroo instance.

```python
workspace_name = 'validation-ccfraud-demonstration'
pipeline_name = 'ccfraud-validation'
model_name = 'ccfraud'
model_file_name = './models/ccfraud.onnx'
```

```python
workspace = wl.create_workspace(workspace_name = workspace_name)
wl.set_current_workspace(workspace)
```

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

```python
import polars as pl

sample_pipeline = sample_pipeline.add_validations(fraud=pl.col("out.dense_1").list.get(0) > 0.9)
```

### Display Pipeline And Validation Steps

The method `wallaroo.pipeline.steps()` shows the current pipeline steps. The added validations are in the `Check` field.  This is used for demonstration purposes to show the added validation to the pipeline.

```python
sample_pipeline.steps()
```

    [{'ModelInference': {'models': [{'name': 'ccfraudmodel', 'version': 'd9d96e2b-de0a-4188-9816-3d07d42038ee', 'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507'}]}},
     {'Check': {'tree': ['{"Alias":[{"BinaryExpr":{"left":{"Function":{"input":[{"Column":"out.dense_1"},{"Literal":{"Int32":0}}],"function":{"ListExpr":"Get"},"options":{"collect_groups":"ApplyFlat","fmt_str":"","input_wildcard_expansion":false,"auto_explode":true,"cast_to_supertypes":false,"allow_rename":false,"pass_name_to_apply":false,"changes_length":false,"check_lengths":true,"allow_group_aware":true}}},"op":"Gt","right":{"Literal":{"Float64":0.9}}}},"fraud"]}']}}]

### Deploy Pipeline

With the pipeline steps set and the validations created, we deploy the pipeline.  Because of it's size, we will only allocate `0.1` cpu from the cluster for the pipeline's use.

```python
deploy_config = wallaroo.deployment_config.DeploymentConfigBuilder() \
    .cpus(0.1)\
    .build()

sample_pipeline.deploy(deployment_config=deploy_config)
```

<table><tr><th>name</th> <td>ccfraudpipeline</td></tr><tr><th>created</th> <td>2024-02-01 20:02:57.850980+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-02 16:05:19.905095+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>c9654ebb-ee4c-4845-8432-3b08ac4546a1, ccb6a66d-5a33-4fe0-a9f5-23e3c7336d46, c71ebdf5-7b8a-4a95-832d-5cd21b6ed99f, d6b3a478-ed2f-4749-9094-52b2bec1184d, 099e50b5-a8c2-44d9-b625-0ef4cfdab9d6, acdf4d01-2fec-4f4b-99b0-60e52a7afda8, 1434f1dc-9d0a-4bc6-b664-8192ec3edc9b, e72648ad-dcf0-4188-b463-2a18f607b926, e05bbab7-e3ff-45b5-a2d3-54883eb3f088, da2d6c6b-d50e-4108-8b08-2fa86c770937</td></tr><tr><th>steps</th> <td>ccfraudmodel</td></tr><tr><th>published</th> <td>False</td></tr></table>

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
sample_pipeline.infer_from_file("dev_smoke_test.pandas.json")
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>anomaly.count</th>
      <th>anomaly.fraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-02 16:05:38.452</td>
      <td>[1.0678324729, 0.2177810266, -1.7115145262, 0....</td>
      <td>[0.0014974177]</td>
      <td>0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>

```python
sample_pipeline.infer_from_file("dev_high_fraud.json")
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>anomaly.count</th>
      <th>anomaly.fraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-02 16:05:42.152</td>
      <td>[1.0678324729, 18.1555563975, -1.6589551058, 5...</td>
      <td>[0.981199]</td>
      <td>1</td>
      <td>True</td>
    </tr>
  </tbody>
</table>

### Specify Dataset

Wallaroo inference requests allow datasets to be excluded or included with the `dataset_exclude` and `dataset` parameters.

| Parameter | Type | Description|
|---|---|---|
| **dataset_exclude** | *List(String)* | The list of datasets to exclude.  Values include:  <ul><li>`metadata`: Returns inference time per model, last model used, and other parameters.</li><li>`anomaly`: The anomaly results of all validations added to the pipeline.</li></ul> |
| **dataset** | *List(String)* | The list of datasets and fields to include. |

For our example, we will **exclude** the `anomaly` dataset, but **include** the datasets `'time'`, `'in'`, `'out'`, `'anomaly.count'`.  Note that while we exclude `anomaly`, we override that with by setting the anomaly field `'anomaly.count'` in our `dataset` parameter.

```python
sample_pipeline.infer_from_file("dev_high_fraud.json", 
                                dataset_exclude=['anomaly'], 
                                dataset=['time', 'in', 'out', 'anomaly.count']
                                )
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-01 23:11:03.579</td>
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

<table><tr><th>name</th> <td>ccfraudpipeline</td></tr><tr><th>created</th> <td>2024-02-01 20:02:57.850980+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-02 16:05:19.905095+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>c9654ebb-ee4c-4845-8432-3b08ac4546a1, ccb6a66d-5a33-4fe0-a9f5-23e3c7336d46, c71ebdf5-7b8a-4a95-832d-5cd21b6ed99f, d6b3a478-ed2f-4749-9094-52b2bec1184d, 099e50b5-a8c2-44d9-b625-0ef4cfdab9d6, acdf4d01-2fec-4f4b-99b0-60e52a7afda8, 1434f1dc-9d0a-4bc6-b664-8192ec3edc9b, e72648ad-dcf0-4188-b463-2a18f607b926, e05bbab7-e3ff-45b5-a2d3-54883eb3f088, da2d6c6b-d50e-4108-8b08-2fa86c770937</td></tr><tr><th>steps</th> <td>ccfraudmodel</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python

```
