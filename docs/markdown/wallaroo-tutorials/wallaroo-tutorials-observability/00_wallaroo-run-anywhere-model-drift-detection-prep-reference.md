This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/wallaroo2024.1_tutorials/wallaroo-observe-tutorials/edge-observability-assays).

## Wallaroo Run Anywhere Model Drift Observability with Assays: Preparation

The Model Insights feature lets you monitor how the environment that your model operates within may be changing in ways that affect it's predictions so that you can intervene (retrain) in an efficient and timely manner. Changes in the inputs, **data drift**, can occur due to errors in the data processing pipeline or due to changes in the environment such as user preference or behavior. 

Wallaroo Run Anywhere allows models to be deployed on edge and other locations, and have their inference result logs uploaded to the Wallaroo Ops center.  Wallaroo assays allow for model drift detection to include the inference results from one or more deployment locations and compare any one or multiple locations results against an established baseline.

This notebook is designed to demonstrate the Wallaroo Run Anywhere with Model Drift Observability with Wallaroo Assays.  This notebook will walk through the process of:

* Preparation:  This notebook focuses on setting up the conditions for model edge deployments to different locations.  This includes:
  * Setting up a workspace, pipeline, and model for deriving the price of a house based on inputs.
  * Performing a sample set of inferences to verify the model deployment.
  * Publish the deployed model to an Open Container Initiative (OCI) Registry, and use that to deploy the model to two difference edge locations.
* Model Drift by Location:
  * Perform inference requests on each of the model edge deployments.
  * Perform the steps in creating an assay:
    * Build an assay baseline with a specified location for inference results.
    * Preview the assay and show different assay configurations based on selecting the inference data from the Wallaroo Ops model deployment versus the edge deployment.
    * Create the assay.
    * View assay results.

This notebook focuses on **Preparation**.

## Goal

Model insights monitors the output of the house price model over a designated time window and compares it to an expected baseline distribution. We measure the performance of model deployments in different locations and compare that to the baseline to detect model drift.

### Resources

This tutorial provides the following:

* Models:
  * `models/rf_model.onnx`: The champion model that has been used in this environment for some time.
  * Various inputs:
    * `smallinputs.df.json`: A set of house inputs that tends to generate low house price values.
    * `biginputs.df.json`: A set of house inputs that tends to generate high house price values.

### Prerequisites

* A deployed Wallaroo instance with [Edge Registry Services](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-edge-deployment/#enable-wallaroo-edge-deployment-registry) and [Edge Observability enabled](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-edge-deployment/#set-edge-observability-service).
* The following Python libraries installed:
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame
* A X64 Docker deployment to deploy the model on an edge location.

## Steps

* Deploying a sample ML model used to determine house prices based on a set of input parameters.
* Publish the model deployment configuration to an OCI registry.
* Use the publish and set edge locations.
* Deploy the model to two different edge locations.

### Import Libraries

The first step will be to import our libraries, and set variables used through this tutorial.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework

from IPython.display import display

# used to display DataFrame information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)

import datetime
import time

workspace_name = f'run-anywhere-assay-demonstration-tutorial'
main_pipeline_name = f'assay-demonstration-tutorial'
model_name_control = f'house-price-estimator'
model_file_name_control = './models/rf_model.onnx'

# Set the name of the assay
assay_name="ops assay example"
edge_assay_name = "edge assay example"
combined_assay_name = "combined assay example"

# ignoring warnings for demonstration
import warnings
warnings.filterwarnings('ignore')

# used to display DataFrame information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)
```

```python
def get_workspace(name, client):
    workspace = None
    for ws in client.list_workspaces():
        if ws.name() == name:
            workspace= ws
    if(workspace == None):
        workspace = client.create_workspace(name)
    return workspace
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

### Create Workspace

We will create a workspace to manage our pipeline and models.  The following variables will set the name of our sample workspace then set it as the current workspace.

Workspace, pipeline, and model names should be unique to each user, so we'll add in a randomly generated suffix so multiple people can run this tutorial in a Wallaroo instance without effecting each other.

```python
workspace = get_workspace(workspace_name, wl)

wl.set_current_workspace(workspace)
```

    {'name': 'run-anywhere-assay-demonstration-tutorial', 'id': 39, 'archived': False, 'created_by': 'b4a9aa3d-83fc-407a-b4eb-37796e96f1ef', 'created_at': '2024-02-28T17:07:43.828688+00:00', 'models': [], 'pipelines': []}

### Upload The Champion Model

For our example, we will upload the champion model that has been trained to derive house prices from a variety of inputs.  The model file is `rf_model.onnx`, and is uploaded with the name `house-price-estimator`.

```python
housing_model_control = (wl.upload_model(model_name_control, 
                                        model_file_name_control, 
                                        framework=Framework.ONNX)
                                        .configure(tensor_fields=["tensor"])
                        )
```

### Build the Pipeline

This pipeline is made to be an example of an existing situation where a model is deployed and being used for inferences in a production environment.  We'll call it `assay-demonstration-tutorial`, set `housing_model_control` as a pipeline step, then run a few sample inferences.

This pipeline will be a simple one - just a single pipeline step.

```python
mainpipeline = wl.build_pipeline(main_pipeline_name)
# clear the steps if used before
mainpipeline.clear()

mainpipeline.add_model_step(housing_model_control)

#minimum deployment config
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()

mainpipeline.deploy(deployment_config = deploy_config)
```

<table><tr><th>name</th> <td>assay-demonstration-tutorial</td></tr><tr><th>created</th> <td>2024-02-28 17:09:13.517363+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-28 17:09:14.388029+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>accel</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>84cc722a-4c2a-49fa-ab85-3b9ebd0f12af, a2ec7a67-a55f-4886-84da-9f0f5d074bb4</td></tr><tr><th>steps</th> <td>house-price-estimator</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
mainpipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.124.0.133',
       'name': 'engine-5496d954fb-mdczx',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'assay-demonstration-tutorial',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'config': {'batch_config': None,
           'filter_threshold': None,
           'id': 22,
           'input_schema': None,
           'model_version_id': 15,
           'output_schema': None,
           'runtime': 'onnx',
           'sidekick_uri': None,
           'tensor_fields': ['tensor']},
          'model_version': {'conversion': {'framework': 'onnx',
            'python_version': '3.8',
            'requirements': []},
           'file_info': {'file_name': 'rf_model.onnx',
            'sha': 'e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6',
            'version': 'fa4be80e-dd8f-4e42-8385-e8eca9fd8500'},
           'id': 15,
           'image_path': None,
           'name': 'house-price-estimator',
           'status': 'ready',
           'task_id': None,
           'visibility': 'private',
           'workspace_id': 39},
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.124.0.132',
       'name': 'engine-lb-d7cc8fc9c-tsj56',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

### Testing

We'll use two inferences as a quick sample test - one that has a house that should be determined around `$700k`, the other with a house determined to be around `$1.5` million.

```python
normal_input = pd.DataFrame.from_records({"tensor": [[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]]})
result = mainpipeline.infer(normal_input)
display(result)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-28 17:09:44.877</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
large_house_input = pd.DataFrame.from_records({'tensor': [[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]]})
large_house_result = mainpipeline.infer(large_house_input)
display(large_house_result)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-28 17:09:45.199</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[1514079.4]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Undeploy Main Pipeline

With the examples and examples complete, we will undeploy the main pipeline and return the resources back to the Wallaroo instance.

```python
mainpipeline.undeploy()
```

<table><tr><th>name</th> <td>assay-demonstration-tutorial</td></tr><tr><th>created</th> <td>2024-02-28 17:09:13.517363+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-28 17:09:14.388029+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>accel</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>84cc722a-4c2a-49fa-ab85-3b9ebd0f12af, a2ec7a67-a55f-4886-84da-9f0f5d074bb4</td></tr><tr><th>steps</th> <td>house-price-estimator</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Edge Deployment

We can now deploy the pipeline to an edge device.  This will require the following steps:

* Publish the pipeline:  Publishes the pipeline to the OCI registry.
* Add Edge:  Add the edge location to the pipeline publish.
* Deploy Edge:  Deploy the edge device with the edge location settings.

### Publish Pipeline

Publishing the pipeline uses the pipeline `wallaroo.pipeline.publish()` command.  This requires that the Wallaroo Ops instance have [Edge Registry Services](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-edge-deployment/#enable-wallaroo-edge-deployment-registry) enabled.

The following publishes the pipeline to the OCI registry and displays the container details.  For more information, see [Wallaroo SDK Essentials Guide: Pipeline Edge Publication](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-publication/).

```python
assay_pub = mainpipeline.publish()
```

    Waiting for pipeline publish... It may take up to 600 sec.
    Pipeline is Publishing......Published.

### Add Edge Location

The edge location is added with the [`wallaroo.pipeline_publish.add_edge(name)`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/pipeline_publish/#PipelinePublish.add_edge) method.  This returns the OCI registration information, and the `EDGE_BUNDLE` information.  The `EDGE_BUNDLE` data is a base64 encoded set of parameters for the pipeline that the edge device is associated with, the workspace, and other data.

For full details, see [Wallaroo SDK Essentials Guide: Pipeline Edge Publication: Edge Observability](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-publication/#edge-observability).

For this example, we will add two locations:

* `houseprice-edge-demonstration-01`
* `houseprice-edge-demostration-02`

These will be used in later steps for demonstrating inferences through different locations.

```python
edge_name_01 = "houseprice-edge-demonstration-01"
edge_publish_01 = assay_pub.add_edge(edge_name_01)
display(edge_publish_01)
```

          <table>
              <tr><td>ID</td><td>11</td></tr>
              <tr><td>Pipeline Name</td><td>assay-demonstration-tutorial</td></tr>
              <tr><td>Pipeline Version</td><td>fa48cea1-77ba-4035-b2d3-65a76b6aa10b</td></tr>
              <tr><td>Status</td><td>Published</td></tr>
              <tr><td>Engine URL</td><td><a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.1.0-main-4609'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.1.0-main-4609</a></td></tr>
              <tr><td>Pipeline URL</td><td><a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/assay-demonstration-tutorial:fa48cea1-77ba-4035-b2d3-65a76b6aa10b'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/assay-demonstration-tutorial:fa48cea1-77ba-4035-b2d3-65a76b6aa10b</a></td></tr>
              <tr><td>Helm Chart URL</td><td>oci://<a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts/assay-demonstration-tutorial'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts/assay-demonstration-tutorial</a></td></tr>
              <tr><td>Helm Chart Reference</td><td>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts@sha256:09ae7e53d9bda5168cdb929fb7e6c024f0c97db94c58b6b660209226347b0e43</td></tr>
              <tr><td>Helm Chart Version</td><td>0.0.1-fa48cea1-77ba-4035-b2d3-65a76b6aa10b</td></tr>
              <tr><td>Engine Config</td><td>{'engine': {'resources': {'limits': {'cpu': 4.0, 'memory': '3Gi'}, 'requests': {'cpu': 4.0, 'memory': '3Gi'}, 'accel': 'none', 'arch': 'x86', 'gpu': False}}, 'engineAux': {'autoscale': {'type': 'none'}, 'images': None}, 'enginelb': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 0.2, 'memory': '512Mi'}, 'accel': 'none', 'arch': 'x86', 'gpu': False}}}</td></tr>
              <tr><td>User Images</td><td>[]</td></tr>
              <tr><td>Created By</td><td>john.hummel@wallaroo.ai</td></tr>
              <tr><td>Created At</td><td>2024-02-28 17:10:25.499966+00:00</td></tr>
              <tr><td>Updated At</td><td>2024-02-28 17:10:25.499966+00:00</td></tr>
              <tr><td>Replaces</td><td></td></tr>
              <tr>
                  <td>Docker Run Command</td>
                  <td>
                      <table><tr><td>
<pre style="text-align: left">docker run \
    -e OCI_USERNAME=$OCI_USERNAME \
    -e OCI_PASSWORD=$OCI_PASSWORD \
    -e EDGE_BUNDLE=ZXhwb3J0IEJVTkRMRV9WRVJTSU9OPTEKZXhwb3J0IENPTkZJR19DUFVTPTQKZXhwb3J0IEVER0VfTkFNRT1ob3VzZXByaWNlLWVkZ2UtZGVtby0wMQpleHBvcnQgT1BTQ0VOVEVSX0hPU1Q9ZWRnZS5hdXRvc2NhbGUtdWF0LWdjcC53YWxsYXJvby5kZXYKZXhwb3J0IFBJUEVMSU5FX1VSTD11cy1jZW50cmFsMS1kb2NrZXIucGtnLmRldi93YWxsYXJvby1kZXYtMjUzODE2L3VhdC9waXBlbGluZXMvYXNzYXktZGVtb25zdHJhdGlvbi10dXRvcmlhbDpmYTQ4Y2VhMS03N2JhLTQwMzUtYjJkMy02NWE3NmI2YWExMGIKZXhwb3J0IEpPSU5fVE9LRU49NzgwMGJlZDgtZTYxYy00NWFiLTk5MTUtZDU1MmZmOWI0MjFi \
    -e CONFIG_CPUS=4 us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.1.0-main-4609</pre></td></tr></table>
                      <br />
                      <i>
                          Note: Please set the <code>OCI_USERNAME</code>, and <code>OCI_PASSWORD</code> environment variables.
                      </i>
                  </td>
              </tr>
              <tr>
                  <td>Helm Install Command</td>
                  <td>
                      <table><tr><td>
<pre style="text-align: left">helm install --atomic $HELM_INSTALL_NAME \
    oci://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts/assay-demonstration-tutorial \
    --namespace $HELM_INSTALL_NAMESPACE \
    --version 0.0.1-fa48cea1-77ba-4035-b2d3-65a76b6aa10b \
    --set ociRegistry.username=$OCI_USERNAME \
    --set ociRegistry.password=$OCI_PASSWORD \
    --set edgeBundle=ZXhwb3J0IEJVTkRMRV9WRVJTSU9OPTEKZXhwb3J0IENPTkZJR19DUFVTPTQKZXhwb3J0IEVER0VfTkFNRT1ob3VzZXByaWNlLWVkZ2UtZGVtby0wMQpleHBvcnQgT1BTQ0VOVEVSX0hPU1Q9ZWRnZS5hdXRvc2NhbGUtdWF0LWdjcC53YWxsYXJvby5kZXYKZXhwb3J0IFBJUEVMSU5FX1VSTD11cy1jZW50cmFsMS1kb2NrZXIucGtnLmRldi93YWxsYXJvby1kZXYtMjUzODE2L3VhdC9waXBlbGluZXMvYXNzYXktZGVtb25zdHJhdGlvbi10dXRvcmlhbDpmYTQ4Y2VhMS03N2JhLTQwMzUtYjJkMy02NWE3NmI2YWExMGIKZXhwb3J0IEpPSU5fVE9LRU49NzgwMGJlZDgtZTYxYy00NWFiLTk5MTUtZDU1MmZmOWI0MjFi</pre></td></tr></table>
                      <br />
                      <i>
                          Note: Please set the <code>HELM_INSTALL_NAME</code>, <code>HELM_INSTALL_NAMESPACE</code>,
                          <code>OCI_USERNAME</code>, and <code>OCI_PASSWORD</code> environment variables.
                      </i>
                  </td>
              </tr>

          </table>

```python
edge_name_02 = "houseprice-edge-demonstration-02"
edge_publish_02 = assay_pub.add_edge(edge_name_02)
display(edge_publish_02)
```

          <table>
              <tr><td>ID</td><td>11</td></tr>
              <tr><td>Pipeline Name</td><td>assay-demonstration-tutorial</td></tr>
              <tr><td>Pipeline Version</td><td>fa48cea1-77ba-4035-b2d3-65a76b6aa10b</td></tr>
              <tr><td>Status</td><td>Published</td></tr>
              <tr><td>Engine URL</td><td><a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.1.0-main-4609'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.1.0-main-4609</a></td></tr>
              <tr><td>Pipeline URL</td><td><a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/assay-demonstration-tutorial:fa48cea1-77ba-4035-b2d3-65a76b6aa10b'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/assay-demonstration-tutorial:fa48cea1-77ba-4035-b2d3-65a76b6aa10b</a></td></tr>
              <tr><td>Helm Chart URL</td><td>oci://<a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts/assay-demonstration-tutorial'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts/assay-demonstration-tutorial</a></td></tr>
              <tr><td>Helm Chart Reference</td><td>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts@sha256:09ae7e53d9bda5168cdb929fb7e6c024f0c97db94c58b6b660209226347b0e43</td></tr>
              <tr><td>Helm Chart Version</td><td>0.0.1-fa48cea1-77ba-4035-b2d3-65a76b6aa10b</td></tr>
              <tr><td>Engine Config</td><td>{'engine': {'resources': {'limits': {'cpu': 4.0, 'memory': '3Gi'}, 'requests': {'cpu': 4.0, 'memory': '3Gi'}, 'accel': 'none', 'arch': 'x86', 'gpu': False}}, 'engineAux': {'autoscale': {'type': 'none'}, 'images': None}, 'enginelb': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 0.2, 'memory': '512Mi'}, 'accel': 'none', 'arch': 'x86', 'gpu': False}}}</td></tr>
              <tr><td>User Images</td><td>[]</td></tr>
              <tr><td>Created By</td><td>john.hummel@wallaroo.ai</td></tr>
              <tr><td>Created At</td><td>2024-02-28 17:10:25.499966+00:00</td></tr>
              <tr><td>Updated At</td><td>2024-02-28 17:10:25.499966+00:00</td></tr>
              <tr><td>Replaces</td><td></td></tr>
              <tr>
                  <td>Docker Run Command</td>
                  <td>
                      <table><tr><td>
<pre style="text-align: left">docker run \
    -e OCI_USERNAME=$OCI_USERNAME \
    -e OCI_PASSWORD=$OCI_PASSWORD \
    -e EDGE_BUNDLE=ZXhwb3J0IEJVTkRMRV9WRVJTSU9OPTEKZXhwb3J0IENPTkZJR19DUFVTPTQKZXhwb3J0IEVER0VfTkFNRT1ob3VzZXByaWNlLWVkZ2UtZGVtby0wMgpleHBvcnQgT1BTQ0VOVEVSX0hPU1Q9ZWRnZS5hdXRvc2NhbGUtdWF0LWdjcC53YWxsYXJvby5kZXYKZXhwb3J0IFBJUEVMSU5FX1VSTD11cy1jZW50cmFsMS1kb2NrZXIucGtnLmRldi93YWxsYXJvby1kZXYtMjUzODE2L3VhdC9waXBlbGluZXMvYXNzYXktZGVtb25zdHJhdGlvbi10dXRvcmlhbDpmYTQ4Y2VhMS03N2JhLTQwMzUtYjJkMy02NWE3NmI2YWExMGIKZXhwb3J0IEpPSU5fVE9LRU49YjJhZDM5NGMtMjQ4Yi00MjhiLTg1NDItNzJlMjIxMTcyOWU3 \
    -e CONFIG_CPUS=4 us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini:v2024.1.0-main-4609</pre></td></tr></table>
                      <br />
                      <i>
                          Note: Please set the <code>OCI_USERNAME</code>, and <code>OCI_PASSWORD</code> environment variables.
                      </i>
                  </td>
              </tr>
              <tr>
                  <td>Helm Install Command</td>
                  <td>
                      <table><tr><td>
<pre style="text-align: left">helm install --atomic $HELM_INSTALL_NAME \
    oci://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts/assay-demonstration-tutorial \
    --namespace $HELM_INSTALL_NAMESPACE \
    --version 0.0.1-fa48cea1-77ba-4035-b2d3-65a76b6aa10b \
    --set ociRegistry.username=$OCI_USERNAME \
    --set ociRegistry.password=$OCI_PASSWORD \
    --set edgeBundle=ZXhwb3J0IEJVTkRMRV9WRVJTSU9OPTEKZXhwb3J0IENPTkZJR19DUFVTPTQKZXhwb3J0IEVER0VfTkFNRT1ob3VzZXByaWNlLWVkZ2UtZGVtby0wMgpleHBvcnQgT1BTQ0VOVEVSX0hPU1Q9ZWRnZS5hdXRvc2NhbGUtdWF0LWdjcC53YWxsYXJvby5kZXYKZXhwb3J0IFBJUEVMSU5FX1VSTD11cy1jZW50cmFsMS1kb2NrZXIucGtnLmRldi93YWxsYXJvby1kZXYtMjUzODE2L3VhdC9waXBlbGluZXMvYXNzYXktZGVtb25zdHJhdGlvbi10dXRvcmlhbDpmYTQ4Y2VhMS03N2JhLTQwMzUtYjJkMy02NWE3NmI2YWExMGIKZXhwb3J0IEpPSU5fVE9LRU49YjJhZDM5NGMtMjQ4Yi00MjhiLTg1NDItNzJlMjIxMTcyOWU3</pre></td></tr></table>
                      <br />
                      <i>
                          Note: Please set the <code>HELM_INSTALL_NAME</code>, <code>HELM_INSTALL_NAMESPACE</code>,
                          <code>OCI_USERNAME</code>, and <code>OCI_PASSWORD</code> environment variables.
                      </i>
                  </td>
              </tr>

          </table>

## Next Steps

The next notebook "Wallaroo Run Anywhere Model Drift Observability with Assays" details creating assay baselines from model edge deployments, and using the data from one or more edge locations to detect model drift.
