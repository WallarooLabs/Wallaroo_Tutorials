This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/wallaroo-tag-management).

## Wallaroo SDK Tag Tutorial

The following tutorial demonstrates how to use Wallaroo Tags.  Tags are applied to either model versions or pipelines.  This allows organizations to track different versions of models, and search for what pipelines have been used for specific purposes such as testing versus production use.

The following will be demonstrated:

* List all tags in a Wallaroo instance.
* List all tags applied to a model.
* List all tags applied to a pipeline.
* Apply a tag to a model.
* Remove a tag from a model.
* Apply a tag to a pipeline.
* Remove a tag from a pipeline.
* Search for a model version by a tag.
* Search for a pipeline by a tag.

This demonstration provides the following through the Wallaroo Tutorials Github Repository:

* `models/ccfraud.onnx`: a sample model used as part of the [Wallaroo 101 Tutorials](https://docs.wallaroo.ai/wallaroo-101/).

## Prerequisites

* A deployed Wallaroo instance
* The following Python libraries installed:
  * `os`
  * `string`
  * `random`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.

## Steps

The following steps are performed use to connect to a Wallaroo instance and demonstrate how to use tags with models and pipelines.

### Load Libraries

The first step is to load the libraries used to connect and use a Wallaroo instance.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd

# used to display dataframe information without truncating
from IPython.display import display
pd.set_option('display.max_colwidth', None)
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Client connection from local Wallaroo instance

wl = wallaroo.Client()
```

### Set Variables

The following variables are used to create or connect to existing workspace and pipeline.  The model name and model file are set as well.  Adjust as required for your organization's needs.

The methods `get_workspace` and `get_pipeline` are used to either create a new workspace and pipeline based on the variables below, or connect to an existing workspace and pipeline with the same name.  Once complete, the workspace will be set as the current workspace where pipelines and models are used.

To allow this tutorial to be run multiple times or by multiple users in the same Wallaroo instance, a random 4 character prefix will be added to the workspace, pipeline, and model.

```python
import string
import random

# make a random 4 character prefix
prefix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

workspace_name = f'{prefix}tagtestworkspace'
pipeline_name = f'{prefix}tagtestpipeline'
model_name = f'{prefix}tagtestmodel'
model_file_name = './models/ccfraud.onnx'
```

```python
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
        pipeline = wl.pipelines_by_name(name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(name)
    return pipeline
```

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
```

    {'name': 'rehqtagtestworkspace', 'id': 24, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-17T21:56:18.63721+00:00', 'models': [], 'pipelines': []}

### Upload Model and Create Pipeline

The `tagtest_model` and `tagtest_pipeline` will be created (or connected if already existing) based on the variables set earlier.

```python
tagtest_model = wl.upload_model(model_name, model_file_name, framework=wallaroo.framework.Framework.ONNX).configure()
tagtest_model
```

    {'name': 'rehqtagtestmodel', 'version': '53febe9a-bb4b-4a01-a6a2-a17f943d6652', 'file_name': 'ccfraud.onnx', 'image_path': None, 'last_update_time': datetime.datetime(2023, 5, 17, 21, 56, 20, 208454, tzinfo=tzutc())}

```python
tagtest_pipeline = get_pipeline(pipeline_name)
tagtest_pipeline
```

<table><tr><th>name</th> <td>rehqtagtestpipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:56:21.405556+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:56:21.405556+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>e259f6db-8ce2-45f1-b2d7-a719fde3b18f</td></tr><tr><th>steps</th> <td></td></tr></table>

### List Pipeline and Model Tags

This tutorial assumes that no tags are currently existing, but that can be verified through the Wallaroo client `list_pipelines` and `list_models` commands.  For this demonstration, it is recommended to use unique tags to verify each example.

```python
wl.list_pipelines()
```

<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>tags</th><th>versions</th><th>steps</th></tr><tr><td>rehqtagtestpipeline</td><td>2023-17-May 21:56:21</td><td>2023-17-May 21:56:21</td><td>(unknown)</td><td></td><td>e259f6db-8ce2-45f1-b2d7-a719fde3b18f</td><td></td></tr><tr><td>osysapiinferenceexamplepipeline</td><td>2023-17-May 21:54:56</td><td>2023-17-May 21:54:56</td><td>False</td><td></td><td>8f244f23-73f9-4af2-a95e-2a03214dca63</td><td>osysccfraud</td></tr><tr><td>fvqusdkinferenceexamplepipeline</td><td>2023-17-May 21:53:14</td><td>2023-17-May 21:53:15</td><td>False</td><td></td><td>a987e13f-ffbe-4826-a6f5-9fd8de9f47fa, 0966d243-ce76-4132-aa69-0d287ae9a572</td><td>fvquccfraud</td></tr><tr><td>gobtedgepipelineexample</td><td>2023-17-May 21:50:13</td><td>2023-17-May 21:51:06</td><td>False</td><td></td><td>dc0238e7-f3e3-4579-9a63-24902cb3e3bd, 5cf788a6-50ff-471f-a3ee-4bfdc24def34, 9efda57b-c18b-4ebb-9681-33647e7d7e66</td><td>gobtalohamodel</td></tr><tr><td>logpipeline</td><td>2023-17-May 21:41:06</td><td>2023-17-May 21:46:51</td><td>False</td><td></td><td>66fb765b-d46c-4472-9976-dba2eac5b8ce, 328b2b59-7a57-403b-abd5-70708a67674e, 18eb212d-0af5-4c0b-8bdb-3abbc4907a3e, c39b5215-0535-4006-a26a-d78b1866435b</td><td>logcontrol</td></tr><tr><td>btffhotswappipeline</td><td>2023-17-May 21:37:16</td><td>2023-17-May 21:37:39</td><td>False</td><td></td><td>438796a3-e320-4a51-9e64-35eb32d57b49, 4fc11650-1003-43c2-bd3a-96b9cdacbb6d, e4b8d7ca-00fa-4e31-8671-3d0a3bf4c16e, 3c5f951b-e815-4bc7-93bf-84de3d46718d</td><td>btffhousingmodelcontrol</td></tr><tr><td>qjjoccfraudpipeline</td><td>2023-17-May 21:32:06</td><td>2023-17-May 21:32:08</td><td>False</td><td></td><td>89b634d6-f538-4ac6-98a2-fbb9883fdeb6, c0f8551d-cefe-49c8-8701-c2a307c0ad99</td><td>qjjoccfraudmodel</td></tr><tr><td>housing-pipe</td><td>2023-17-May 21:26:56</td><td>2023-17-May 21:29:05</td><td>False</td><td></td><td>34e75a0c-01bd-4ca2-a6e8-ebdd25473aab, b7dbd380-e48c-487c-8f23-398a2ba558c3, 5ea6f182-5764-4377-9f83-d363e349ef32</td><td>preprocess</td></tr><tr><td>xgboost-regression-autoconvert-pipeline</td><td>2023-17-May 21:21:56</td><td>2023-17-May 21:21:59</td><td>False</td><td></td><td>f5337089-2756-469a-871a-1cb9e3416847, 324433ae-db9a-4d43-9563-ff76df59953d</td><td>xgb-regression-model</td></tr><tr><td>xgboost-classification-autoconvert-pipeline</td><td>2023-17-May 21:21:19</td><td>2023-17-May 21:21:22</td><td>False</td><td></td><td>5f7bb0cc-f60d-4cee-8425-c5e85331ae2f, bbe4dce4-f62a-4f4f-a45c-aebbfce23304</td><td>xgb-class-model</td></tr><tr><td>statsmodelpipeline</td><td>2023-17-May 21:19:52</td><td>2023-17-May 21:19:55</td><td>False</td><td></td><td>4af264e3-f427-4b02-b5ad-4f6690b0ee06, 5456dd2a-3167-4b3c-ad3a-85544292a230</td><td>bikedaymodel</td></tr><tr><td>isoletpipeline</td><td>2023-17-May 21:17:33</td><td>2023-17-May 21:17:44</td><td>False</td><td></td><td>c129b33c-cefc-4873-ad2c-d186fe2b8228, 145b768e-79f2-44fd-ab6b-14d675501b83</td><td>isolettest</td></tr><tr><td>externalkerasautoconvertpipeline</td><td>2023-17-May 21:13:27</td><td>2023-17-May 21:13:30</td><td>False</td><td></td><td>7be0dd01-ef82-4335-b60d-6f1cd5287e5b, 3948e0dc-d591-4ff5-a48f-b8d17195a806</td><td>externalsimple-sentiment-model</td></tr><tr><td>gcpsdkpipeline</td><td>2023-17-May 21:03:44</td><td>2023-17-May 21:03:49</td><td>False</td><td></td><td>6398cafc-50c4-49e3-9499-6025b7808245, 7c043d3c-c894-4ae9-9ec1-c35518130b90</td><td>gcpsdkmodel</td></tr><tr><td>databricksazuresdkpipeline</td><td>2023-17-May 21:02:55</td><td>2023-17-May 21:02:59</td><td>False</td><td></td><td>f125dc67-f690-4011-986a-8f6a9a23c48a, 8c4a15b4-2ef0-4da1-8e2d-38088fde8c56</td><td>ccfraudmodel</td></tr><tr><td>azuremlsdkpipeline</td><td>2023-17-May 21:01:46</td><td>2023-17-May 21:01:51</td><td>False</td><td></td><td>28a7a5aa-5359-4320-842b-bad84258f7e4, e011272d-c22c-4b2d-ab9f-b17c60099434</td><td>azuremlsdkmodel</td></tr><tr><td>copiedmodelpipeline</td><td>2023-17-May 20:54:01</td><td>2023-17-May 20:54:01</td><td>(unknown)</td><td></td><td>bcf5994f-1729-4036-a910-00b662946801</td><td></td></tr><tr><td>pipelinemodels</td><td>2023-17-May 20:52:06</td><td>2023-17-May 20:52:06</td><td>False</td><td></td><td>55f45c16-591e-4a16-8082-3ab6d843b484</td><td>apimodel</td></tr><tr><td>pipelinenomodel</td><td>2023-17-May 20:52:04</td><td>2023-17-May 20:52:04</td><td>(unknown)</td><td></td><td>a6dd2cee-58d6-4d24-9e25-f531dbbb95ad</td><td></td></tr><tr><td>sdkquickpipeline</td><td>2023-17-May 20:43:38</td><td>2023-17-May 20:46:02</td><td>False</td><td></td><td>961c909d-f5ae-472a-b8ae-1e6a00fbc36e, bf7c2146-ed14-430b-bf96-1e8b1047eb2e, 2bd5c838-f7cc-4f48-91ea-28a9ce0f7ed8, d72c468a-a0e2-4189-aa7a-4e27127a2f2b</td><td>sdkquickmodel</td></tr><tr><td>housepricepipe</td><td>2023-17-May 20:41:50</td><td>2023-17-May 20:41:50</td><td>False</td><td></td><td>4d9dfb3b-c9ae-402a-96fc-20ae0a2b2279, fc68f5f2-7bbf-435e-b434-e0c89c28c6a9</td><td>housepricemodel</td></tr></table>

```python
wl.list_models()
```

<table>
  <tr>
    <th>Name</th>
    <th># of Versions</th>
    <th>Owner ID</th>
    <th>Last Updated</th>
    <th>Created At</th>
  </tr>

  <tr>
    <td>rehqtagtestmodel</td>
    <td>1</td>
    <td>""</td>
    <td>2023-05-17 21:56:20.208454+00:00</td>
    <td>2023-05-17 21:56:20.208454+00:00</td>
  </tr>

</table>

### Create Tag

Tags are created with the Wallaroo client command `create_tag(String tagname)`.  This creates the tag and makes it available for use.

The tag will be saved to the variable `currentTag` to be used in the rest of these examples.

```python
# Now we create our tag
currentTag = wl.create_tag("My Great Tag")
```

### List Tags

Tags are listed with the Wallaroo client command `list_tags()`, which shows all tags and what models and pipelines they have been assigned to.  Note that if a tag has not been assigned, it will not be displayed.

```python
# List all tags

wl.list_tags()
```

(no tags)

### Assign Tag to a Model

Tags are assigned to a model through the Wallaroo Tag `add_to_model(model_id)` command, where `model_id` is the model's numerical ID number.  The tag is applied to the most current version of the model.

For this example, the `currentTag` will be applied to the `tagtest_model`.  All tags will then be listed to show it has been assigned to this model.

```python
# add tag to model

currentTag.add_to_model(tagtest_model.id())
```

    {'model_id': 29, 'tag_id': 1}

```python
# list all tags to verify

wl.list_tags()
```

<table><tr><th>id</th><th>tag</th><th>models</th><th>pipelines</th></tr><tr><td>1</td><td>My Great Tag</td><td>[('rehqtagtestmodel', ['53febe9a-bb4b-4a01-a6a2-a17f943d6652'])]</td><td>[]</td></tr></table>

### Search Models by Tag

Model versions can be searched via tags using the Wallaroo Client method `search_models(search_term)`, where `search_term` is a string value.  All models versions containing the tag will be displayed.  In this example, we will be using the text from our tag to list all models that have the text from `currentTag` in them.

```python
# Search models by tag

wl.search_models('My Great Tag')
```

<table><tr><th>name</th><th>version</th><th>file_name</th><th>image_path</th><th>last_update_time</th></tr>
            <tr>
                <td>rehqtagtestmodel</td>
                <td>53febe9a-bb4b-4a01-a6a2-a17f943d6652</td>
                <td>ccfraud.onnx</td>
                <td>None</td>
                <td>2023-05-17 21:56:20.208454+00:00</td>
            </tr>
          </table>

### Remove Tag from Model

Tags are removed from models using the Wallaroo Tag `remove_from_model(model_id)` command.

In this example, the `currentTag` will be removed from `tagtest_model`.  A list of all tags will be shown with the `list_tags` command, followed by searching the models for the tag to verify it has been removed.

```python
### remove tag from model

currentTag.remove_from_model(tagtest_model.id())
```

    {'model_id': 29, 'tag_id': 1}

```python
# list all tags to verify it has been removed from `tagtest_model`.

wl.list_tags()
```

(no tags)

```python
# search models for currentTag to verify it has been removed from `tagtest_model`.

wl.search_models('My Great Tag')
```

(no model versions)

### Add Tag to Pipeline

Tags are added to a pipeline through the Wallaroo Tag `add_to_pipeline(pipeline_id)` method, where `pipeline_id` is the pipeline's integer id.

For this example, we will add `currentTag` to `testtest_pipeline`, then verify it has been added through the `list_tags` command and `list_pipelines` command.

```python
# add this tag to the pipeline
currentTag.add_to_pipeline(tagtest_pipeline.id())
```

    {'pipeline_pk_id': 45, 'tag_pk_id': 1}

```python
# list tags to verify it was added to tagtest_pipeline

wl.list_tags()

```

<table><tr><th>id</th><th>tag</th><th>models</th><th>pipelines</th></tr><tr><td>1</td><td>My Great Tag</td><td>[]</td><td>[('rehqtagtestpipeline', ['e259f6db-8ce2-45f1-b2d7-a719fde3b18f'])]</td></tr></table>

```python
# get all of the pipelines to show the tag was added to tagtest-pipeline

wl.list_pipelines()
```

<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>tags</th><th>versions</th><th>steps</th></tr><tr><td>rehqtagtestpipeline</td><td>2023-17-May 21:56:21</td><td>2023-17-May 21:56:21</td><td>(unknown)</td><td>My Great Tag</td><td>e259f6db-8ce2-45f1-b2d7-a719fde3b18f</td><td></td></tr><tr><td>osysapiinferenceexamplepipeline</td><td>2023-17-May 21:54:56</td><td>2023-17-May 21:54:56</td><td>False</td><td></td><td>8f244f23-73f9-4af2-a95e-2a03214dca63</td><td>osysccfraud</td></tr><tr><td>fvqusdkinferenceexamplepipeline</td><td>2023-17-May 21:53:14</td><td>2023-17-May 21:53:15</td><td>False</td><td></td><td>a987e13f-ffbe-4826-a6f5-9fd8de9f47fa, 0966d243-ce76-4132-aa69-0d287ae9a572</td><td>fvquccfraud</td></tr><tr><td>gobtedgepipelineexample</td><td>2023-17-May 21:50:13</td><td>2023-17-May 21:51:06</td><td>False</td><td></td><td>dc0238e7-f3e3-4579-9a63-24902cb3e3bd, 5cf788a6-50ff-471f-a3ee-4bfdc24def34, 9efda57b-c18b-4ebb-9681-33647e7d7e66</td><td>gobtalohamodel</td></tr><tr><td>logpipeline</td><td>2023-17-May 21:41:06</td><td>2023-17-May 21:46:51</td><td>False</td><td></td><td>66fb765b-d46c-4472-9976-dba2eac5b8ce, 328b2b59-7a57-403b-abd5-70708a67674e, 18eb212d-0af5-4c0b-8bdb-3abbc4907a3e, c39b5215-0535-4006-a26a-d78b1866435b</td><td>logcontrol</td></tr><tr><td>btffhotswappipeline</td><td>2023-17-May 21:37:16</td><td>2023-17-May 21:37:39</td><td>False</td><td></td><td>438796a3-e320-4a51-9e64-35eb32d57b49, 4fc11650-1003-43c2-bd3a-96b9cdacbb6d, e4b8d7ca-00fa-4e31-8671-3d0a3bf4c16e, 3c5f951b-e815-4bc7-93bf-84de3d46718d</td><td>btffhousingmodelcontrol</td></tr><tr><td>qjjoccfraudpipeline</td><td>2023-17-May 21:32:06</td><td>2023-17-May 21:32:08</td><td>False</td><td></td><td>89b634d6-f538-4ac6-98a2-fbb9883fdeb6, c0f8551d-cefe-49c8-8701-c2a307c0ad99</td><td>qjjoccfraudmodel</td></tr><tr><td>housing-pipe</td><td>2023-17-May 21:26:56</td><td>2023-17-May 21:29:05</td><td>False</td><td></td><td>34e75a0c-01bd-4ca2-a6e8-ebdd25473aab, b7dbd380-e48c-487c-8f23-398a2ba558c3, 5ea6f182-5764-4377-9f83-d363e349ef32</td><td>preprocess</td></tr><tr><td>xgboost-regression-autoconvert-pipeline</td><td>2023-17-May 21:21:56</td><td>2023-17-May 21:21:59</td><td>False</td><td></td><td>f5337089-2756-469a-871a-1cb9e3416847, 324433ae-db9a-4d43-9563-ff76df59953d</td><td>xgb-regression-model</td></tr><tr><td>xgboost-classification-autoconvert-pipeline</td><td>2023-17-May 21:21:19</td><td>2023-17-May 21:21:22</td><td>False</td><td></td><td>5f7bb0cc-f60d-4cee-8425-c5e85331ae2f, bbe4dce4-f62a-4f4f-a45c-aebbfce23304</td><td>xgb-class-model</td></tr><tr><td>statsmodelpipeline</td><td>2023-17-May 21:19:52</td><td>2023-17-May 21:19:55</td><td>False</td><td></td><td>4af264e3-f427-4b02-b5ad-4f6690b0ee06, 5456dd2a-3167-4b3c-ad3a-85544292a230</td><td>bikedaymodel</td></tr><tr><td>isoletpipeline</td><td>2023-17-May 21:17:33</td><td>2023-17-May 21:17:44</td><td>False</td><td></td><td>c129b33c-cefc-4873-ad2c-d186fe2b8228, 145b768e-79f2-44fd-ab6b-14d675501b83</td><td>isolettest</td></tr><tr><td>externalkerasautoconvertpipeline</td><td>2023-17-May 21:13:27</td><td>2023-17-May 21:13:30</td><td>False</td><td></td><td>7be0dd01-ef82-4335-b60d-6f1cd5287e5b, 3948e0dc-d591-4ff5-a48f-b8d17195a806</td><td>externalsimple-sentiment-model</td></tr><tr><td>gcpsdkpipeline</td><td>2023-17-May 21:03:44</td><td>2023-17-May 21:03:49</td><td>False</td><td></td><td>6398cafc-50c4-49e3-9499-6025b7808245, 7c043d3c-c894-4ae9-9ec1-c35518130b90</td><td>gcpsdkmodel</td></tr><tr><td>databricksazuresdkpipeline</td><td>2023-17-May 21:02:55</td><td>2023-17-May 21:02:59</td><td>False</td><td></td><td>f125dc67-f690-4011-986a-8f6a9a23c48a, 8c4a15b4-2ef0-4da1-8e2d-38088fde8c56</td><td>ccfraudmodel</td></tr><tr><td>azuremlsdkpipeline</td><td>2023-17-May 21:01:46</td><td>2023-17-May 21:01:51</td><td>False</td><td></td><td>28a7a5aa-5359-4320-842b-bad84258f7e4, e011272d-c22c-4b2d-ab9f-b17c60099434</td><td>azuremlsdkmodel</td></tr><tr><td>copiedmodelpipeline</td><td>2023-17-May 20:54:01</td><td>2023-17-May 20:54:01</td><td>(unknown)</td><td></td><td>bcf5994f-1729-4036-a910-00b662946801</td><td></td></tr><tr><td>pipelinemodels</td><td>2023-17-May 20:52:06</td><td>2023-17-May 20:52:06</td><td>False</td><td></td><td>55f45c16-591e-4a16-8082-3ab6d843b484</td><td>apimodel</td></tr><tr><td>pipelinenomodel</td><td>2023-17-May 20:52:04</td><td>2023-17-May 20:52:04</td><td>(unknown)</td><td></td><td>a6dd2cee-58d6-4d24-9e25-f531dbbb95ad</td><td></td></tr><tr><td>sdkquickpipeline</td><td>2023-17-May 20:43:38</td><td>2023-17-May 20:46:02</td><td>False</td><td></td><td>961c909d-f5ae-472a-b8ae-1e6a00fbc36e, bf7c2146-ed14-430b-bf96-1e8b1047eb2e, 2bd5c838-f7cc-4f48-91ea-28a9ce0f7ed8, d72c468a-a0e2-4189-aa7a-4e27127a2f2b</td><td>sdkquickmodel</td></tr><tr><td>housepricepipe</td><td>2023-17-May 20:41:50</td><td>2023-17-May 20:41:50</td><td>False</td><td></td><td>4d9dfb3b-c9ae-402a-96fc-20ae0a2b2279, fc68f5f2-7bbf-435e-b434-e0c89c28c6a9</td><td>housepricemodel</td></tr></table>

### Search Pipelines by Tag

Pipelines can be searched through the Wallaroo Client `search_pipelines(search_term)` method, where `search_term` is a string value for tags assigned to the pipelines.

In this example, the text "My Great Tag" that corresponds to `currentTag` will be searched for and displayed.

```python
wl.search_pipelines('My Great Tag')
```

<table><tr><th>name</th><th>version</th><th>creation_time</th><th>last_updated_time</th><th>deployed</th><th>tags</th><th>steps</th></tr><tr><td>rehqtagtestpipeline</td><td>e259f6db-8ce2-45f1-b2d7-a719fde3b18f</td><td>2023-17-May 21:56:21</td><td>2023-17-May 21:56:21</td><td>(unknown)</td><td>My Great Tag</td><td></td></tr></table>

### Remove Tag from Pipeline

Tags are removed from a pipeline with the Wallaroo Tag `remove_from_pipeline(pipeline_id)` command, where `pipeline_id` is the integer value of the pipeline's id.

For this example, `currentTag` will be removed from `tagtest_pipeline`.  This will be verified through the `list_tags` and `search_pipelines` command.

```python
## remove from pipeline
currentTag.remove_from_pipeline(tagtest_pipeline.id())
```

    {'pipeline_pk_id': 45, 'tag_pk_id': 1}

```python
wl.list_tags()
```

(no tags)

```python
## Verify it was removed
wl.search_pipelines('My Great Tag')
```

(no pipelines)

