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

### Connect to Wallaroo

The following command is used to connect to a Wallaroo instance from within a Wallaroo Jupyter Hub service.  For more information on connecting to a Wallaroo instance, see the [Wallaroo SDK Guides](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/).


```python
# Client connection from local Wallaroo instance

wl = wallaroo.Client()

# SSO login through keycloak

# wallarooPrefix = "YOUR PREFIX"
# wallarooSuffix = "YOUR SUFFIX"

# wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
#                     auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
#                     auth_type="sso")
```

### Arrow Support

As of the 2023.1 release, Wallaroo provides support for dataframe and Arrow for inference inputs.  This tutorial allows users to adjust their experience based on whether they have enabled Arrow support in their Wallaroo instance or not.

If Arrow support has been enabled, `arrowEnabled=True`. If disabled or you're not sure, set it to `arrowEnabled=False`

The examples below will be shown in an arrow enabled environment.


```python
import os
# Only set the below to make the OS environment ARROW_ENABLED to TRUE.  Otherwise, leave as is.
# os.environ["ARROW_ENABLED"]="True"

if "ARROW_ENABLED" not in os.environ or os.environ["ARROW_ENABLED"].casefold() == "False".casefold():
    arrowEnabled = False
else:
    arrowEnabled = True
print(arrowEnabled)
```

    True


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
        pipeline = wl.pipelines_by_name(pipeline_name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(pipeline_name)
    return pipeline
```


```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
```




    {'name': 'efxvtagtestworkspace', 'id': 15, 'archived': False, 'created_by': '435da905-31e2-4e74-b423-45c38edb5889', 'created_at': '2023-02-27T18:02:16.272068+00:00', 'models': [], 'pipelines': []}



### Upload Model and Create Pipeline

The `tagtest_model` and `tagtest_pipeline` will be created (or connected if already existing) based on the variables set earlier.


```python
tagtest_model = wl.upload_model(model_name, model_file_name).configure()
tagtest_model
```




    {'name': 'efxvtagtestmodel', 'version': '254a2888-0c8b-4172-97c3-c3547bbe6644', 'file_name': 'ccfraud.onnx', 'image_path': None, 'last_update_time': datetime.datetime(2023, 2, 27, 18, 2, 19, 155698, tzinfo=tzutc())}




```python
tagtest_pipeline = get_pipeline(pipeline_name)
tagtest_pipeline
```




<table><tr><th>name</th> <td>efxvtagtestpipeline</td></tr><tr><th>created</th> <td>2023-02-27 18:02:20.861896+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 18:02:20.861896+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>461dd376-e42e-4479-9de1-7a253c1c5197</td></tr><tr><th>steps</th> <td></td></tr></table>



### List Pipeline and Model Tags

This tutorial assumes that no tags are currently existing, but that can be verified through the Wallaroo client `list_pipelines` and `list_models` commands.  For this demonstration, it is recommended to use unique tags to verify each example.


```python
wl.list_pipelines()
```




<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>tags</th><th>versions</th><th>steps</th></tr><tr><td>efxvtagtestpipeline</td><td>2023-27-Feb 18:02:20</td><td>2023-27-Feb 18:02:20</td><td>(unknown)</td><td></td><td>461dd376-e42e-4479-9de1-7a253c1c5197</td><td></td></tr><tr><td>urldemopipeline</td><td>2023-27-Feb 17:55:12</td><td>2023-27-Feb 17:59:03</td><td>False</td><td></td><td>6db21694-9e11-42cb-914c-1528549cedca, 930fe54d-9503-4768-8bf9-499f72272098, 54158104-c71d-4980-a6a3-25564c909b44</td><td>urldemomodel</td></tr><tr><td>mlbaedgepipelineexample</td><td>2023-27-Feb 17:40:15</td><td>2023-27-Feb 17:44:15</td><td>False</td><td></td><td>b97189b0-7782-441a-84b0-2b2ed2fbf36b, 9b46e1e8-a40e-4a2d-a5f1-f2cef2ad57e9, f2aa4340-7495-4b72-b28c-98362eb72399</td><td>mlbaalohamodel</td></tr><tr><td>azwsedgepipelineexample</td><td>2023-27-Feb 17:37:37</td><td>2023-27-Feb 17:37:38</td><td>False</td><td></td><td>d8e4fce3-590c-46d5-871e-96bb1b0288c6, 93b18cbc-d951-43ba-9228-ef2e1add98cc</td><td>azwsalohamodel</td></tr><tr><td>ggwzhotswappipeline</td><td>2023-27-Feb 17:33:53</td><td>2023-27-Feb 17:34:11</td><td>False</td><td></td><td>a620354f-291e-4a98-b5f7-9d8bf165b1df, 3078dffa-4e10-41ef-85bc-e7a0de5afa82, ad943ff6-1a38-4304-a243-6958ba118df2</td><td>ggwzccfraudoriginal</td></tr><tr><td>uuzmhotswappipeline</td><td>2023-27-Feb 17:28:01</td><td>2023-27-Feb 17:32:15</td><td>False</td><td></td><td>869fd391-c562-4c51-b38a-07003d252e62, d6afc451-404b-4785-8ba0-28f0ba833f0b, 2d14a5bf-9aaf-4020-b385-9b69805f5c3c, 2b61eea9-cb7c-43c3-9ff3-2507cade98a1</td><td>uuzmccfraudoriginal</td></tr><tr><td>beticcfraudpipeline</td><td>2023-27-Feb 17:23:37</td><td>2023-27-Feb 17:23:38</td><td>False</td><td></td><td>118aefc4-b71e-4b51-84bd-85e31dbcb44a, 77463125-631b-427d-a60a-bff6d1a09eed</td><td>beticcfraudmodel</td></tr><tr><td>jnhcccfraudpipeline</td><td>2023-27-Feb 17:19:34</td><td>2023-27-Feb 17:19:36</td><td>False</td><td></td><td>a5e2db56-5ac5-49b5-9842-60b6dfe2980c, 7d50b378-e093-49dc-9458-74f439c0894d</td><td>jnhcccfraudmodel</td></tr></table>




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
    <td>efxvtagtestmodel</td>
    <td>1</td>
    <td>""</td>
    <td>2023-02-27 18:02:19.155698+00:00</td>
    <td>2023-02-27 18:02:19.155698+00:00</td>
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




    {'model_id': 10, 'tag_id': 1}




```python
# list all tags to verify

wl.list_tags()
```




<table><tr><th>id</th><th>tag</th><th>models</th><th>pipelines</th></tr><tr><td>1</td><td>My Great Tag</td><td>[('efxvtagtestmodel', ['254a2888-0c8b-4172-97c3-c3547bbe6644'])]</td><td>[]</td></tr></table>



### Search Models by Tag

Model versions can be searched via tags using the Wallaroo Client method `search_models(search_term)`, where `search_term` is a string value.  All models versions containing the tag will be displayed.  In this example, we will be using the text from our tag to list all models that have the text from `currentTag` in them.


```python
# Search models by tag

wl.search_models('My Great Tag')
```




<table><tr><th>name</th><th>version</th><th>file_name</th><th>image_path</th><th>last_update_time</th></tr>
            <tr>
                <td>efxvtagtestmodel</td>
                <td>254a2888-0c8b-4172-97c3-c3547bbe6644</td>
                <td>ccfraud.onnx</td>
                <td>None</td>
                <td>2023-02-27 18:02:19.155698+00:00</td>
            </tr>
          </table>



### Remove Tag from Model

Tags are removed from models using the Wallaroo Tag `remove_from_model(model_id)` command.

In this example, the `currentTag` will be removed from `tagtest_model`.  A list of all tags will be shown with the `list_tags` command, followed by searching the models for the tag to verify it has been removed.


```python
### remove tag from model

currentTag.remove_from_model(tagtest_model.id())
```




    {'model_id': 10, 'tag_id': 1}




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




    {'pipeline_pk_id': 20, 'tag_pk_id': 1}




```python
# list tags to verify it was added to tagtest_pipeline

wl.list_tags()

```




<table><tr><th>id</th><th>tag</th><th>models</th><th>pipelines</th></tr><tr><td>1</td><td>My Great Tag</td><td>[]</td><td>[('efxvtagtestpipeline', ['461dd376-e42e-4479-9de1-7a253c1c5197'])]</td></tr></table>




```python
# get all of the pipelines to show the tag was added to tagtest-pipeline

wl.list_pipelines()
```




<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>tags</th><th>versions</th><th>steps</th></tr><tr><td>efxvtagtestpipeline</td><td>2023-27-Feb 18:02:20</td><td>2023-27-Feb 18:02:20</td><td>(unknown)</td><td>My Great Tag</td><td>461dd376-e42e-4479-9de1-7a253c1c5197</td><td></td></tr><tr><td>urldemopipeline</td><td>2023-27-Feb 17:55:12</td><td>2023-27-Feb 17:59:03</td><td>False</td><td></td><td>6db21694-9e11-42cb-914c-1528549cedca, 930fe54d-9503-4768-8bf9-499f72272098, 54158104-c71d-4980-a6a3-25564c909b44</td><td>urldemomodel</td></tr><tr><td>mlbaedgepipelineexample</td><td>2023-27-Feb 17:40:15</td><td>2023-27-Feb 17:44:15</td><td>False</td><td></td><td>b97189b0-7782-441a-84b0-2b2ed2fbf36b, 9b46e1e8-a40e-4a2d-a5f1-f2cef2ad57e9, f2aa4340-7495-4b72-b28c-98362eb72399</td><td>mlbaalohamodel</td></tr><tr><td>azwsedgepipelineexample</td><td>2023-27-Feb 17:37:37</td><td>2023-27-Feb 17:37:38</td><td>False</td><td></td><td>d8e4fce3-590c-46d5-871e-96bb1b0288c6, 93b18cbc-d951-43ba-9228-ef2e1add98cc</td><td>azwsalohamodel</td></tr><tr><td>ggwzhotswappipeline</td><td>2023-27-Feb 17:33:53</td><td>2023-27-Feb 17:34:11</td><td>False</td><td></td><td>a620354f-291e-4a98-b5f7-9d8bf165b1df, 3078dffa-4e10-41ef-85bc-e7a0de5afa82, ad943ff6-1a38-4304-a243-6958ba118df2</td><td>ggwzccfraudoriginal</td></tr><tr><td>uuzmhotswappipeline</td><td>2023-27-Feb 17:28:01</td><td>2023-27-Feb 17:32:15</td><td>False</td><td></td><td>869fd391-c562-4c51-b38a-07003d252e62, d6afc451-404b-4785-8ba0-28f0ba833f0b, 2d14a5bf-9aaf-4020-b385-9b69805f5c3c, 2b61eea9-cb7c-43c3-9ff3-2507cade98a1</td><td>uuzmccfraudoriginal</td></tr><tr><td>beticcfraudpipeline</td><td>2023-27-Feb 17:23:37</td><td>2023-27-Feb 17:23:38</td><td>False</td><td></td><td>118aefc4-b71e-4b51-84bd-85e31dbcb44a, 77463125-631b-427d-a60a-bff6d1a09eed</td><td>beticcfraudmodel</td></tr><tr><td>jnhcccfraudpipeline</td><td>2023-27-Feb 17:19:34</td><td>2023-27-Feb 17:19:36</td><td>False</td><td></td><td>a5e2db56-5ac5-49b5-9842-60b6dfe2980c, 7d50b378-e093-49dc-9458-74f439c0894d</td><td>jnhcccfraudmodel</td></tr></table>



### Search Pipelines by Tag

Pipelines can be searched through the Wallaroo Client `search_pipelines(search_term)` method, where `search_term` is a string value for tags assigned to the pipelines.

In this example, the text "My Great Tag" that corresponds to `currentTag` will be searched for and displayed.


```python
wl.search_pipelines('My Great Tag')
```




<table><tr><th>name</th><th>version</th><th>creation_time</th><th>last_updated_time</th><th>deployed</th><th>tags</th><th>steps</th></tr><tr><td>efxvtagtestpipeline</td><td>461dd376-e42e-4479-9de1-7a253c1c5197</td><td>2023-27-Feb 18:02:20</td><td>2023-27-Feb 18:02:20</td><td>(unknown)</td><td>My Great Tag</td><td></td></tr></table>



### Remove Tag from Pipeline

Tags are removed from a pipeline with the Wallaroo Tag `remove_from_pipeline(pipeline_id)` command, where `pipeline_id` is the integer value of the pipeline's id.

For this example, `currentTag` will be removed from `tagtest_pipeline`.  This will be verified through the `list_tags` and `search_pipelines` command.


```python
## remove from pipeline
currentTag.remove_from_pipeline(tagtest_pipeline.id())
```




    {'pipeline_pk_id': 20, 'tag_pk_id': 1}




```python
wl.list_tags()
```




(no tags)




```python
## Verify it was removed
wl.search_pipelines('My Great Tag')
```




(no pipelines)


