## Wallaroo SDK Tag Tutorial

The following tutorial demonstrates how to use Wallaroo Tags.  Tags are applied to either model versions or pipelines.  This allows organizations to track different versions of models, and search for what pipelines have been used for specific purposes such as testing versus production use.

The following will be demonstrated:

* List all tags in a Wallaroo instance.
* List all tags applied to a model.
* List all tags applied to a pipeline.
* Apply a tag to a model.
* Remove a tag from a model.
* Add a tag to a pipeline.
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
```

### Connect to Wallaroo

The following command is used to connect to a Wallaroo instance from within a Wallaroo Jupyter Hub service.  For more information on connecting to a Wallaroo instance, see the [Wallaroo SDK Guides](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/).

```python
wl = wallaroo.Client()
```

### Set Variables

The following variables are used to create or connect to existing workspace and pipeline.  The model name and model file are set as well.  Adjust as required for your organization's needs.

The methods `get_workspace` and `get_pipeline` are used to either create a new workspace and pipeline based on the variables below, or connect to an existing workspace and pipeline with the same name.  Once complete, the workspace will be set as the current workspace where pipelines and models are used.

```python
workspace_name = 'tagtestworkspace'
pipeline_name = 'tagtestpipeline'
model_name = 'tagtestmodel'
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

    {'name': 'tagtestworkspace', 'id': 2, 'archived': False, 'created_by': '0bbf2f62-a4f1-4fe5-aad8-ec1cb7485939', 'created_at': '2022-11-29T17:14:57.185365+00:00', 'models': [], 'pipelines': []}

### Upload Model and Create Pipeline

The `tagtest_model` and `tagtest_pipeline` will be created (or connected if already existing) based on the variables set earlier.

```python
tagtest_model = wl.upload_model(model_name, model_file_name).configure()
tagtest_model
```

    {'name': 'tagtestmodel', 'version': '70169e97-fb7e-4922-82ba-4f5d37e75253', 'file_name': 'ccfraud.onnx', 'image_path': None, 'last_update_time': datetime.datetime(2022, 11, 29, 17, 15, 21, 703465, tzinfo=tzutc())}

```python
tagtest_pipeline = get_pipeline(pipeline_name)
tagtest_pipeline
```

<table><tr><th>name</th> <td>tagtestpipeline</td></tr><tr><th>created</th> <td>2022-11-29 17:15:21.785352+00:00</td></tr><tr><th>last_updated</th> <td>2022-11-29 17:15:21.785352+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>5a4ff3c7-1a2d-4b0a-ad9f-78941e6f5677</td></tr><tr><th>steps</th> <td></td></tr></table>

### List Pipeline and Model Tags

This tutorial assumes that no tags are currently existing, but that can be verified through the Wallaroo client `list_pipelines` and `list_models` commands.  For this demonstration, it is recommended to use unique tags to verify each example.

```python
wl.list_pipelines()
```

<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>tags</th><th>versions</th><th>steps</th></tr><tr><td>tagtestpipeline</td><td>2022-29-Nov 17:15:21</td><td>2022-29-Nov 17:15:21</td><td>(unknown)</td><td></td><td>5a4ff3c7-1a2d-4b0a-ad9f-78941e6f5677</td><td></td></tr></table>

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
    <td>tagtestmodel</td>
    <td>1</td>
    <td>""</td>
    <td>2022-11-29 17:15:21.703465+00:00</td>
    <td>2022-11-29 17:15:21.703465+00:00</td>
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

Tags are listed with the Wallaroo client command `list_tags()`, which shows all tags and what models and pipelines they have been assigned to.

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

    {'model_id': 1, 'tag_id': 1}

```python
# list all tags to verify

wl.list_tags()
```

<table><tr><th>id</th><th>tag</th><th>models</th><th>pipelines</th></tr><tr><td>1</td><td>My Great Tag</td><td>[('tagtestmodel', ['70169e97-fb7e-4922-82ba-4f5d37e75253'])]</td><td>[]</td></tr></table>

### Search Models by Tag

Model versions can be searched via tags using the Wallaroo Client method `search_models(search_term)`, where `search_term` is a string value.  All models versions containing the tag will be displayed.  In this example, we will be using the text from our tag to list all models that have the text from `currentTag` in them.

```python
# Search models by tag

wl.search_models('My Great Tag')
```

<table><tr><th>name</th><th>version</th><th>file_name</th><th>image_path</th><th>last_update_time</th></tr>
            <tr>
                <td>tagtestmodel</td>
                <td>70169e97-fb7e-4922-82ba-4f5d37e75253</td>
                <td>ccfraud.onnx</td>
                <td>None</td>
                <td>2022-11-29 17:15:21.703465+00:00</td>
            </tr>
          </table>

### Remove Tag from Model

Tags are removed from models using the Wallaroo Tag `remove_from_model(model_id)` command.

In this example, the `currentTag` will be removed from `tagtest_model`.  A list of all tags will be shown with the `list_tags` command, followed by searching the models for the tag to verify it has been removed.

```python
### remove tag from model

currentTag.remove_from_model(tagtest_model.id())
```

    {'model_id': 1, 'tag_id': 1}

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

    {'pipeline_pk_id': 1, 'tag_pk_id': 1}

```python
# list tags to verify it was added to tagtest_pipeline

wl.list_tags()

```

<table><tr><th>id</th><th>tag</th><th>models</th><th>pipelines</th></tr><tr><td>1</td><td>My Great Tag</td><td>[]</td><td>[('tagtestpipeline', ['5a4ff3c7-1a2d-4b0a-ad9f-78941e6f5677'])]</td></tr></table>

```python
# get all of the pipelines to show the tag was added to tagtest-pipeline

wl.list_pipelines()
```

<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>tags</th><th>versions</th><th>steps</th></tr><tr><td>tagtestpipeline</td><td>2022-29-Nov 17:15:21</td><td>2022-29-Nov 17:15:21</td><td>(unknown)</td><td>My Great Tag</td><td>5a4ff3c7-1a2d-4b0a-ad9f-78941e6f5677</td><td></td></tr></table>

### Search Pipelines by Tag

Pipelines can be searched through the Wallaroo Client `search_pipelines(search_term)` method, where `search_term` is a string value for tags assigned to the pipelines.

In this example, the text "My Great Tag" that corresponds to `currentTag` will be searched for and displayed.

```python
wl.search_pipelines('My Great Tag')
```

<table><tr><th>name</th><th>version</th><th>creation_time</th><th>last_updated_time</th><th>deployed</th><th>tags</th><th>steps</th></tr><tr><td>tagtestpipeline</td><td>5a4ff3c7-1a2d-4b0a-ad9f-78941e6f5677</td><td>2022-29-Nov 17:15:21</td><td>2022-29-Nov 17:15:21</td><td>(unknown)</td><td>My Great Tag</td><td></td></tr></table>

### Remove Tag from Pipeline

Tags are removed from a pipeline with the Wallaroo Tag `remove_from_pipeline(pipeline_id)` command, where `pipeline_id` is the integer value of the pipeline's id.

For this example, `currentTag` will be removed from `tagtest_pipeline`.  This will be verified through the `list_tags` and `search_pipelines` command.

```python
## remove from pipeline
currentTag.remove_from_pipeline(tagtest_pipeline.id())
```

    {'pipeline_pk_id': 1, 'tag_pk_id': 1}

```python
wl.list_tags()
```

(no tags)

```python
## Verify it was removed
wl.search_pipelines('My Great Tag')
```

(no pipelines)

