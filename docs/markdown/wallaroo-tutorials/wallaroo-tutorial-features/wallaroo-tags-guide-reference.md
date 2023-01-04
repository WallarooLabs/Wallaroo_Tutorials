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
```

### Connect to Wallaroo

The following command is used to connect to a Wallaroo instance from within a Wallaroo Jupyter Hub service.  For more information on connecting to a Wallaroo instance, see the [Wallaroo SDK Guides](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/).

```python
# Login through local Wallaroo instance

# wl = wallaroo.Client()

# SSO login through keycloak

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR SUFFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")
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

    {'name': 'tagtestworkspace', 'id': 18, 'archived': False, 'created_by': 'f1f32bdf-9bd9-4595-a531-aca5778ceaf0', 'created_at': '2022-12-15T15:43:59.979692+00:00', 'models': [], 'pipelines': []}

### Upload Model and Create Pipeline

The `tagtest_model` and `tagtest_pipeline` will be created (or connected if already existing) based on the variables set earlier.

```python
tagtest_model = wl.upload_model(model_name, model_file_name).configure()
tagtest_model
```

    {'name': 'tagtestmodel', 'version': '1175c69e-db6d-487d-847d-840c5e29b41e', 'file_name': 'ccfraud.onnx', 'image_path': None, 'last_update_time': datetime.datetime(2022, 12, 15, 15, 44, 2, 66011, tzinfo=tzutc())}

```python
tagtest_pipeline = get_pipeline(pipeline_name)
tagtest_pipeline
```

<table><tr><th>name</th> <td>tagtestpipeline</td></tr><tr><th>created</th> <td>2022-12-15 15:44:03.356719+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-15 15:44:03.356719+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>dec7c573-004e-469c-a463-eac7d7c44489</td></tr><tr><th>steps</th> <td></td></tr></table>

### List Pipeline and Model Tags

This tutorial assumes that no tags are currently existing, but that can be verified through the Wallaroo client `list_pipelines` and `list_models` commands.  For this demonstration, it is recommended to use unique tags to verify each example.

```python
wl.list_pipelines()
```

<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>tags</th><th>versions</th><th>steps</th></tr><tr><td>tagtestpipeline</td><td>2022-15-Dec 15:44:03</td><td>2022-15-Dec 15:44:03</td><td>(unknown)</td><td></td><td>dec7c573-004e-469c-a463-eac7d7c44489</td><td></td></tr><tr><td>ccfraudpipeline</td><td>2022-15-Dec 15:40:52</td><td>2022-15-Dec 15:40:54</td><td>False</td><td></td><td>a923f5e0-23d5-49eb-8908-db85406387f1, c6ca1538-4a87-48e7-9187-c4d80d1b2a9c</td><td>ccfraudmodel</td></tr><tr><td>cc-shadow</td><td>2022-15-Dec 15:37:36</td><td>2022-15-Dec 15:37:41</td><td>False</td><td></td><td>1cd86d74-f74c-4291-bca5-8007ffb8fbc7, ec69f1b4-9037-42ca-8589-a392448edaf6</td><td>ccfraud-lstm</td></tr><tr><td>xgboost-regression-autoconvert-pipeline</td><td>2022-15-Dec 15:26:34</td><td>2022-15-Dec 15:26:34</td><td>(unknown)</td><td></td><td>076ef1be-0718-4c51-a412-fd5a4f45e125</td><td></td></tr><tr><td>xgboost-classification-autoconvert-pipeline</td><td>2022-15-Dec 15:24:09</td><td>2022-15-Dec 15:24:09</td><td>(unknown)</td><td></td><td>dbdfb968-3ff5-4a12-bb30-888b2be5a9f2</td><td></td></tr><tr><td>bikedayevalpipeline</td><td>2022-14-Dec 22:29:10</td><td>2022-14-Dec 22:31:28</td><td>False</td><td></td><td>f3df950a-91f8-4fb9-b961-038eddc83b11, ec929e70-7c5d-49e2-8c4e-2e64d86a18b0, d725b33a-35e5-40a1-bc56-5c6504dbdfda</td><td>bikedaymodel</td></tr><tr><td>keras-autoconvert-pipeline</td><td>2022-14-Dec 22:21:00</td><td>2022-14-Dec 22:27:50</td><td>False</td><td></td><td>38779ba0-0f7d-4d3c-802f-ff01fd735464, cfeb911c-8c44-477a-b9b0-e5d238adc756, 4fdf73f0-0861-40ce-9dcc-3857ecf1c8dc, 337879b9-32b9-4ad8-a6f2-2dc18abfa9f9</td><td>simple-sentiment-model</td></tr><tr><td>imdbpipeline</td><td>2022-14-Dec 22:09:42</td><td>2022-14-Dec 22:09:48</td><td>False</td><td></td><td>bfeea210-003f-4d83-9aad-73cd4ba1b3bf, efd29028-df4a-46a5-a552-32346b9c7196</td><td>embedder-o</td></tr><tr><td>demandcurvepipeline</td><td>2022-14-Dec 22:06:45</td><td>2022-14-Dec 22:06:50</td><td>False</td><td></td><td>bc32a3e8-fa4e-4e9a-85bf-b9cec7acd0eb, 76ff1b6c-742c-4992-a95d-cd266771b185</td><td>preprocess</td></tr><tr><td>anomaly-housing-pipeline</td><td>2022-14-Dec 22:03:38</td><td>2022-14-Dec 22:03:39</td><td>False</td><td></td><td>a3c41417-a663-47c9-bbd4-6de85d139b83, 76077c2f-52fe-43fa-a906-18dc4d3a76d3</td><td>anomaly-housing-model</td></tr><tr><td>randomsplitpipeline-demo</td><td>2022-14-Dec 21:04:30</td><td>2022-14-Dec 21:14:19</td><td>False</td><td></td><td>1f061d93-9d23-4e03-882e-ed1299bb99a2, 53118f9c-582f-42c8-8820-934d7c10719c, 72ccc6f7-e4af-4a6e-9f14-15c509a0e851, 7f4540ad-f1eb-4c77-880b-3f24eca1ca68, 13d61236-1fb9-4ac2-833e-4c8f36bffe2b, 986dd8f4-bd88-4978-8c13-3a1ae59113f2, 24176a3b-5da9-433e-820b-e411fc9a1b93</td><td>aloha-control</td></tr><tr><td>alohapipeline-regression</td><td>2022-14-Dec 20:03:19</td><td>2022-14-Dec 20:53:33</td><td>False</td><td></td><td>9929701c-59bf-4e22-8b96-f8b634b6f115, 462a234b-8a6c-4a64-99a7-2881da71f37a, 00e1f944-f780-4185-86e9-0e65ca33fdac, 5e699f22-4a9a-47e0-892c-6855baa5b9ba, e9516442-3eec-464b-9b77-d773a55861b3, 2aae2bc8-a274-4cd0-b6c1-148e3007223b, fb55aa25-d0f3-4520-92b1-40c054e85e05, 0a06ed9d-5201-452f-895d-8e86e3e6c0cc, ccfff584-a0f8-4091-9d3e-e0d429b34682, 8bfdc3a5-2e67-4e01-b412-262b7a39f09f, 0a3d5233-c5e7-48aa-bc34-6fe69bc588b0, b7f1efeb-2755-41fb-90cd-c01a70b68d74</td><td>alohamodel-regression</td></tr><tr><td>housepricepipe</td><td>2022-13-Dec 16:32:11</td><td>2022-13-Dec 20:37:56</td><td>False</td><td></td><td>0675e460-cf1b-4c9f-971a-d275d7086a70, a775d89b-b78c-40de-9d71-92787c67013b, 3dac5e67-3838-4456-aade-bed972bade9b, 3e007172-62ed-4a45-bad0-3ca4f7ad83cc</td><td>housepricemodel</td></tr><tr><td>sdkpipeline</td><td>2022-12-Dec 22:53:13</td><td>2022-14-Dec 20:51:06</td><td>False</td><td></td><td>c0839ce0-9d9c-4914-9c2b-e150a8e979fb, ef7429d1-f83d-4e49-bd5a-4ae63d7f005d, 25051356-f70d-472a-9eae-59e21f31f9a9, 48b6e755-adfd-4da9-8104-7b3494128c66, 287a0170-ae2f-4360-b121-1fd89ba31df8, ef60d525-e959-4b4f-acb8-e7f1c9540668, 751d6910-14c2-47d2-bd33-debf39bb475b, 85cb2cc5-39a2-4e68-b5bf-e3cceb270df2, 862b5c66-98a6-4dee-9c92-4c82d7ce49a6, 7a373546-27b8-4541-bc96-33a30e14200c, 929a93db-1478-400a-8e21-0ecfd8090faf, 682dc9af-c3b7-401d-a2bd-d8511dfa3bcc</td><td>sdkmodel</td></tr><tr><td>alohapipeline</td><td>2022-12-Dec 22:48:05</td><td>2022-14-Dec 19:58:42</td><td>False</td><td></td><td>19b2c0c9-95b4-42b1-be87-65fb20bdc107, e9b01132-bf2d-4bbc-90a7-449220f995de, 03912e7c-e295-4d86-bb29-0b1ad2d37851, 9f286821-a9f4-4819-8b73-7d7a4f65faa2, 6d14aecf-9dd7-49d4-809f-ce6cefeff526, 68f1dda3-b054-4d9e-9cc2-98506e5513d6, 923ea235-dc4a-44eb-a5eb-da8b2f7d9b35, 541e38f4-d863-4ac7-af6e-966e9641613f</td><td>alohamodel</td></tr></table>

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
    <td>2022-12-15 15:44:02.066011+00:00</td>
    <td>2022-12-15 15:44:02.066011+00:00</td>
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

    {'model_id': 48, 'tag_id': 1}

```python
# list all tags to verify

wl.list_tags()
```

<table><tr><th>id</th><th>tag</th><th>models</th><th>pipelines</th></tr><tr><td>1</td><td>My Great Tag</td><td>[('tagtestmodel', ['1175c69e-db6d-487d-847d-840c5e29b41e'])]</td><td>[]</td></tr></table>

### Search Models by Tag

Model versions can be searched via tags using the Wallaroo Client method `search_models(search_term)`, where `search_term` is a string value.  All models versions containing the tag will be displayed.  In this example, we will be using the text from our tag to list all models that have the text from `currentTag` in them.

```python
# Search models by tag

wl.search_models('My Great Tag')
```

<table><tr><th>name</th><th>version</th><th>file_name</th><th>image_path</th><th>last_update_time</th></tr>
            <tr>
                <td>tagtestmodel</td>
                <td>1175c69e-db6d-487d-847d-840c5e29b41e</td>
                <td>ccfraud.onnx</td>
                <td>None</td>
                <td>2022-12-15 15:44:02.066011+00:00</td>
            </tr>
          </table>

### Remove Tag from Model

Tags are removed from models using the Wallaroo Tag `remove_from_model(model_id)` command.

In this example, the `currentTag` will be removed from `tagtest_model`.  A list of all tags will be shown with the `list_tags` command, followed by searching the models for the tag to verify it has been removed.

```python
### remove tag from model

currentTag.remove_from_model(tagtest_model.id())
```

    {'model_id': 48, 'tag_id': 1}

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

    {'pipeline_pk_id': 63, 'tag_pk_id': 1}

```python
# list tags to verify it was added to tagtest_pipeline

wl.list_tags()

```

<table><tr><th>id</th><th>tag</th><th>models</th><th>pipelines</th></tr><tr><td>1</td><td>My Great Tag</td><td>[]</td><td>[('tagtestpipeline', ['dec7c573-004e-469c-a463-eac7d7c44489'])]</td></tr></table>

```python
# get all of the pipelines to show the tag was added to tagtest-pipeline

wl.list_pipelines()
```

<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>tags</th><th>versions</th><th>steps</th></tr><tr><td>tagtestpipeline</td><td>2022-15-Dec 15:44:03</td><td>2022-15-Dec 15:44:03</td><td>(unknown)</td><td>My Great Tag</td><td>dec7c573-004e-469c-a463-eac7d7c44489</td><td></td></tr><tr><td>ccfraudpipeline</td><td>2022-15-Dec 15:40:52</td><td>2022-15-Dec 15:40:54</td><td>False</td><td></td><td>a923f5e0-23d5-49eb-8908-db85406387f1, c6ca1538-4a87-48e7-9187-c4d80d1b2a9c</td><td>ccfraudmodel</td></tr><tr><td>cc-shadow</td><td>2022-15-Dec 15:37:36</td><td>2022-15-Dec 15:37:41</td><td>False</td><td></td><td>1cd86d74-f74c-4291-bca5-8007ffb8fbc7, ec69f1b4-9037-42ca-8589-a392448edaf6</td><td>ccfraud-lstm</td></tr><tr><td>xgboost-regression-autoconvert-pipeline</td><td>2022-15-Dec 15:26:34</td><td>2022-15-Dec 15:26:34</td><td>(unknown)</td><td></td><td>076ef1be-0718-4c51-a412-fd5a4f45e125</td><td></td></tr><tr><td>xgboost-classification-autoconvert-pipeline</td><td>2022-15-Dec 15:24:09</td><td>2022-15-Dec 15:24:09</td><td>(unknown)</td><td></td><td>dbdfb968-3ff5-4a12-bb30-888b2be5a9f2</td><td></td></tr><tr><td>bikedayevalpipeline</td><td>2022-14-Dec 22:29:10</td><td>2022-14-Dec 22:31:28</td><td>False</td><td></td><td>f3df950a-91f8-4fb9-b961-038eddc83b11, ec929e70-7c5d-49e2-8c4e-2e64d86a18b0, d725b33a-35e5-40a1-bc56-5c6504dbdfda</td><td>bikedaymodel</td></tr><tr><td>keras-autoconvert-pipeline</td><td>2022-14-Dec 22:21:00</td><td>2022-14-Dec 22:27:50</td><td>False</td><td></td><td>38779ba0-0f7d-4d3c-802f-ff01fd735464, cfeb911c-8c44-477a-b9b0-e5d238adc756, 4fdf73f0-0861-40ce-9dcc-3857ecf1c8dc, 337879b9-32b9-4ad8-a6f2-2dc18abfa9f9</td><td>simple-sentiment-model</td></tr><tr><td>imdbpipeline</td><td>2022-14-Dec 22:09:42</td><td>2022-14-Dec 22:09:48</td><td>False</td><td></td><td>bfeea210-003f-4d83-9aad-73cd4ba1b3bf, efd29028-df4a-46a5-a552-32346b9c7196</td><td>embedder-o</td></tr><tr><td>demandcurvepipeline</td><td>2022-14-Dec 22:06:45</td><td>2022-14-Dec 22:06:50</td><td>False</td><td></td><td>bc32a3e8-fa4e-4e9a-85bf-b9cec7acd0eb, 76ff1b6c-742c-4992-a95d-cd266771b185</td><td>preprocess</td></tr><tr><td>anomaly-housing-pipeline</td><td>2022-14-Dec 22:03:38</td><td>2022-14-Dec 22:03:39</td><td>False</td><td></td><td>a3c41417-a663-47c9-bbd4-6de85d139b83, 76077c2f-52fe-43fa-a906-18dc4d3a76d3</td><td>anomaly-housing-model</td></tr><tr><td>randomsplitpipeline-demo</td><td>2022-14-Dec 21:04:30</td><td>2022-14-Dec 21:14:19</td><td>False</td><td></td><td>1f061d93-9d23-4e03-882e-ed1299bb99a2, 53118f9c-582f-42c8-8820-934d7c10719c, 72ccc6f7-e4af-4a6e-9f14-15c509a0e851, 7f4540ad-f1eb-4c77-880b-3f24eca1ca68, 13d61236-1fb9-4ac2-833e-4c8f36bffe2b, 986dd8f4-bd88-4978-8c13-3a1ae59113f2, 24176a3b-5da9-433e-820b-e411fc9a1b93</td><td>aloha-control</td></tr><tr><td>alohapipeline-regression</td><td>2022-14-Dec 20:03:19</td><td>2022-14-Dec 20:53:33</td><td>False</td><td></td><td>9929701c-59bf-4e22-8b96-f8b634b6f115, 462a234b-8a6c-4a64-99a7-2881da71f37a, 00e1f944-f780-4185-86e9-0e65ca33fdac, 5e699f22-4a9a-47e0-892c-6855baa5b9ba, e9516442-3eec-464b-9b77-d773a55861b3, 2aae2bc8-a274-4cd0-b6c1-148e3007223b, fb55aa25-d0f3-4520-92b1-40c054e85e05, 0a06ed9d-5201-452f-895d-8e86e3e6c0cc, ccfff584-a0f8-4091-9d3e-e0d429b34682, 8bfdc3a5-2e67-4e01-b412-262b7a39f09f, 0a3d5233-c5e7-48aa-bc34-6fe69bc588b0, b7f1efeb-2755-41fb-90cd-c01a70b68d74</td><td>alohamodel-regression</td></tr><tr><td>housepricepipe</td><td>2022-13-Dec 16:32:11</td><td>2022-13-Dec 20:37:56</td><td>False</td><td></td><td>0675e460-cf1b-4c9f-971a-d275d7086a70, a775d89b-b78c-40de-9d71-92787c67013b, 3dac5e67-3838-4456-aade-bed972bade9b, 3e007172-62ed-4a45-bad0-3ca4f7ad83cc</td><td>housepricemodel</td></tr><tr><td>sdkpipeline</td><td>2022-12-Dec 22:53:13</td><td>2022-14-Dec 20:51:06</td><td>False</td><td></td><td>c0839ce0-9d9c-4914-9c2b-e150a8e979fb, ef7429d1-f83d-4e49-bd5a-4ae63d7f005d, 25051356-f70d-472a-9eae-59e21f31f9a9, 48b6e755-adfd-4da9-8104-7b3494128c66, 287a0170-ae2f-4360-b121-1fd89ba31df8, ef60d525-e959-4b4f-acb8-e7f1c9540668, 751d6910-14c2-47d2-bd33-debf39bb475b, 85cb2cc5-39a2-4e68-b5bf-e3cceb270df2, 862b5c66-98a6-4dee-9c92-4c82d7ce49a6, 7a373546-27b8-4541-bc96-33a30e14200c, 929a93db-1478-400a-8e21-0ecfd8090faf, 682dc9af-c3b7-401d-a2bd-d8511dfa3bcc</td><td>sdkmodel</td></tr><tr><td>alohapipeline</td><td>2022-12-Dec 22:48:05</td><td>2022-14-Dec 19:58:42</td><td>False</td><td></td><td>19b2c0c9-95b4-42b1-be87-65fb20bdc107, e9b01132-bf2d-4bbc-90a7-449220f995de, 03912e7c-e295-4d86-bb29-0b1ad2d37851, 9f286821-a9f4-4819-8b73-7d7a4f65faa2, 6d14aecf-9dd7-49d4-809f-ce6cefeff526, 68f1dda3-b054-4d9e-9cc2-98506e5513d6, 923ea235-dc4a-44eb-a5eb-da8b2f7d9b35, 541e38f4-d863-4ac7-af6e-966e9641613f</td><td>alohamodel</td></tr></table>

### Search Pipelines by Tag

Pipelines can be searched through the Wallaroo Client `search_pipelines(search_term)` method, where `search_term` is a string value for tags assigned to the pipelines.

In this example, the text "My Great Tag" that corresponds to `currentTag` will be searched for and displayed.

```python
wl.search_pipelines('My Great Tag')
```

<table><tr><th>name</th><th>version</th><th>creation_time</th><th>last_updated_time</th><th>deployed</th><th>tags</th><th>steps</th></tr><tr><td>tagtestpipeline</td><td>dec7c573-004e-469c-a463-eac7d7c44489</td><td>2022-15-Dec 15:44:03</td><td>2022-15-Dec 15:44:03</td><td>(unknown)</td><td>My Great Tag</td><td></td></tr></table>

### Remove Tag from Pipeline

Tags are removed from a pipeline with the Wallaroo Tag `remove_from_pipeline(pipeline_id)` command, where `pipeline_id` is the integer value of the pipeline's id.

For this example, `currentTag` will be removed from `tagtest_pipeline`.  This will be verified through the `list_tags` and `search_pipelines` command.

```python
## remove from pipeline
currentTag.remove_from_pipeline(tagtest_pipeline.id())
```

    {'pipeline_pk_id': 63, 'tag_pk_id': 1}

```python
wl.list_tags()
```

(no tags)

```python
## Verify it was removed
wl.search_pipelines('My Great Tag')
```

(no pipelines)

