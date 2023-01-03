## Assays

**IMPORTANT NOTE**: These assays were run in a Wallaroo environment with canned historical data.  See the [Wallaroo Assay Tutorial](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_insights) for details on setting up this environment.  This historical data is **required** for these examples.

* [Create Assay](#create-assay)
* [List Assays](#list-assays)
* [Activate or Deactivate Assay](#activate-or-deactivate-assay)
* [Create Interactive Baseline](#create-interactive-baseline)
* [Get Assay Baseline](#get-assay-baseline)
* [Run Assay Interactively](#run-assay-interactively)
* [Get Assay Results](#get-assay-results)

### Create Assay

Create a new array in a specified pipeline.

* **PARAMETERS**
  * **id** - (*OPTIONAL int*):  The numerical identifier for the assay.
  * **name** - (*REQUIRED string*): The name of the assay.
  * **pipeline_id** - (*REQUIRED int*): The numerical idenfifier the assay will be placed into.
  * **pipeline_name** - (*REQUIRED string*): The name of the pipeline
  * **active** - (*REQUIRED bool*): Indicates whether the assay will be active upon creation or not.
  * **status** - (*REQUIRED string*): The status of the assay upon creation.
  * **iopath** - (*REQUIRED string*): The iopath of the assay.
  * **baseline** - (*REQUIRED baseline*): The baseline for the assay.
    * **Fixed** - (*REQUIRED AssayFixConfiguration*): The fixed configuration for the assay.
      * **pipeline** - (*REQUIRED string*): The name of the pipeline with the baseline data.
      * **model** - (*REQUIRED string*): The name of the model used.
      * **start_at** - (*REQUIRED string*): The DateTime of the baseline start date.
      * **end_at** - (*REQUIRED string*): The DateTime of the baseline end date.
  * **window** (*REQUIRED AssayWindow*): Assay window.
    * **pipeline** - (*REQUIRED string*): The name of the pipeline for the assay window.
    * **model** - (*REQUIRED string*): The name of the model used for the assay window.
    * **width** - (*REQUIRED string*): The width of the assay window.
    * **start** - (*OPTIONAL string*): The DateTime of when to start the assay window.
    * **interval** - (*OPTIONAL string*): The assay window interval.
  * **summarizer** - (*REQUIRED AssaySummerizer*): The summarizer type for the array aka "advanced settings" in the Wallaroo Dashboard UI.
    * **type** - (*REQUIRED string*): Type of summarizer.
    * **bin_mode** - (*REQUIRED string*): The binning model type.  Values can be:
      * Quantile
      * Equal
    * **aggregation** - (*REQUIRED string*): Aggregation type.
    * **metric** - (*REQUIRED string*): Metric type.  Values can be:
      * PSI
      * Maximum Difference of Bins
      * Sum of the Difference of Bins
    * **num_bins** - (*REQUIRED int*): The number of bins.  Recommanded values are between 5 and 14.
    * **bin_weights** - (*OPTIONAL AssayBinWeight*): The weights assigned to the assay bins.
    * **bin_width** - (*OPTIONAL AssayBinWidth*): The width assigned to the assay bins.
    * **provided_edges** - (*OPTIONAL AssayProvidedEdges*): The edges used for the assay bins.
    * **add_outlier_edges** - (*REQUIRED bool*): Indicates whether to add outlier edges or not.
  * **warning_threshold** - (*OPTIONAL number*): Optional warning threshold.
  * **alert_threshold** - (*REQUIRED number*): Alert threshold.
  * **run_until** - (*OPTIONAL string*): DateTime of when to end the assay.
  * **workspace_id** - (*REQUIRED integer*): The workspace the assay is part of.
  * **model_insights_url** - (*OPTIONAL string*): URL for model insights.
* **RETURNS**
  * **assay_id** - (*integer*): The id of the new assay.

As noted this example requires the [Wallaroo Assay Tutorial](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_insights) for historical data.  Before running this example, set the sample pipeline id, pipeline, name, model name, and workspace id in the code sample below.  For more information on retrieving this information, see the [Wallaroo Developer Guides](https://docs.wallaroo.ai/wallaroo-developer-guides/).

```python
# Create assay

apiRequest = "/assays/create"

exampleAssayName = "api_assay_test2"

## Now get all of the assays for the pipeline in workspace 4 `housepricedrift`

exampleAssayPipelineId = 4
exampleAssayPipelineName = "housepricepipe"
exampleAssayModelName = "housepricemodel"
exampleAssayWorkspaceId = 4

# iopath can be input 00 or output 0 0
data = {
    'name': exampleAssayName,
    'pipeline_id': exampleAssayPipelineId,
    'pipeline_name': exampleAssayPipelineName,
    'active': True,
    'status': 'active',
    'iopath': "input 0 0",
    'baseline': {
        'Fixed': {
            'pipeline': exampleAssayPipelineName,
            'model': 'houseprice-model-yns',
            'start_at': '2022-01-01T00:00:00-05:00',
            'end_at': '2022-01-02T00:00:00-05:00'
        }
    },
    'window': {
        'pipeline': exampleAssayPipelineName,
        'model': exampleAssayModelName,
        'width': '24 hours',
        'start': None,
        'interval': None
    },
    'summarizer': {
        'type': 'UnivariateContinuous',
        'bin_mode': 'Quantile',
        'aggregation': 'Density',
        'metric': 'PSI',
        'num_bins': 5,
        'bin_weights': None,
        'bin_width': None,
        'provided_edges': None,
        'add_outlier_edges': True
    },
    'warning_threshold': 0,
    'alert_threshold': 0.1,
    'run_until': None,
    'workspace_id': exampleAssayWorkspaceId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
example_assay_id = response['assay_id']
response
```

    {'assay_id': 2}

### List Assays

Lists all assays in the specified pipeline.

* **PARAMETERS**
  * **pipeline_id** - (*REQUIRED int*):  The numerical ID of the pipeline.
* **RETURNS**
  * **assays** - (*Array assays*): A list of all assays.

Example:  Display a list of all assays in a workspace.  This will assume we have a workspace with an existing Assay and the associated data has been upload.  See the tutorial [Wallaroo Assays Tutorial](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_insights).

For this reason, these values are hard coded for now.

```python
## First list all of the workspaces and the list of pipelines

# List workspaces

apiRequest = "/workspaces/list"

data = {
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    {'workspaces': [{'id': 1,
       'name': 'john.hansarick@wallaroo.ai - Default Workspace',
       'created_at': '2022-10-10T16:32:45.355874+00:00',
       'created_by': 'f68760ad-a27c-4f9b-808f-0b512f07571f',
       'archived': False,
       'models': [],
       'pipelines': []},
      {'id': 3,
       'name': 'testapiworkspace-e87e543f-25f1-4f6d-82c6-4eb48902575a',
       'created_at': '2022-10-10T18:25:27.926919+00:00',
       'created_by': 'f68760ad-a27c-4f9b-808f-0b512f07571f',
       'archived': False,
       'models': [1],
       'pipelines': [1, 2, 3]},
      {'id': 4,
       'name': 'housepricedrift',
       'created_at': '2022-10-10T18:38:50.748057+00:00',
       'created_by': 'f68760ad-a27c-4f9b-808f-0b512f07571f',
       'archived': False,
       'models': [2],
       'pipelines': [4]},
      {'id': 5,
       'name': 'housepricedrifts',
       'created_at': '2022-10-10T18:45:00.152716+00:00',
       'created_by': 'f68760ad-a27c-4f9b-808f-0b512f07571f',
       'archived': False,
       'models': [],
       'pipelines': []}]}

```python
# Get assays

apiRequest = "/assays/list"

data = {
    "pipeline_id": exampleAssayPipelineId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    [{'id': 3,
      'name': 'example assay',
      'active': True,
      'status': 'created',
      'warning_threshold': None,
      'alert_threshold': 0.1,
      'pipeline_id': 4,
      'pipeline_name': 'housepricepipe',
      'last_run': None,
      'next_run': '2022-10-10T19:00:43.941894+00:00',
      'run_until': None,
      'updated_at': '2022-10-10T19:00:43.945411+00:00',
      'baseline': {'Fixed': {'pipeline': 'housepricepipe',
        'model': 'housepricemodel',
        'start_at': '2022-01-01T00:00:00+00:00',
        'end_at': '2022-01-02T00:00:00+00:00'}},
      'window': {'pipeline': 'housepricepipe',
       'model': 'housepricemodel',
       'width': '24 hours',
       'start': None,
       'interval': None},
      'summarizer': {'type': 'UnivariateContinuous',
       'bin_mode': 'Quantile',
       'aggregation': 'Density',
       'metric': 'PSI',
       'num_bins': 5,
       'bin_weights': None,
       'bin_width': None,
       'provided_edges': None,
       'add_outlier_edges': True}},
     {'id': 2,
      'name': 'api_assay_test2',
      'active': True,
      'status': 'created',
      'warning_threshold': 0.0,
      'alert_threshold': 0.1,
      'pipeline_id': 4,
      'pipeline_name': 'housepricepipe',
      'last_run': None,
      'next_run': '2022-10-10T18:53:16.444786+00:00',
      'run_until': None,
      'updated_at': '2022-10-10T18:53:16.450269+00:00',
      'baseline': {'Fixed': {'pipeline': 'housepricepipe',
        'model': 'houseprice-model-yns',
        'start_at': '2022-01-01T00:00:00-05:00',
        'end_at': '2022-01-02T00:00:00-05:00'}},
      'window': {'pipeline': 'housepricepipe',
       'model': 'housepricemodel',
       'width': '24 hours',
       'start': None,
       'interval': None},
      'summarizer': {'type': 'UnivariateContinuous',
       'bin_mode': 'Quantile',
       'aggregation': 'Density',
       'metric': 'PSI',
       'num_bins': 5,
       'bin_weights': None,
       'bin_width': None,
       'provided_edges': None,
       'add_outlier_edges': True}},
     {'id': 1,
      'name': 'api_assay_test',
      'active': True,
      'status': 'created',
      'warning_threshold': 0.0,
      'alert_threshold': 0.1,
      'pipeline_id': 4,
      'pipeline_name': 'housepricepipe',
      'last_run': None,
      'next_run': '2022-10-10T18:48:00.829479+00:00',
      'run_until': None,
      'updated_at': '2022-10-10T18:48:00.833336+00:00',
      'baseline': {'Fixed': {'pipeline': 'housepricepipe',
        'model': 'houseprice-model-yns',
        'start_at': '2022-01-01T00:00:00-05:00',
        'end_at': '2022-01-02T00:00:00-05:00'}},
      'window': {'pipeline': 'housepricepipe',
       'model': 'housepricemodel',
       'width': '24 hours',
       'start': None,
       'interval': None},
      'summarizer': {'type': 'UnivariateContinuous',
       'bin_mode': 'Quantile',
       'aggregation': 'Density',
       'metric': 'PSI',
       'num_bins': 5,
       'bin_weights': None,
       'bin_width': None,
       'provided_edges': None,
       'add_outlier_edges': True}}]

### Activate or Deactivate Assay

Activates or deactivates an existing assay.

* **Parameters**
  * **id** - (*REQUIRED int*): The numerical id of the assay.
  * **active** - (*REQUIRED bool*): True to activate the assay, False to deactivate it.
* **Returns**
  * * **id** - (*integer*): The numerical id of the assay.
  * **active** - (*bool*): True to activate the assay, False to deactivate it.

Example:  Assay 8 "House Output Assay" will be deactivated then activated.

```python
# Deactivate assay

apiRequest = "/assays/set_active"

data = {
    'id': example_assay_id,
    'active': False
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    {'id': 2, 'active': False}

```python
# Activate assay

apiRequest = "/assays/set_active"

data = {
    'id': example_assay_id,
    'active': True
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    {'id': 2, 'active': True}

### Create Interactive Baseline

Creates an interactive assay baseline.

* **PARAMETERS**
  * **id** - (*REQUIRED int*):  The numerical identifier for the assay.
  * **name** - (*REQUIRED string*): The name of the assay.
  * **pipeline_id** - (*REQUIRED int*): The numerical idenfifier the assay will be placed into.
  * **pipeline_name** - (*REQUIRED string*): The name of the pipeline
  * **active** - (*REQUIRED bool*): Indicates whether the assay will be active upon creation or not.
  * **status** - (*REQUIRED string*): The status of the assay upon creation.
  * **iopath** - (*REQUIRED string*): The iopath of the assay.
  * **baseline** - (*REQUIRED baseline*): The baseline for the assay.
    * **Fixed** - (*REQUIRED AssayFixConfiguration*): The fixed configuration for the assay.
      * **pipeline** - (*REQUIRED string*): The name of the pipeline with the baseline data.
      * **model** - (*REQUIRED string*): The name of the model used.
      * **start_at** - (*REQUIRED string*): The DateTime of the baseline start date.
      * **end_at** - (*REQUIRED string*): The DateTime of the baseline end date.
  * **window** (*REQUIRED AssayWindow*): Assay window.
    * **pipeline** - (*REQUIRED string*): The name of the pipeline for the assay window.
    * **model** - (*REQUIRED string*): The name of the model used for the assay window.
    * **width** - (*REQUIRED string*): The width of the assay window.
    * **start** - (*OPTIONAL string*): The DateTime of when to start the assay window.
    * **interval** - (*OPTIONAL string*): The assay window interval.
  * **summarizer** - (*REQUIRED AssaySummerizer*): The summarizer type for the array aka "advanced settings" in the Wallaroo Dashboard UI.
    * **type** - (*REQUIRED string*): Type of summarizer.
    * **bin_mode** - (*REQUIRED string*): The binning model type.  Values can be:
      * Quantile
      * Equal
    * **aggregation** - (*REQUIRED string*): Aggregation type.
    * **metric** - (*REQUIRED string*): Metric type.  Values can be:
      * PSI
      * Maximum Difference of Bins
      * Sum of the Difference of Bins
    * **num_bins** - (*REQUIRED int*): The number of bins.  Recommanded values are between 5 and 14.
    * **bin_weights** - (*OPTIONAL AssayBinWeight*): The weights assigned to the assay bins.
    * **bin_width** - (*OPTIONAL AssayBinWidth*): The width assigned to the assay bins.
    * **provided_edges** - (*OPTIONAL AssayProvidedEdges*): The edges used for the assay bins.
    * **add_outlier_edges** - (*REQUIRED bool*): Indicates whether to add outlier edges or not.
  * **warning_threshold** - (*OPTIONAL number*): Optional warning threshold.
  * **alert_threshold** - (*REQUIRED number*): Alert threshold.
  * **run_until** - (*OPTIONAL string*): DateTime of when to end the assay.
  * **workspace_id** - (*REQUIRED integer*): The workspace the assay is part of.
  * **model_insights_url** - (*OPTIONAL string*): URL for model insights.
* **RETURNS**
  * {} when successful.

Example:  An interactive assay baseline will be set for the assay "Test Assay" on Pipeline 4.

```python
# Run interactive baseline

apiRequest = "/assays/run_interactive_baseline"

exampleAssayPipelineId = 4
exampleAssayPipelineName = "housepricepipe"
exampleAssayModelName = "housepricemodel"
exampleAssayWorkspaceId = 4
exampleAssayId = 3
exampleAssayName = "example assay"

data = {
    'id': exampleAssayId,
    'name': exampleAssayName,
    'pipeline_id': exampleAssayPipelineId,
    'pipeline_name': exampleAssayPipelineName,
    'active': True,
    'status': 'active',
    'iopath': "input 0 0",
    'baseline': {
        'Fixed': {
            'pipeline': exampleAssayPipelineName,
            'model': exampleAssayModelName,
            'start_at': '2022-01-01T00:00:00-05:00',
            'end_at': '2022-01-02T00:00:00-05:00'
        }
    },
    'window': {
        'pipeline': exampleAssayPipelineName,
        'model': exampleAssayModelName,
        'width': '24 hours',
        'start': None,
        'interval': None
    },
    'summarizer': {
        'type': 'UnivariateContinuous',
        'bin_mode': 'Quantile',
        'aggregation': 'Density',
        'metric': 'PSI',
        'num_bins': 5,
        'bin_weights': None,
        'bin_width': None,
        'provided_edges': None,
        'add_outlier_edges': True
    },
    'warning_threshold': 0,
    'alert_threshold': 0.1,
    'run_until': None,
    'workspace_id': exampleAssayWorkspaceId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    {'assay_id': 3,
     'name': 'example assay',
     'created_at': 1665428974654,
     'elapsed_millis': 3,
     'pipeline_id': 4,
     'pipeline_name': 'housepricepipe',
     'iopath': 'input 0 0',
     'baseline_summary': {'count': 1812,
      'min': -3.6163810435665233,
      'max': 7.112734553641073,
      'mean': 0.03518936967736047,
      'median': -0.39764636440424433,
      'std': 0.9885006118746916,
      'edges': [-3.6163810435665233,
       -0.39764636440424433,
       -0.39764636440424433,
       0.6752651953165153,
       0.6752651953165153,
       7.112734553641073,
       None],
      'edge_names': ['left_outlier',
       'q_20',
       'q_40',
       'q_60',
       'q_80',
       'q_100',
       'right_outlier'],
      'aggregated_values': [0.0,
       0.5739514348785872,
       0.0,
       0.3383002207505519,
       0.0,
       0.08774834437086093,
       0.0],
      'aggregation': 'Density',
      'start': '2022-01-01T05:00:00Z',
      'end': '2022-01-02T05:00:00Z'},
     'window_summary': {'count': 1812,
      'min': -3.6163810435665233,
      'max': 7.112734553641073,
      'mean': 0.03518936967736047,
      'median': -0.39764636440424433,
      'std': 0.9885006118746916,
      'edges': [-3.6163810435665233,
       -0.39764636440424433,
       -0.39764636440424433,
       0.6752651953165153,
       0.6752651953165153,
       7.112734553641073,
       None],
      'edge_names': ['left_outlier',
       'e_-3.98e-1',
       'e_-3.98e-1',
       'e_6.75e-1',
       'e_6.75e-1',
       'e_7.11e0',
       'right_outlier'],
      'aggregated_values': [0.0,
       0.5739514348785872,
       0.0,
       0.3383002207505519,
       0.0,
       0.08774834437086093,
       0.0],
      'aggregation': 'Density',
      'start': '2022-01-01T05:00:00Z',
      'end': '2022-01-02T05:00:00Z'},
     'warning_threshold': 0.0,
     'alert_threshold': 0.1,
     'score': 0.0,
     'scores': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     'index': None,
     'summarizer_meta': '{"type":"UnivariateContinuous","bin_mode":"Quantile","aggregation":"Density","metric":"PSI","num_bins":5,"bin_weights":null,"provided_edges":null}',
     'status': 'BaselineRun'}

### Get Assay Baseline

Retrieve an assay baseline.

* **Parameters**
  * **workspace_id** - (*REQUIRED integer*): Numerical id for the workspace the assay is in.
  * **pipeline_name** - (*REQUIRED string*): Name of the pipeline the assay is in.
  * **start** - (*OPTIONAL string*): DateTime for when the baseline starts.
  * **end** - (*OPTIONAL string*): DateTime for when the baseline ends.
  * **model_name** - (*OPTIONAL string*): Name of the model.
  * **limit** - (*OPTIONAL integer*): Maximum number of baselines to return.
* **Returns**
  * Assay Baseline
  
Example:  3 assay baselines for Workspace 6 and pipeline `houseprice-pipe-yns` will be retrieved.

```python
# Get Assay Baseline

apiRequest = "/assays/get_baseline"

data = {
    'workspace_id': exampleAssayWorkspaceId,
    'pipeline_name': exampleAssayPipelineName,
    'limit': 3
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    [{'check_failures': [],
      'elapsed': 138,
      'model_name': 'housepricemodel',
      'model_version': 'test_version',
      'original_data': {'tensor': [[0.6752651953165153,
         0.4999342471069234,
         -0.1508359058521761,
         0.20024994573167013,
         -0.08666382440547035,
         0.009116407905326388,
         -0.002872821251696453,
         -0.9179715198382244,
         -0.305653139057544,
         2.4393894526979074,
         0.29288456205300767,
         -0.3485179782510063,
         1.1121054807107582,
         0.20193559456886756,
         -0.20817781526102327,
         1.0279052268485522,
         -0.0196096612880121]]},
      'outputs': [{'Float': {'data': [13.262725830078123],
         'dim': [1, 1],
         'v': 1}}],
      'pipeline_name': 'housepricepipe',
      'time': 1643673456974},
     {'check_failures': [],
      'elapsed': 136,
      'model_name': 'housepricemodel',
      'model_version': 'test_version',
      'original_data': {'tensor': [[-0.39764636440424433,
         -1.4463372267359147,
         0.044822346031326635,
         -0.4259897870655369,
         -0.08666382440547035,
         -0.009153974747246364,
         -0.2568455220872559,
         0.005746226275241667,
         -0.305653139057544,
         -0.6285378875598833,
         -0.5584151415472702,
         -0.9748223338538442,
         -0.65605032361317,
         -1.5328599554165074,
         -0.20817781526102327,
         0.06504981348446033,
         -0.20382525042318508]]},
      'outputs': [{'Float': {'data': [12.82761001586914], 'dim': [1, 1], 'v': 1}}],
      'pipeline_name': 'housepricepipe',
      'time': 1643673504654},
     {'check_failures': [],
      'elapsed': 93,
      'model_name': 'housepricemodel',
      'model_version': 'test_version',
      'original_data': {'tensor': [[-1.470557924125004,
         -0.4732014898144956,
         1.0989221532266944,
         1.3317512811267456,
         -0.08666382440547035,
         0.006116141374609494,
         -0.21472817109954076,
         -0.9179715198382244,
         -0.305653139057544,
         -0.6285378875598833,
         0.29288456205300767,
         -0.14376463122700162,
         -0.65605032361317,
         1.1203567680905366,
         -0.20817781526102327,
         0.2692918708647222,
         -0.23870674508328787]]},
      'outputs': [{'Float': {'data': [13.03465175628662], 'dim': [1, 1], 'v': 1}}],
      'pipeline_name': 'housepricepipe',
      'time': 1643673552333}]

### Run Assay Interactively

Runs an assay.

* **Parameters**
  * **id** - (*REQUIRED int*):  The numerical identifier for the assay.
  * **name** - (*REQUIRED string*): The name of the assay.
  * **pipeline_id** - (*REQUIRED int*): The numerical idenfifier the assay will be placed into.
  * **pipeline_name** - (*REQUIRED string*): The name of the pipeline
  * **active** - (*REQUIRED bool*): Indicates whether the assay will be active upon creation or not.
  * **status** - (*REQUIRED string*): The status of the assay upon creation.
  * **iopath** - (*REQUIRED string*): The iopath of the assay.
  * **baseline** - (*REQUIRED baseline*): The baseline for the assay.
    * **Fixed** - (*REQUIRED AssayFixConfiguration*): The fixed configuration for the assay.
      * **pipeline** - (*REQUIRED string*): The name of the pipeline with the baseline data.
      * **model** - (*REQUIRED string*): The name of the model used.
      * **start_at** - (*REQUIRED string*): The DateTime of the baseline start date.
      * **end_at** - (*REQUIRED string*): The DateTime of the baseline end date.
  * **window** (*REQUIRED AssayWindow*): Assay window.
    * **pipeline** - (*REQUIRED string*): The name of the pipeline for the assay window.
    * **model** - (*REQUIRED string*): The name of the model used for the assay window.
    * **width** - (*REQUIRED string*): The width of the assay window.
    * **start** - (*OPTIONAL string*): The DateTime of when to start the assay window.
    * **interval** - (*OPTIONAL string*): The assay window interval.
  * **summarizer** - (*REQUIRED AssaySummerizer*): The summarizer type for the array aka "advanced settings" in the Wallaroo Dashboard UI.
    * **type** - (*REQUIRED string*): Type of summarizer.
    * **bin_mode** - (*REQUIRED string*): The binning model type.  Values can be:
      * Quantile
      * Equal
    * **aggregation** - (*REQUIRED string*): Aggregation type.
    * **metric** - (*REQUIRED string*): Metric type.  Values can be:
      * PSI
      * Maximum Difference of Bins
      * Sum of the Difference of Bins
    * **num_bins** - (*REQUIRED int*): The number of bins.  Recommanded values are between 5 and 14.
    * **bin_weights** - (*OPTIONAL AssayBinWeight*): The weights assigned to the assay bins.
    * **bin_width** - (*OPTIONAL AssayBinWidth*): The width assigned to the assay bins.
    * **provided_edges** - (*OPTIONAL AssayProvidedEdges*): The edges used for the assay bins.
    * **add_outlier_edges** - (*REQUIRED bool*): Indicates whether to add outlier edges or not.
  * **warning_threshold** - (*OPTIONAL number*): Optional warning threshold.
  * **alert_threshold** - (*REQUIRED number*): Alert threshold.
  * **run_until** - (*OPTIONAL string*): DateTime of when to end the assay.
  * **workspace_id** - (*REQUIRED integer*): The workspace the assay is part of.
  * **model_insights_url** - (*OPTIONAL string*): URL for model insights.
* **Returns**
  * Assay
  
Example:  An interactive assay will be run for Assay exampleAssayId exampleAssayName.  Depending on the number of assay results and the data window, this may take some time.  This returns *all* of the results for this assay at this time.  The total number of responses will be displayed after.

```python
# Run interactive assay

apiRequest = "/assays/run_interactive"

data = {
    'id': exampleAssayId,
    'name': exampleAssayName,
    'pipeline_id': exampleAssayPipelineId,
    'pipeline_name': exampleAssayPipelineName,
    'active': True,
    'status': 'active',
    'iopath': "input 0 0",
    'baseline': {
        'Fixed': {
            'pipeline': exampleAssayPipelineName,
            'model': exampleAssayModelName,
            'start_at': '2022-01-01T00:00:00-05:00',
            'end_at': '2022-01-02T00:00:00-05:00'
        }
    },
    'window': {
        'pipeline': exampleAssayPipelineName,
        'model': exampleAssayModelName,
        'width': '24 hours',
        'start': None,
        'interval': None
    },
    'summarizer': {
        'type': 'UnivariateContinuous',
        'bin_mode': 'Quantile',
        'aggregation': 'Density',
        'metric': 'PSI',
        'num_bins': 5,
        'bin_weights': None,
        'bin_width': None,
        'provided_edges': None,
        'add_outlier_edges': True
    },
    'warning_threshold': 0,
    'alert_threshold': 0.1,
    'run_until': None,
    'workspace_id': exampleAssayWorkspaceId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response[0]
```

    {'assay_id': 3,
     'name': 'example assay',
     'created_at': 1665429281268,
     'elapsed_millis': 178,
     'pipeline_id': 4,
     'pipeline_name': 'housepricepipe',
     'iopath': 'input 0 0',
     'baseline_summary': {'count': 1812,
      'min': -3.6163810435665233,
      'max': 7.112734553641073,
      'mean': 0.03518936967736047,
      'median': -0.39764636440424433,
      'std': 0.9885006118746916,
      'edges': [-3.6163810435665233,
       -0.39764636440424433,
       -0.39764636440424433,
       0.6752651953165153,
       0.6752651953165153,
       7.112734553641073,
       None],
      'edge_names': ['left_outlier',
       'q_20',
       'q_40',
       'q_60',
       'q_80',
       'q_100',
       'right_outlier'],
      'aggregated_values': [0.0,
       0.5739514348785872,
       0.0,
       0.3383002207505519,
       0.0,
       0.08774834437086093,
       0.0],
      'aggregation': 'Density',
      'start': '2022-01-01T05:00:00Z',
      'end': '2022-01-02T05:00:00Z'},
     'window_summary': {'count': 1812,
      'min': -3.6163810435665233,
      'max': 3.8939998744787943,
      'mean': 0.006175756859303479,
      'median': -0.39764636440424433,
      'std': 0.9720429128755866,
      'edges': [-3.6163810435665233,
       -0.39764636440424433,
       -0.39764636440424433,
       0.6752651953165153,
       0.6752651953165153,
       7.112734553641073,
       None],
      'edge_names': ['left_outlier',
       'e_-3.98e-1',
       'e_-3.98e-1',
       'e_6.75e-1',
       'e_6.75e-1',
       'e_7.11e0',
       'right_outlier'],
      'aggregated_values': [0.0,
       0.5883002207505519,
       0.0,
       0.3162251655629139,
       0.0,
       0.09547461368653422,
       0.0],
      'aggregation': 'Density',
      'start': '2022-01-02T05:00:00Z',
      'end': '2022-01-03T05:00:00Z'},
     'warning_threshold': 0.0,
     'alert_threshold': 0.1,
     'score': 0.002495916218595029,
     'scores': [0.0,
      0.0003543090106786176,
      0.0,
      0.0014896074883327124,
      0.0,
      0.0006519997195836994,
      0.0],
     'index': None,
     'summarizer_meta': {'type': 'UnivariateContinuous',
      'bin_mode': 'Quantile',
      'aggregation': 'Density',
      'metric': 'PSI',
      'num_bins': 5,
      'bin_weights': None,
      'provided_edges': None},
     'status': 'Warning'}

```python
print(len(response))
```

    30

### Get Assay Results

Retrieve the results for an assay.

* **Parameters**
  * **assay_id** - (*REQUIRED integer*): Numerical id for the assay.
  * **start** - (*OPTIONAL string*): DateTime for when the baseline starts.
  * **end** - (*OPTIONAL string*): DateTime for when the baseline ends.
  * **limit** - (*OPTIONAL integer*): Maximum number of results to return.
  * **pipeline_id** - (*OPTIONAL integer*): Numerical id of the pipeline the assay is in.
* **Returns**
  * Assay Baseline
  
Example:  Results for Assay 3 "example assay" will be retrieved for January 2 to January 3.  For the sake of time, only the first record will be displayed.

```python
# Get Assay Results

apiRequest = "/assays/get_results"

data = {
    'assay_id': exampleAssayId,
    'pipeline_id': exampleAssayPipelineId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```
