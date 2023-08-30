import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd

wl = wallaroo.Client()

# get the arguments
arguments = wl.task_args()

if "workspace_name" in arguments:
    workspace_name = arguments['workspace_name']
else:
    workspace_name="forecast-model-workshop"

if "pipeline_name" in arguments:
    pipeline_name = arguments['pipeline_name']
else:
    pipeline_name="bikedaypipe"

def get_workspace(name):
    workspace = None
    for ws in wl.list_workspaces():
        if ws.name() == name:
            workspace= ws
    return workspace

def get_pipeline(name):
    try:
        pipeline = wl.pipelines_by_name(name)[0]
    except EntityNotFoundError:
        print(f"Pipeline not found:{name}")
    return pipeline

def get_singleton_forecast(df, field):
    singleton = pd.DataFrame({field: [df[field].values.tolist()]})
    return singleton

print(f"Workspace: {workspace_name}")
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
print(workspace)

# the pipeline is assumed to be deployed
print(f"Pipeline: {pipeline_name}")
pipeline = get_pipeline(pipeline_name)
print(pipeline)

print(pipeline.status())

sample_count = pd.read_csv('./data/test_data.csv')
inference_df = get_singleton_forecast(sample_count.loc[2:22], 'count')

results = pipeline.infer(inference_df)

print(results)
