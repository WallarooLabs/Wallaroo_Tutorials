import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd

# bigquery library
from google.cloud import bigquery
from google.oauth2 import service_account
import db_dtypes

wl = wallaroo.Client()

# get the arguments
arguments = wl.task_args()

if "workspace_name" in arguments:
    workspace_name = arguments['workspace_name']
else:
    workspace_name="bigqueryworkspace"

if "pipeline_name" in arguments:
    pipeline_name = arguments['pipeline_name']
else:
    pipeline_name="bigquerypipeline"

if "bigquery_connection_input_name" in arguments:
    bigquery_connection_input_name = arguments['bigquery_connection_input_name']
else:
    bigquery_connection_input_name = "bigqueryhouseinputs"

if "bigquery_connection_output_name" in arguments:
    bigquery_connection_output_name = arguments['bigquery_connection_output_name']
else:
    bigquery_connection_output_name = "bigqueryhouseoutputs"

# helper methods to retrieve workspaces and pipelines

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

# set the workspace and pipeline
workspace = get_workspace(workspace_name)
wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)

# deploy the pipeline
pipeline.deploy()

# get the connections
big_query_input_connection = wl.get_connection(name=bigquery_connection_input_name)
big_query_output_connection = wl.get_connection(name=bigquery_connection_output_name)

# Set the bigquery input and output credentials
bigquery_input_credentials = service_account.Credentials.from_service_account_info(
    big_query_input_connection.details())

bigquery_output_credentials = service_account.Credentials.from_service_account_info(
    big_query_output_connection.details())

# start the input and output clients
bigqueryinputclient = bigquery.Client(
    credentials=bigquery_input_credentials, 
    project=big_query_input_connection.details()['project_id']
)
bigqueryoutputclient = bigquery.Client(
    credentials=bigquery_output_credentials, 
    project=big_query_output_connection.details()['project_id']
)

# query for the inference input data
inference_dataframe_input = bigqueryinputclient.query(
        f"""
        SELECT tensor
        FROM {big_query_input_connection.details()['dataset']}.{big_query_input_connection.details()['table']}"""
    ).to_dataframe()

# perform the inference and display the first 5 rows
result = pipeline.infer(inference_dataframe_input)
print(result.head(5))

# Get the output table, then upload the inference results
output_table = bigqueryoutputclient.get_table(f"{big_query_output_connection.details()['dataset']}.{big_query_output_connection.details()['table']}")

bigqueryoutputclient.insert_rows_from_dataframe(
    output_table, 
    dataframe=result.rename(columns={"in.tensor":"in_tensor", "out.variable":"out_variable"})
)

# close the bigquery clients
bigqueryinputclient.close()
bigqueryoutputclient.close()

# deploy the pipeline
pipeline.undeploy()