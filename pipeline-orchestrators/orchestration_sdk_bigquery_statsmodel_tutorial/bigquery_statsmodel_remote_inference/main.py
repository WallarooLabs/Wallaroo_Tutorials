import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd
import os
#Used for the Wallaroo SDK version 2023.1
os.environ["ARROW_ENABLED"]="True"

# bigquery library
from google.cloud import bigquery
from google.oauth2 import service_account
import db_dtypes
import json

wl = wallaroo.Client()
# get the arguments

if wl.in_task():
    arguments = wl.task_args()
    print(wl.task_args())

    if "workspace_name" in arguments:
        workspace_name = arguments['workspace_name']
    else:
        workspace_name="bigquerystatsmodelworkspace"

    if "pipeline_name" in arguments:
        pipeline_name = arguments['pipeline_name']
    else:
        pipeline_name="bigquerystatsmodelpipeline"

    if "bigquery_connection_input_name" in arguments:
        bigquery_connection_input_name = arguments['bigquery_connection_input_name']
    else:
        bigquery_connection_input_name = "bigqueryforecastinputs"

    if "bigquery_connection_output_name" in arguments:
        bigquery_connection_output_name = arguments['bigquery_connection_output_name']
    else:
        bigquery_connection_output_name = "bigqueryforecastoutputs"
else:
    # we're not in the task, so use the default values
    workspace_name = 'bigquerystatsmodelworkspace'
    pipeline_name = 'bigquerystatsmodelpipeline'

    bigquery_connection_input_name = "bigqueryforecastinputs"
    
    bigquery_connection_output_name = "bigqueryforecastoutputs"
    
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
print(wl.get_current_workspace())

pipeline = get_pipeline(pipeline_name)

# deploy the pipeline
print("\nDeploying pipeline.")
print(pipeline_name)
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

inference_dataframe_input = bigqueryinputclient.query(
        f"""
        (select dteday, temp, holiday, workingday, windspeed
        FROM {big_query_input_connection.details()['dataset']}.{big_query_input_connection.details()['table']}
        ORDER BY dteday DESC LIMIT 7)
        ORDER BY dteday
        """
    ).to_dataframe().drop(columns=['dteday'])

# convert to a dict, show the first 7 rows
print(f"\n{inference_dataframe_input.to_dict()}")

# perform the inference and display the result
results = pipeline.infer(inference_dataframe_input.to_dict())
print(f"\n{results[0]['forecast']}")

# Get the output table, then upload the inference results
output_table = bigqueryoutputclient.get_table(f"{big_query_output_connection.details()['dataset']}.{big_query_output_connection.details()['table']}")

job = bigqueryoutputclient.query(
        f"""
        INSERT {big_query_output_connection.details()['dataset']}.{big_query_output_connection.details()['table']}
        VALUES
        (current_timestamp(), "{results[0]['forecast']}")
        """
    )

# Show the last 5 output inserts 
# Get the last insert to the output table to verify

task_inference_results = bigqueryoutputclient.query(
        f"""
        SELECT *
        FROM {big_query_output_connection.details()['dataset']}.{big_query_output_connection.details()['table']}
        ORDER BY date DESC
        LIMIT 5
        """
    ).to_dataframe()

print(f"\n{task_inference_results}")

# close the bigquery clients
bigqueryinputclient.close()
bigqueryoutputclient.close()

# deploy the pipeline
print("\nUndeploying pipeline.")
pipeline.undeploy()
