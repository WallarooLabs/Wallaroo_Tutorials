print("starting", flush=True)

import wallaroo
import pandas as pd
import numpy as np

import datetime

#
# usual convenience functions. 
# I really should have these in util, 
# but this makes cut-n-paste from notebook safer
#

def get_workspace(name):
    workspace = None
    for ws in wl.list_workspaces():
        if ws.name() == name:
            workspace= ws
    if(workspace == None):
       raise KeyError(f"workspace {name} not found.")
    return workspace


# get a pipeline by name in the workspace
# assumes that whoever is running this orchestration 
# has access to the workspace
def get_pipeline(pname, workspace, create_if_absent=False):
    plist = workspace.pipelines()
    pipeline = [p for p in plist if p.name() == pname]
    if len(pipeline) <= 0:
        if create_if_absent:
            pipeline = wl.build_pipeline(pname)
        else:
            raise KeyError(f"pipeline {pname} not found in workspace {workspace.name()}")
    else:
        pipeline = pipeline[0]
    return pipeline


def is_deployed(pipeline_name):
    try:
        return wl.deployment_by_name(pipeline_name).deployed()
    except:
        return False 




    
#
# function to do the task. Assumes connection to wallaroo is "wl"
#

def do_task(arguments): 
    
    print("starting task", flush=True)
    
  
    if 'llm_workspace' in arguments:
        llm_workspace = arguments['llm_workspace']
    else:
        llm_workspace = default_args['llm_workspace']
        
    if 'llm_pipeline' in arguments:
        llm_pipeline = arguments['llm_pipeline']
    else:
        llm_pipeline = default_args['llm_pipeline']
        
    if 'llm_output_field' in arguments:
        llm_output_field = arguments['llm_output_field']
    else:
        llm_output_field = default_args['llm_output_field']
        
    if 'monitor_workspace' in arguments:
        monitor_workspace = arguments['monitor_workspace']
    else:
        monitor_workspace = default_args['monitor_workspace']
        
    if 'monitor_pipeline' in arguments:
        monitor_pipeline = arguments['monitor_pipeline']
    else:
        monitor_pipeline = default_args['monitor_pipeline']    
        
    if 'window_length' in arguments:
        window_length = arguments['window_length']
    else:
        window_length = default_args['window_length']
        
    if 'n_toxlabels' in arguments:
        n_toxlabels = arguments['n_toxlabels']
    else:
        n_toxlabels = default_args['n_toxlabels']
                                                
    
    # retrieve logs from llm
    llm = get_pipeline(llm_pipeline, get_workspace(llm_workspace))
    if window_length == -1:
        llm_logs = llm.logs()
    else:
        print(f"getting logs for the last {window_length} hours")
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(hours=window_length)
        print(f"time interval from {start_time} to {end_time}")
        llm_logs = llm.logs(start_datetime=start_time, end_datetime=end_time) 
    
    
    # check whether there are any logs to process
    if(llm_logs.shape[0] > 0):

        # go to monitor workspace                                    
        workspace = get_workspace(monitor_workspace)
        _ = wl.set_current_workspace(workspace)
        toxmonitor = get_pipeline(monitor_pipeline, workspace)


        if not is_deployed(toxmonitor.name()):
            print("Deploying pipeline.", flush=True)
            
            # Get the model_version from the pipeline itself, rather than passing it into the orch.
            # In theory, toxmonitor.model_configs()[0].model_version() should do the same thing
            # but the list is empty on a freshly fetched (undeployed) pipeline.
            # This also assumes the model itself is the first step -- no preprocessing before it.
            
            pconfig = toxmonitor.get_pipeline_configuration()
            model_desc = pconfig['definition']['steps'][0]
            mname, mver = (model_desc['ModelInference']['models'][0]['name'], model_desc['ModelInference']['models'][0]['version'])
            monitor_model = wl.model_by_name(mname, mver)
            
            deployment_config = wallaroo.DeploymentConfigBuilder() \
                .cpus(0.25).memory('1Gi') \
                .sidekick_cpus(monitor_model, 4) \
                .sidekick_memory(monitor_model, "8Gi") \
                .build()

            toxmonitor.deploy(deployment_config=deployment_config)


        # create the input for the toxicity model
        input_data = {
                "inputs": llm_logs[llm_output_field], 
        }
        dataframe = pd.DataFrame(input_data)
        dataframe['top_k'] = n_toxlabels                    

        toxresults = toxmonitor.infer(dataframe)
        print(toxresults)
        # this is mostly for demo purposes
        print("Avg Batch Toxicity:", np.mean(toxresults['out.toxic'].apply(lambda x:x[0])))
        print("Over Threshold:", sum(toxresults['out.toxic'].apply(lambda x:x[0]) > 0.001))
        toxmonitor.undeploy()
    else:                              # if(llm_logs.shape[0] <= 0)
        print("No logs to process.")
        
    
                                                                             
    print("Task complete", flush=True)
    
    
    
#
# set default arguments
# these are hardcoded for testing
# I will also use them as fallbacks if not all args are supplied to task
#

default_args = {
    'llm_workspace' : 'llm-monitoring' ,
    'llm_pipeline': 'summarizer-pipe',
    'llm_output_field': 'out.generated_text',
    'monitor_workspace': 'llm-monitoring',
    'monitor_pipeline' : 'full-toxmonitor-pipeline',
    'window_length': -1,  # in hours. If -1, no limit (for testing)
    'n_toxlabels': 6,
}

#
# now actually do something
#


wl = wallaroo.Client(request_timeout=120)

if wl.in_task():              # if running the orchestrated task in wallaroo
    do_task(wl.task_args())
else:                         # if testing main.py from the command line or console
    do_task(default_args)

