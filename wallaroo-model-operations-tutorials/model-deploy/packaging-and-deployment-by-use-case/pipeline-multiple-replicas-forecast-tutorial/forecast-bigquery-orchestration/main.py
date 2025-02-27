import json
import os
import datetime
import asyncio


import wallaroo
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework

import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', None)

# for Big Query connections
from google.cloud import bigquery
from google.oauth2 import service_account
import db_dtypes

import time

async def main():
    
    wl = wallaroo.Client()

    # get the arguments
    arguments = wl.task_args()

    if "workspace_name" in arguments:
        workspace_name = arguments['workspace_name']
    else:
        workspace_name="multiple-replica-forecast-tutorial"

    if "pipeline_name" in arguments:
        pipeline_name = arguments['pipeline_name']
    else:
        pipeline_name="bikedaypipe"

    if "bigquery_connection_input_name" in arguments:
        bigquery_connection_name = arguments['bigquery_connection_input_name']
    else:
        bigquery_connection_name = "statsmodel-bike-rentals"

    print(bigquery_connection_name)
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

    print(f"BigQuery Connection: {bigquery_connection_name}")
    forecast_connection = wl.get_connection(bigquery_connection_name)




    print(f"Workspace: {workspace_name}")
    workspace = get_workspace(workspace_name)

    wl.set_current_workspace(workspace)
    print(workspace)

    # the pipeline is assumed to be deployed
    print(f"Pipeline: {pipeline_name}")
    pipeline = get_pipeline(pipeline_name)
    print(pipeline)

    print("Getting date and input query.")

    bigquery_statsmodel_credentials = service_account.Credentials.from_service_account_info(
        forecast_connection.details())

    bigquery_statsmodel_client = bigquery.Client(
        credentials=bigquery_statsmodel_credentials, 
        project=forecast_connection.details()['project_id']
    )



    print("Get the current month and retrieve next month's forecasts")
    month = datetime.datetime.now().month
    start_date = f"{month+1}-1-2011"
    print(f"Start date: {start_date}")

    def get_forecast_days(firstdate) :
        days = [i*7 for i in [-1,0,1,2,3,4]]
        deltadays = pd.to_timedelta(pd.Series(days), unit='D') 

        analysis_days = (pd.to_datetime(firstdate) + deltadays).dt.date
        analysis_days = [str(day) for day in analysis_days]
        analysis_days
        seed_day = analysis_days.pop(0)

        return analysis_days

    forecast_dates = get_forecast_days(start_date)
    print(f"Forecast dates: {forecast_dates}")

    # get our list of items to run through

    inference_data = []
    days = []

    # get the days from the start date to the end date
    def get_forecast_dates(forecast_day: str, nforecast=7):
        days = [i for i in range(nforecast)]
        deltadays = pd.to_timedelta(pd.Series(days), unit='D')

        last_day = pd.to_datetime(forecast_day)
        dates = last_day + deltadays
        datestr = dates.dt.date.astype(str)
        return datestr 

    # used to generate our queries
    def mk_dt_range_query(*, tablename: str, forecast_day: str) -> str:
        assert isinstance(tablename, str)
        assert isinstance(forecast_day, str)
        query = f"""
                select cnt from {tablename} where 
                dteday >= DATE_SUB(DATE('{forecast_day}'), INTERVAL 1 month) 
                AND dteday < DATE('{forecast_day}') 
                ORDER BY dteday
                """
        return query

    for day in forecast_dates:
        print(f"Current date: {day}")
        day_range=get_forecast_dates(day)
        days.append({"date": day_range})
        query = mk_dt_range_query(tablename=f"{forecast_connection.details()['dataset']}.{forecast_connection.details()['input_table']}", forecast_day=day)
        print(query)
        data = bigquery_statsmodel_client.query(query).to_dataframe().apply({"cnt":int}).to_dict(orient='list')
        # add the date into the list
        inference_data.append(data)

    print(inference_data)

    parallel_results = await pipeline.parallel_infer(tensor_list=inference_data, timeout=20, num_parallel=16, retries=2)

    days_results = list(zip(days, parallel_results))
    print(days_results)

    # merge our parallel results into the predicted date sales
    results_table = pd.DataFrame(columns=["date", "forecast"])

    # match the dates to predictions
    # display(days_results)
    for date in days_results:
        # display(date)
        new_days = date[0]['date'].tolist()
        new_forecast = date[1][0]['forecast']
        new_results = list(zip(new_days, new_forecast))
        results_table = results_table.append(pd.DataFrame(list(zip(new_days, new_forecast)), columns=['date','forecast']))

    print("Uploading results to results table.")
    output_table = bigquery_statsmodel_client.get_table(f"{forecast_connection.details()['dataset']}.{forecast_connection.details()['results_table']}")

    bigquery_statsmodel_client.insert_rows_from_dataframe(
        output_table, 
        dataframe=results_table
    )
    
asyncio.run(main())