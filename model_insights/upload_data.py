import sys
import pandas as pd
import numpy as np
import datetime
import pytz
import json
import joblib
import random

import requests


def rfc3339(d: datetime.datetime, verbose=False) -> str:
    return d.astimezone(tz=datetime.timezone.utc).isoformat()


def upload_data(
    logs,
    pipeline="mypipeline",
    model="mymodel",
    start="2022-01-01T00:00:00+00:00",
    end="2022-02-01T00:00:00+00:00",
    topic="default-topic",
):
    partition = "part-1"
    url = f"http://plateau:3030/topic/{topic}/partition/{partition}"
    print(f"Url: {url}")

    start_date = datetime.datetime.fromisoformat(start)
    end_date = datetime.datetime.fromisoformat(end)

    span = end_date - start_date
    span_offset = span / len(logs)

    # Write our logs to a partition in plateau
    i = 0
    ts = start_date
    for i, r in enumerate(logs):
        # ts = datetime.datetime.utcfromtimestamp(int(r["time"] / 1000))
        ts = start_date + (i * span_offset)
        r["time"] = int(ts.timestamp() * 1000)

        params = {"time": rfc3339(ts)}
        if i == 0:
            print("PARAMS", r["time"], params)

        r["pipeline_name"] = pipeline
        r["model_name"] = model
        r["elapsed"] = int(
            random.gauss(100, 30) if random.random() < 0.5 else random.gauss(250, 60)
        )
        resp = requests.post(url, params=params, json={"records": [json.dumps(r)]})
        if resp.status_code != 200:
            raise Exception(f"Could not post log {params} ", resp.text)
        if i % 10_000 == 0:
            print(f"{i} {ts}, ", end="", flush=True)

    print(f"\nFinal: {i} {ts}")

    return logs


if __name__ == "__main__":
    uploaded_logs = upload_data()
    num_uploaded_logs = len(uploaded_logs)
    print(f"\n Uploaded {num_uploaded_logs} canned logs")
