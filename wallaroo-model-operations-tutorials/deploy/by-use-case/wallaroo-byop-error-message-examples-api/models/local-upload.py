import time
import wallaroo
import requests
import json
import pyarrow as pa
from wallaroo.framework import Framework
import base64
import os

wl=wallaroo.Client(api_endpoint='https://autoscale-uat-gcp.wallaroo.dev', auth_type='sso')

headers = wl.auth.auth_header()
url = f"{wl.api_endpoint}/v1/api/models/upload_and_convert"
workspace_id = wl.get_current_workspace().id()
framework = 'custom'
model_name = 'cleaned-up-inf-no-exp2-jch-vscodeless'
filename = 'byop-llamacpp-inf-no-exp-model-types.zip'

input_schema, output_schema = (
        pa.schema([pa.field("inputs", pa.list_(pa.float32(), list_size=10))]),
        pa.schema([pa.field("predictions", pa.float32())]),
)

metadata = {
    "name": model_name,
    "visibility": "public",
    "workspace_id": workspace_id,
    "conversion": {
        "framework": framework,
        "python_version": "3.8",
        "requirements": [],
    },
    "input_schema": base64.b64encode(bytes(input_schema.serialize())).decode("utf8"),
    "output_schema": base64.b64encode(bytes(output_schema.serialize())).decode("utf8"),
}
  
files = {
    'metadata': (None, json.dumps(metadata), "application/json"),
    'file': (filename, open(filename, 'rb'), 'application/octet-stream')
}
response = requests.post(url, headers=headers, files=files)
print(response.json())


url = wl.api_endpoint + '/v1/api/models/get_version_by_id'
data = {
  "model_version_id": response.json()['insert_models']['returning'][0]['models'][0]['id'],
}

mver = requests.post(url, headers=headers, json=data).json()
while mver['model_version']['model_version']['status'] not in ('error', 'ready'):
    time.sleep(4)
    print(".", end='')
    headers = wl.auth.auth_header()
    mver = requests.post(url, headers=headers, json=data).json()
print()
print(json.dumps(mver, indent=4))

if mver['model_version']['model_version']['status'] == 'error':
    time.sleep(1)  # Give DB a moment to fully propagate
    headers = wl.auth.auth_header()
    mver = requests.post(url, headers=headers, json=data).json()
