The following tutorial is available on the [Wallaroo Github Repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/20231004-wallaroo-inference-server/wallaroo-inference-server-tutorials/wallaroo-inference-server-hf-summarizer).

## Wallaroo Inference Server:  Hugging Face Summarizer

This notebook is used in conjunction with the [Wallaroo Inference Server Free Edition](https://docs.wallaroo.ai/wallaroo-inferencing-server/) for LLama 2.  This provides a free license for performing inferences through the Hugging Face Summarizer model.  For more information, see [the Llama 2 reference page](https://ai.meta.com/llama/).

### Prerequisites

* A deployed Wallaroo Inference Server Free Edition with one of the following options:
  * **Wallaroo.AI Llama Inference Server - GPU**
* Access via port 8080 to the Wallaroo Inference Server Free Edition.

## Llama 2 Model Schemas

### Inputs

The Llama 2 Model takes the following inputs.

| Field | Type | Description |
|---|---|---|
| `text` | String (*Required*) | The prompt for the llama model. |

### Outputs

| Field | Type | Description |
|---|---|---|
| `generated_text` | String | The generated text output. |

## Wallaroo Inference Server API Endpoints

The following HTTPS API endpoints are available for Wallaroo Inference Server.

### Pipelines Endpoint

* Endpoint: HTTPS GET `/pipelines`
* Returns:
  * List of `pipelines` with the following fields.
    * **id** (*String*): The name of the pipeline.
    * **status** (*String*): The pipeline status.  `Running` indicates the pipeline is available for inferences.

#### Pipeline Endpoint Example

The following demonstrates using `curl` to retrieve the Pipelines endpoint.  Replace the HOSTNAME with the address of your Wallaroo Inference Server.

```python
!curl mb-freemium-hfstd86.eastus.cloudapp.azure.com:8080/pipelines
```

    {"pipelines":[{"id":"hf-summarizer-standard","status":"Running"}]}

### Models Endpoint

* Endpoint: GET `/models`
* Returns:
  * List of `models` with the following fields.
    * **name** (*String*):  The name of the model.
    * **sha** (*String*):  The `sha` hash of the model.
    * **status** (*String*):  The model status.  `Running` indicates the models is available for inferences.
    * **version** (*String*): The model version in UUID format.

#### Models Endpoint Example

The following demonstrates using `curl` to retrieve the Models endpoint.  Replace the HOSTNAME with the address of your Wallaroo Inference Server.

```python
!curl doc-example-hf-summarizer.westus2.cloudapp.azure.com:8080/models
```

    {"models":[{"name":"hf-summarizer-standard","sha":"ee71d066a83708e7ca4a3c07caf33fdc528bb000039b6ca2ef77fa2428dc6268","status":"Running","version":"7dbae7b4-20d0-40f7-a3f5-eeabdd77f418"}]}

### Inference Endpoint

The following inference endpoint is available from the Wallaroo Server for HuggingFace Summarizer.

* Endpoint: HTTPS POST `/pipelines/hf-summarizer-standard`
* Headers:
  * `Content-Type: application/vnd.apache.arrow.file`: For Apache Arrow tables.
  * `Content-Type: application/json; format=pandas-records`: For pandas DataFrame in record format.
* Input Parameters: DataFrame in `/pipelines/hf-summarizer-standard` **OR** Apache Arrow table in `application/vnd.apache.arrow.file` with the following inputs:
  * **text** (*String* *Required*): The text prompt.
* Returns:
  * Headers
    * `Content-Type: application/json; format=pandas-records`: pandas DataFrame in record format.
  * Data
    * **check_failures** (*List[Integer]*): Whether any validation checks were triggered.  For more information, see [Wallaroo SDK Essentials Guide: Pipeline Management: Anomaly Testing]({{<ref "wallaroo-sdk-essentials-pipeline#anomaly-testing">}}).
    * **elapsed** (*List[Integer]*): A list of time in nanoseconds for:
    * [0] The time to serialize the input.
    * [1...n] How long each step took.
    * **model_name** (*String*): The name of the model used.
    * **model_version** (*String*): The version of the model in UUID format.
    * **original_data**: The original input data.  Returns `null` if the input may be too long for a proper return.
    * **outputs** (*List*): The outputs of the inference result separated by data type.
    * **String**: The string outputs for the inference.
      * **data** (*List[String]*): The generated text from the prompt.
        * **dim** (*List[Integer]*): The dimension shape returned, always returned as `[1,1]` for this model deployment.
        * **v** (*Integer*): The vector shape of the data, always returned as `1` for this mnodel deployment.
    * **pipeline_name**  (*String*): The name of the pipeline.
    * **shadow_data**: Any shadow deployed data inferences in the same format as **outputs**.
    * **time** (*Integer*): The time since UNIX epoch.

### Inference Endpoint Example

The following example performs an inference using the pandas record input `./data/test_summarization.df.json` with a text string to summarize.

```python
!curl -X POST doc-example-llama.westus2.cloudapp.azure.com:8080/pipelines/llama \
    -H "Content-Type: application/json; format=pandas-records" \
    -d @./data/llamaquery.df.json
```

    curl: (7) Failed to connect to doc-example-llama.westus2.cloudapp.azure.com port 8080 after 2080 ms: Couldn't connect to server

