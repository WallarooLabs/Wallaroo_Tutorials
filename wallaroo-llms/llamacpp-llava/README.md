# Llava 1.6 with Llama.cpp Python bindings

## Business Context

This BYOP model takes a text prompt and an image in an array format, and it will give an answer about what the question and the image were. It uses the latest version of a quantized model called `Llava 1.6` found on `HuggingFace`, which is loaded via `Llama.cpp`.

## Model Overview

- What is the framework(s) that the customer used in order to create this model? This uses a `GGUF` quantized version of the Llava 1.6 34B model, alongside `llama-cpp-python`.
- What are the model artifacts? There are two artifacts that are being used by the model, the actual model and the CLIP model for calculating the image embeddings, which can be found here:
  - [Llava 1.6 34B Quantized](https://huggingface.co/cjpais/llava-v1.6-34B-gguf/blob/main/llava-v1.6-34b.Q5_K_M.gguf)
  - [CLIP Model](https://huggingface.co/cjpais/llava-v1.6-34B-gguf/blob/main/mmproj-model-f16.gguf)
  
  Make sure to move the model artifact under the `byop/artifacts` directory before zipping and deploying the BYOP.
- What are the input/output data types? This `BYOP` model requires two inputs, the text prompt alongside an image that can have variable size.
- Expected output data for testing: The expected output for this model is just a generated text from the model.

## Baseline Requirements

### Hardware Requirements (Optional)

- What did the customer use when training/testing the model? N/A
- Are there any constraints in terms of the hardware that will be used? (due to costs maybe?) This should work on both CPU and GPU, although since we are talking about a model that is roughly ~24 GB in size, the acceptable response time is achieved on GPU only.

### Performance Metrics (Optional)

- Does the customer have different performance metrics that they are tracking for this particular model? N/A

## Implementation details

In this section, we will emphasize the interesting parts of the BYOP code that has been developed for this multimodal model:

1. In order to run [Llama-cpp-python](https://github.com/abetlen/llama-cpp-python) on GPU, some extra arguments need to be passed to the `pip install` command, which raises some issues with the way we parse requirements in BYOP. Because of this fact, we will install `llama-cpp-python` using the `subprocess` library in `python`, straight into the Python BYOP code:

``` python
import subprocess
import sys

pip_command = (
    f'CMAKE_ARGS="-DLLAMA_CUDA=on" {sys.executable} -m pip install llama-cpp-python'
)

subprocess.check_call(pip_command, shell=True)
```

2. Since we are using a multimodal model, images are transformed into embeddings using `CLIP`, which also needs to be loaded using a `ChatHandler` from `llama-cpp-python`.

```python
def _load_model(self, model_path):
    chat_handler = NanoLlavaChatHandler(
        clip_model_path=f"{model_path}/artifacts/mmproj-model-f16.gguf"
    )
    llm = Llama(
        model_path=f"{model_path}/artifacts/llava-v1.6-34b.Q5_K_M.gguf",
        chat_format="llava-1-6",
        chat_handler=chat_handler,
        n_ctx=4096,
        n_gpu_layers=-1,
        logits_all=True,
    )

    return llm
```

**Note:** There are multiple chat handlers that can be used with different `Llava` models, depending on their version. We have used `NanoLlavaChatHandler` instead of `Llava16ChatHandler` which should be the default for the type of model that we use because of some bugs that are found in `llama-cpp-python`.

3. From the `llama-cpp-python`'s documentation, we can see that the usual use-case for `Llava` models involves reading images from either local files, or URLs. In our case though, the images are passed as numpy arrays in the BYOP, so we had to find a way to convert a `numpy.ndarray` image to a text-encoded image. The code below which uses `opencv` and `base64` shows how we have achieved that:

```python
_, buffer = cv2.imencode(".png", np.array(image))
prefix = "data:image/png;base64,"
b64encoded_image = prefix + base64.b64encode(buffer).decode("utf-8")
```

4. The prompt had to go through a couple of development cycles in order to be stable enough for a decent inference:

```python
messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": b64encoded_image}},
            {
                "type": "text",
                "text": prompt,
            },
        ],
    },
]

result = self.model.create_chat_completion(
    messages=messages, max_tokens=256, stop=["<|im_end|>"]
)
```

### Notebook Relevant Details

An example notebook can be found in `notebooks/llava-llamacpp-deployment.ipynb`. The aforementioned pre-processing logic that segments the video into frames is provided in the notebook.

There are a couple of things that can be mentioned, as they are relevant to the running of this use-case:

- The model is uploaded via API, due to the sheer amount of memory.
- Due to some issues with how garbage collection works in `llama-cpp-python`, the deployment needs `30Gi` of memory, even though this model will be offloaded to the GPU that's available. The `DeploymentConfig` can be seen below:

```python
deployment_config = DeploymentConfigBuilder() \
    .cpus(1).memory('2Gi') \
    .sidekick_cpus(model, 8) \
    .sidekick_memory(model, "30Gi") \
    .sidekick_gpus(model, 1) \
    .build()
```

- The image that has been passed in to the model has been loaded using `PIL.Image`, and then converted to `numpy.ndarray`.

### How to run the BYOP?

1. Clone the repository
2. Create an `artifacts` folder inside `byop`
3. Download the artifacts from [here](https://huggingface.co/cjpais/llava-v1.6-34B-gguf/blob/main/llava-v1.6-34b.Q5_K_M.gguf) and [here](https://huggingface.co/cjpais/llava-v1.6-34B-gguf/blob/main/mmproj-model-f16.gguf) and move both files to the `artifacts` folder
