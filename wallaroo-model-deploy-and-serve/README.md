## Model Conversion Tutorials

Wallaroo makes it easy to upload your models and deploy them into a pipeline.  The following tutorials show how to upload models of various flavors (ONNX, SKLearn, etc) into Wallaroo and deploy them for test inferences.

Each model flavor other than ONNX requires the following:

* The model file.
* The model framework from the `wallaroo.framework` library.  The following frameworks are supported:

    ```python
    from wallaroo.framework import Framework
    [e.value for e in Framework]

    ['onnx',
    'tensorflow',
    'python',
    'keras',
    'sklearn',
    'pytorch',
    'xgboost',
    'hugging-face-feature-extraction',
    'hugging-face-image-classification',
    'hugging-face-image-segmentation',
    'hugging-face-image-to-text',
    'hugging-face-object-detection',
    'hugging-face-question-answering',
    'hugging-face-stable-diffusion-text-2-img',
    'hugging-face-summarization',
    'hugging-face-text-classification',
    'hugging-face-translation',
    'hugging-face-zero-shot-classification',
    'hugging-face-zero-shot-image-classification',
    'hugging-face-zero-shot-object-detection',
    'hugging-face-sentiment-analysis',
    'hugging-face-text-generation']
    ```

* input_schema: The input schema from the Apache Arrow pyarrow.lib.Schema format.
* output_schema: The output schema from the Apache Arrow pyarrow.lib.Schema format.

For example:

```python
input_schema = pa.schema([
    pa.field('sepal length (cm)', pa.float64()),
    pa.field('sepal width (cm)', pa.float64()),
    pa.field('petal length (cm)', pa.float64()),
    pa.field('petal width (cm)', pa.float64())
])

output_schema = pa.schema([
    pa.field('output', pa.float64())
])

model = wl.upload_model(
    "sklearn-kmeans", 
    'models/model-auto-conversion_sklearn_kmeans.pkl', 
    framework=Framework.SKLEARN, 
    input_schema=input_schema, 
    output_schema=output_schema)
```

Each tutorial includes:

* A sample model of a particular flavor.
* Sample data for that model.
* Example of input and output data formats.
* Deploying a pipeline with that model as a model step.
* Performing a sample inference.
