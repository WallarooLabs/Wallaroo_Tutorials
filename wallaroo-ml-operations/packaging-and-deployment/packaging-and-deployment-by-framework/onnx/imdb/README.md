# Linear Model Chaining Example

Sentiment analysis on IMDB data. Internally developed models.

Two models: The first model creates a text embedding from pre-tokenized text documents (model input: 100 integers/datum; output 800 numbers/datum). The second model classifies the resulting embeddings (output 0/1; 1 = positive review).

## Models:

* **embedder.onnx**: Embedder model, in zipped TF format, and converted to onnx
* **sentiment_model.onnx**: Sentiment model, in zipped TF format and converted to onnx
