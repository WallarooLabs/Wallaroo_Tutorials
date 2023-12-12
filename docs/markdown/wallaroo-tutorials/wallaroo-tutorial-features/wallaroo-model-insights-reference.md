This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/model_insights).

The Model Insights feature lets you monitor how the environment that your model operates within may be changing in ways that affect it's predictions so that you can intervene (retrain) in an efficient and timely manner. Changes in the inputs, **data drift**, can occur due to errors in the data processing pipeline or due to changes in the environment such as user preference or behavior. 

The [validation framework](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipeline/#anomaly-testing) performs per inference range checks with count frequency based thresholds for alerts and is ideal for catching many errors in input and output data.

In complement to the validation framework model insights focuses on the differences in the distributions of data in a time based window measured against a baseline for a given pipeline and can detect situations where values are still within the expected range but the distribution has shifted. For example, if your model predicts housing prices you might expect the predictions to be between \\$200,000 and \\$1,000,000 with a distribution centered around \\$400,000. If your model suddenly starts predicting prices centered around \\$250,000 or \\$750,000 the predictions may still be within the expected range but the shift may signal something has changed that should be investigated.

Ideally we'd also monitor the _quality_ of the predictions, **concept drift**. However this can be difficult as true labels are often not available or are severely delayed in practice. That is there may be a signficant lag between the time the prediction is made and the true (sale price) value is observed.

Consequently, model insights uses data drift detection techniques on both inputs and outputs to detect changes in the distributions of the data.

There are many useful statistical tests for calculating the difference between distributions; however, they typically require assumptions about the underlying distributions or confusing and expensive calculations. We've implemented a data drift framework that is easy to understand, fast to compute, runs in an automated fashion and is extensible to many specific use cases.

The methodology currently revolves around calculating the specific percentile-based bins of the baseline distribution and measuring how future distributions fall into these bins. This approach is both visually intuitive and supports an easy to calculate difference score between distributions. Users can tune the scoring mechanism to emphasize different regions of the distribution: for example, you may only care if there is a change in the top 20th percentile of the distribution, compared to the baseline.

You can specify the inputs or outputs that you want to monitor and the data to use for your baselines. You can also specify how often you want to monitor distributions and set parameters to define what constitutes a meaningful change in a distribution for your application. 

Once you've set up a monitoring task, called an assay, comparisons against your baseline are then run automatically on a scheduled basis. You can be notified if the system notices any abnormally different behavior. The framework also allows you to quickly investigate the cause of any unexpected drifts in your predictions.

The rest of this notebook will shows how to create assays to monitor your pipelines.

**NOTE:** model insights operates over time and is difficult to demo in a notebook without pre-canned data. **We assume you have an active pipeline that has been running and making predictions over time and show you the code you may use to analyze your pipeline.**

## Prerequisites

* A deployed Wallaroo instance
* The following Python libraries installed:
  * `os`
  * `datetime`
  * `json`
  * `string`
  * `random`
  * [`numpy`](https://pypi.org/project/numpy/)
  * [`matplotlib`](https://pypi.org/project/matplotlib/)
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame.
  * [`pyarrow`](https://pypi.org/project/pyarrow/): PyArrow for Apache Arrow support.

### Workflow

Model Insights has the capability to perform interactive assays so that you can explore the data from a pipeline and learn how the data is behaving. With this information and the knowledge of your particular business use case you can then choose appropriate thresholds for persistent automatic assays as desired.

To get started lets import some libraries we'll need.

```python
import datetime as dt
from datetime import datetime, timedelta, timezone, tzinfo
import wallaroo
from wallaroo.object import EntityNotFoundError

import wallaroo.assay
from wallaroo.assay_config import BinMode, Aggregation, Metric

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import json
from IPython.display import display

# used to display dataframe information without truncating
from IPython.display import display
pd.set_option('display.max_colwidth', None)

plt.rcParams["figure.figsize"] = (12,6)
pd.options.display.float_format = '{:,.2f}'.format

# ignoring warnings for demonstration
import warnings
warnings.filterwarnings('ignore')
```

```python
wallaroo.__version__
```

    '2023.2.0rc3'

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Client connection from local Wallaroo instance

wl = wallaroo.Client()
```

### Connect to Workspace and Pipeline

We will now connect to the existing workspace and pipeline.  Update the variables below to match the ones used for past inferences.

```python
workspace_name = 'housepricedrift'
pipeline_name = 'housepricepipe'
model_name = 'housepricemodel'

# Used to generate a unique assay name for each run

import string
import random
# make a random 4 character prefix
prefix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

assay_name = f"{prefix}example assay"
```

```python
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
```

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
pipeline
```

<table><tr><th>name</th> <td>housepricepipe</td></tr><tr><th>created</th> <td>2023-05-17 20:41:50.504206+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 20:41:50.757679+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>4d9dfb3b-c9ae-402a-96fc-20ae0a2b2279, fc68f5f2-7bbf-435e-b434-e0c89c28c6a9</td></tr><tr><th>steps</th> <td>housepricemodel</td></tr></table>

We assume the pipeline has been running for a while and there is a period of time that is free of errors that we'd like to use as the _baseline_. Lets note the start and end times. For this example we have 30 days of data from Jan 2023 and well use Jan 1 data as our baseline.

```python
import datetime
baseline_start = datetime.datetime.fromisoformat('2023-01-01T00:00:00+00:00')
baseline_end = datetime.datetime.fromisoformat('2023-01-02T00:00:00+00:00')
last_day = datetime.datetime.fromisoformat('2023-02-01T00:00:00+00:00')
```

Lets create an assay using that pipeline and the model in the pipeline. We also specify the start end end of the baseline.

It is highly recommended when creating assays to set the input/output path with the `add_iopath` method.  This specifies:

* Whether to track the input or output variables of an inference.
* The name of the field to track.
* The index of the field.

In our example, that is `output dense_2 0` for "track the outputs, by the field `dense_2`, and the index of `dense_2` at 0.

```python
assay_builder = wl.build_assay(assay_name=assay_name, 
                               pipeline=pipeline, 
                               model_name=model_name, 
                               iopath="output dense_2 0", 
                               baseline_start=baseline_start, 
                               baseline_end=last_day)
```

We don't know much about our baseline data yet so lets examine the data and create a couple of visual representations. First lets get some basic stats on the baseline data.

```python
baseline_run = assay_builder.build().interactive_run()
baseline_run[0].baseline_stats()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Baseline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>182</td>
    </tr>
    <tr>
      <th>min</th>
      <td>12.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.97</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.94</td>
    </tr>
    <tr>
      <th>median</th>
      <td>12.88</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.45</td>
    </tr>
    <tr>
      <th>start</th>
      <td>2023-01-01T00:00:00Z</td>
    </tr>
    <tr>
      <th>end</th>
      <td>2023-01-02T00:00:00Z</td>
    </tr>
  </tbody>
</table>

Another option is the `baseline_dataframe` method to retrieve the baseline data with each field as a DataFrame column.  To cut down on space, we'll display just the `output_dense_2_0` column, which corresponds to the `output output_dense 2` iopath set earlier.

```python
assay_dataframe = assay_builder.baseline_dataframe()
display(assay_dataframe.loc[:, ["time", "metadata", "output_dense_2_0"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>metadata</th>
      <th>output_dense_2_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1672531200000</td>
      <td>{'last_model': '{"model_name": "housepricemodel", "model_sha": "test_version"}', 'profile': '{"elapsed_ns": 243}'}</td>
      <td>12.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1672531676753</td>
      <td>{'last_model': '{"model_name": "housepricemodel", "model_sha": "test_version"}', 'profile': '{"elapsed_ns": 216}'}</td>
      <td>13.36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1672532153506</td>
      <td>{'last_model': '{"model_name": "housepricemodel", "model_sha": "test_version"}', 'profile': '{"elapsed_ns": 128}'}</td>
      <td>12.80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1672532630259</td>
      <td>{'last_model': '{"model_name": "housepricemodel", "model_sha": "test_version"}', 'profile': '{"elapsed_ns": 333}'}</td>
      <td>12.79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1672533107013</td>
      <td>{'last_model': '{"model_name": "housepricemodel", "model_sha": "test_version"}', 'profile': '{"elapsed_ns": 53}'}</td>
      <td>13.16</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>177</th>
      <td>1672615585332</td>
      <td>{'last_model': '{"model_name": "housepricemodel", "model_sha": "test_version"}', 'profile': '{"elapsed_ns": 228}'}</td>
      <td>12.37</td>
    </tr>
    <tr>
      <th>178</th>
      <td>1672616062086</td>
      <td>{'last_model': '{"model_name": "housepricemodel", "model_sha": "test_version"}', 'profile': '{"elapsed_ns": 195}'}</td>
      <td>12.96</td>
    </tr>
    <tr>
      <th>179</th>
      <td>1672616538839</td>
      <td>{'last_model': '{"model_name": "housepricemodel", "model_sha": "test_version"}', 'profile': '{"elapsed_ns": 113}'}</td>
      <td>12.37</td>
    </tr>
    <tr>
      <th>180</th>
      <td>1672617015592</td>
      <td>{'last_model': '{"model_name": "housepricemodel", "model_sha": "test_version"}', 'profile': '{"elapsed_ns": 94}'}</td>
      <td>12.61</td>
    </tr>
    <tr>
      <th>181</th>
      <td>1672617492346</td>
      <td>{'last_model': '{"model_name": "housepricemodel", "model_sha": "test_version"}', 'profile': '{"elapsed_ns": 211}'}</td>
      <td>12.47</td>
    </tr>
  </tbody>
</table>
<p>182 rows × 3 columns</p>

Now lets look at a histogram, kernel density estimate (KDE), and Emperical Cumulative Distribution (ecdf) charts of the baseline data. These will give us insite into the distributions of the predictions and features that the assay is configured for.

```python
assay_builder.baseline_histogram()
```

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_18_0.png" width="800" label="png">}}
    

```python
assay_builder.baseline_kde()
```

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_19_0.png" width="800" label="png">}}
    

```python
assay_builder.baseline_ecdf()
```

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_20_0.png" width="800" label="png">}}
    

## List Assays

Assays are listed through the Wallaroo Client `list_assays` method. 

```python
wl.list_assays()
```

<table><tr><th>name</th><th>active</th><th>status</th><th>warning_threshold</th><th>alert_threshold</th><th>pipeline_name</th></tr><tr><td>api_assay</td><td>True</td><td>created</td><td>0.0</td><td>0.1</td><td>housepricepipe</td></tr></table>

### Interactive Baseline Runs
We can do an interactive run of just the baseline part to see how the baseline data will be put into bins. This assay uses quintiles so all 5 bins (not counting the outlier bins) have 20% of the predictions. We can see the bin boundaries along the x-axis.

```python
baseline_run.chart()
```

    baseline mean = 12.940910643273655
    baseline median = 12.884286880493164
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_24_1.png" width="800" label="png">}}
    

We can also get a dataframe with the bin/edge information.

```python
baseline_run.baseline_bins()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b_edges</th>
      <th>b_edge_names</th>
      <th>b_aggregated_values</th>
      <th>b_aggregation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.00</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.55</td>
      <td>q_20</td>
      <td>0.20</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.81</td>
      <td>q_40</td>
      <td>0.20</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.98</td>
      <td>q_60</td>
      <td>0.20</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.33</td>
      <td>q_80</td>
      <td>0.20</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>5</th>
      <td>14.97</td>
      <td>q_100</td>
      <td>0.20</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>6</th>
      <td>inf</td>
      <td>right_outlier</td>
      <td>0.00</td>
      <td>Density</td>
    </tr>
  </tbody>
</table>

The previous assay used quintiles so all of the bins had the same percentage/count of samples.  To get bins that are divided equaly along the range of values we can use `BinMode.EQUAL`.

```python
equal_bin_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end)
equal_bin_builder.summarizer_builder.add_bin_mode(BinMode.EQUAL)
equal_baseline = equal_bin_builder.build().interactive_baseline_run()
equal_baseline.chart()
```

    baseline mean = 12.940910643273655
    baseline median = 12.884286880493164
    bin_mode = Equal
    aggregation = Density
    metric = PSI
    weighted = False

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_28_1.png" width="800" label="png">}}
    

We now see very different bin edges and sample percentages per bin.

```python
equal_baseline.baseline_bins()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b_edges</th>
      <th>b_edge_names</th>
      <th>b_aggregated_values</th>
      <th>b_aggregation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.00</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.60</td>
      <td>p_1.26e1</td>
      <td>0.24</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.19</td>
      <td>p_1.32e1</td>
      <td>0.49</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13.78</td>
      <td>p_1.38e1</td>
      <td>0.22</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.38</td>
      <td>p_1.44e1</td>
      <td>0.04</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>5</th>
      <td>14.97</td>
      <td>p_1.50e1</td>
      <td>0.01</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>6</th>
      <td>inf</td>
      <td>right_outlier</td>
      <td>0.00</td>
      <td>Density</td>
    </tr>
  </tbody>
</table>

### Interactive Assay Runs
By default the assay builder creates an assay with some good starting parameters. In particular the assay is configured to run a new analysis for every 24 hours starting at the end of the baseline period. Additionally, it sets the number of bins to 5 so creates quintiles, and sets the target `iopath` to `"outputs 0 0"` which means we want to monitor the first column of the first output/prediction.

We can do an interactive run of just the baseline part to see how the baseline data will be put into bins. This assay uses quintiles so all 5 bins (not counting the outlier bins) have 20% of the predictions. We can see the bin boundaries along the x-axis.

We then run it with `interactive_run` and convert it to a dataframe for easy analysis with `to_dataframe`.

Now lets do an interactive run of the first assay as it is configured.  Interactive runs don't save the assay to the database (so they won't be scheduled in the future) nor do they save the assay results. Instead the results are returned after a short while for further analysis.

```python
assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end)
assay_config = assay_builder.add_run_until(last_day).build()
assay_results = assay_config.interactive_run()
```

```python
assay_df = assay_results.to_dataframe()
assay_df.loc[:, ~assay_df.columns.isin(['assay_id', 'iopath', 'name', 'warning_threshold'])]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score</th>
      <th>start</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>median</th>
      <th>std</th>
      <th>alert_threshold</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>2023-01-02T00:00:00+00:00</td>
      <td>12.05</td>
      <td>14.71</td>
      <td>12.97</td>
      <td>12.90</td>
      <td>0.48</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.09</td>
      <td>2023-01-03T00:00:00+00:00</td>
      <td>12.04</td>
      <td>14.65</td>
      <td>12.96</td>
      <td>12.93</td>
      <td>0.41</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.04</td>
      <td>2023-01-04T00:00:00+00:00</td>
      <td>11.87</td>
      <td>14.02</td>
      <td>12.98</td>
      <td>12.95</td>
      <td>0.46</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.06</td>
      <td>2023-01-05T00:00:00+00:00</td>
      <td>11.92</td>
      <td>14.46</td>
      <td>12.93</td>
      <td>12.87</td>
      <td>0.46</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.02</td>
      <td>2023-01-06T00:00:00+00:00</td>
      <td>12.02</td>
      <td>14.15</td>
      <td>12.95</td>
      <td>12.90</td>
      <td>0.43</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.03</td>
      <td>2023-01-07T00:00:00+00:00</td>
      <td>12.18</td>
      <td>14.58</td>
      <td>12.96</td>
      <td>12.93</td>
      <td>0.44</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.02</td>
      <td>2023-01-08T00:00:00+00:00</td>
      <td>12.01</td>
      <td>14.60</td>
      <td>12.92</td>
      <td>12.90</td>
      <td>0.46</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.04</td>
      <td>2023-01-09T00:00:00+00:00</td>
      <td>12.01</td>
      <td>14.40</td>
      <td>13.00</td>
      <td>12.97</td>
      <td>0.45</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.06</td>
      <td>2023-01-10T00:00:00+00:00</td>
      <td>11.99</td>
      <td>14.79</td>
      <td>12.94</td>
      <td>12.91</td>
      <td>0.46</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.02</td>
      <td>2023-01-11T00:00:00+00:00</td>
      <td>11.90</td>
      <td>14.66</td>
      <td>12.91</td>
      <td>12.88</td>
      <td>0.45</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.02</td>
      <td>2023-01-12T00:00:00+00:00</td>
      <td>11.96</td>
      <td>14.82</td>
      <td>12.94</td>
      <td>12.90</td>
      <td>0.46</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.03</td>
      <td>2023-01-13T00:00:00+00:00</td>
      <td>12.07</td>
      <td>14.61</td>
      <td>12.96</td>
      <td>12.93</td>
      <td>0.47</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.15</td>
      <td>2023-01-14T00:00:00+00:00</td>
      <td>12.00</td>
      <td>14.20</td>
      <td>13.06</td>
      <td>13.03</td>
      <td>0.43</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2.92</td>
      <td>2023-01-15T00:00:00+00:00</td>
      <td>12.74</td>
      <td>15.62</td>
      <td>14.00</td>
      <td>14.01</td>
      <td>0.57</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>14</th>
      <td>7.89</td>
      <td>2023-01-16T00:00:00+00:00</td>
      <td>14.64</td>
      <td>17.19</td>
      <td>15.91</td>
      <td>15.87</td>
      <td>0.63</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>15</th>
      <td>8.87</td>
      <td>2023-01-17T00:00:00+00:00</td>
      <td>16.60</td>
      <td>19.23</td>
      <td>17.94</td>
      <td>17.94</td>
      <td>0.63</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>16</th>
      <td>8.87</td>
      <td>2023-01-18T00:00:00+00:00</td>
      <td>18.67</td>
      <td>21.29</td>
      <td>20.01</td>
      <td>20.04</td>
      <td>0.64</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>17</th>
      <td>8.87</td>
      <td>2023-01-19T00:00:00+00:00</td>
      <td>20.72</td>
      <td>23.57</td>
      <td>22.17</td>
      <td>22.18</td>
      <td>0.65</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>18</th>
      <td>8.87</td>
      <td>2023-01-20T00:00:00+00:00</td>
      <td>23.04</td>
      <td>25.72</td>
      <td>24.32</td>
      <td>24.33</td>
      <td>0.66</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>19</th>
      <td>8.87</td>
      <td>2023-01-21T00:00:00+00:00</td>
      <td>25.06</td>
      <td>27.67</td>
      <td>26.48</td>
      <td>26.49</td>
      <td>0.63</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>20</th>
      <td>8.87</td>
      <td>2023-01-22T00:00:00+00:00</td>
      <td>27.21</td>
      <td>29.89</td>
      <td>28.63</td>
      <td>28.58</td>
      <td>0.65</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>21</th>
      <td>8.87</td>
      <td>2023-01-23T00:00:00+00:00</td>
      <td>29.36</td>
      <td>32.18</td>
      <td>30.82</td>
      <td>30.80</td>
      <td>0.67</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>22</th>
      <td>8.87</td>
      <td>2023-01-24T00:00:00+00:00</td>
      <td>31.56</td>
      <td>34.35</td>
      <td>32.98</td>
      <td>32.98</td>
      <td>0.65</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>23</th>
      <td>8.87</td>
      <td>2023-01-25T00:00:00+00:00</td>
      <td>33.68</td>
      <td>36.44</td>
      <td>35.14</td>
      <td>35.14</td>
      <td>0.66</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>24</th>
      <td>8.87</td>
      <td>2023-01-26T00:00:00+00:00</td>
      <td>35.93</td>
      <td>38.51</td>
      <td>37.31</td>
      <td>37.33</td>
      <td>0.65</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>25</th>
      <td>3.69</td>
      <td>2023-01-27T00:00:00+00:00</td>
      <td>12.06</td>
      <td>39.91</td>
      <td>29.29</td>
      <td>38.65</td>
      <td>12.66</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.05</td>
      <td>2023-01-28T00:00:00+00:00</td>
      <td>11.87</td>
      <td>13.88</td>
      <td>12.92</td>
      <td>12.90</td>
      <td>0.38</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.10</td>
      <td>2023-01-29T00:00:00+00:00</td>
      <td>12.02</td>
      <td>14.36</td>
      <td>12.98</td>
      <td>12.96</td>
      <td>0.38</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.11</td>
      <td>2023-01-30T00:00:00+00:00</td>
      <td>11.99</td>
      <td>14.44</td>
      <td>12.89</td>
      <td>12.88</td>
      <td>0.37</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.01</td>
      <td>2023-01-31T00:00:00+00:00</td>
      <td>12.00</td>
      <td>14.64</td>
      <td>12.92</td>
      <td>12.89</td>
      <td>0.40</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
  </tbody>
</table>

Basic functionality for creating quick charts is included.

```python
assay_results.chart_scores()
```

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_35_0.png" width="800" label="png">}}
    

We see that the difference scores are low for a while and then jump up to indicate there is an issue. We can examine that particular window to help us decide if that threshold is set correctly or not.

We can generate a quick chart of the results. This chart shows the 5 quantile bins (quintiles) derived from the baseline data plus one for left outliers and one for right outliers.  We also see that the data from the window falls within the baseline quintiles but in a different proportion and is skewing higher. Whether this is an issue or not is specific to your use case.

First lets examine a day that is only slightly different than the baseline. We see that we do see some values that fall outside of the range from the baseline values, the left and right outliers, and that the bin values are different but similar.

```python
assay_results[0].chart()
```

    baseline mean = 12.940910643273655
    window mean = 12.969964654406132
    baseline median = 12.884286880493164
    window median = 12.899214744567873
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.0029273068646199748
    scores = [0.0, 0.000514261205558409, 0.0002139202456922972, 0.0012617897456473992, 0.0002139202456922972, 0.0007234154220295724, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_37_1.png" width="800" label="png">}}
    

Other days, however are significantly different.

```python
assay_results[12].chart()
```

    baseline mean = 12.940910643273655
    window mean = 13.06380216891949
    baseline median = 12.884286880493164
    window median = 13.027600288391112
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.15060511096978788
    scores = [4.6637149189075455e-05, 0.05969428191167242, 0.00806617426854112, 0.008316273402678306, 0.07090885609902021, 0.003572888138686759, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_39_1.png" width="800" label="png">}}
    

```python
assay_results[13].chart()
```

    baseline mean = 12.940910643273655
    window mean = 14.004728427908038
    baseline median = 12.884286880493164
    window median = 14.009637832641602
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 2.9220486095961196
    scores = [0.0, 0.7090936334784107, 0.7130482300184766, 0.33500731896676245, 0.12171058214520876, 0.9038825518183468, 0.1393062931689142]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_40_1.png" width="800" label="png">}}
    

If we want to investigate further, we can run interactive assays on each of the inputs to see if any of them show anything abnormal. In this example we'll provide the feature labels to create more understandable titles.

The current assay expects continuous data. Sometimes categorical data is encoded as 1 or 0 in a feature and sometimes in a limited number of values such as 1, 2, 3. If one value has high a percentage the analysis emits a warning so that we know the scores for that feature may not behave as we expect.

```python
labels = ['bedrooms', 'bathrooms', 'lat', 'long', 'waterfront', 'sqft_living', 'sqft_lot', 'floors', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']

topic = wl.get_topic_name(pipeline.id())

all_inferences = wl.get_raw_pipeline_inference_logs(topic, baseline_start, last_day, model_name, limit=1_000_000)

assay_builder = wl.build_assay("Input Assay", pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.window_builder().add_width(hours=4)
assay_config = assay_builder.build()
assay_results = assay_config.interactive_input_run(all_inferences, labels)
iadf = assay_results.to_dataframe()
display(iadf.loc[:, ~iadf.columns.isin(['assay_id', 'iopath', 'name', 'warning_threshold'])])
```

    column distinct_vals label           largest_pct
         0            17 bedrooms        0.4244 
         1            44 bathrooms       0.2398 
         2          3281 lat             0.0014 
         3           959 long            0.0066 
         4             4 waterfront      0.9156 *** May not be continuous feature
         5          3901 sqft_living     0.0032 
         6          3487 sqft_lot        0.0173 
         7            11 floors          0.4567 
         8            10 view            0.8337 
         9             9 condition       0.5915 
        10            19 grade           0.3943 
        11           745 sqft_above      0.0096 
        12           309 sqft_basement   0.5582 
        13           224 yr_built        0.0239 
        14            77 yr_renovated    0.8889 
        15           649 sqft_living15   0.0093 
        16          3280 sqft_lot15      0.0199 

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score</th>
      <th>start</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>median</th>
      <th>std</th>
      <th>alert_threshold</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.19</td>
      <td>2023-01-02T00:00:00+00:00</td>
      <td>-2.54</td>
      <td>1.75</td>
      <td>0.21</td>
      <td>0.68</td>
      <td>0.99</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.03</td>
      <td>2023-01-02T04:00:00+00:00</td>
      <td>-1.47</td>
      <td>2.82</td>
      <td>0.21</td>
      <td>-0.40</td>
      <td>0.95</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.09</td>
      <td>2023-01-02T08:00:00+00:00</td>
      <td>-2.54</td>
      <td>3.89</td>
      <td>-0.04</td>
      <td>-0.40</td>
      <td>1.22</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.05</td>
      <td>2023-01-02T12:00:00+00:00</td>
      <td>-1.47</td>
      <td>2.82</td>
      <td>-0.12</td>
      <td>-0.40</td>
      <td>0.94</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.08</td>
      <td>2023-01-02T16:00:00+00:00</td>
      <td>-1.47</td>
      <td>1.75</td>
      <td>-0.00</td>
      <td>-0.40</td>
      <td>0.76</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3055</th>
      <td>0.08</td>
      <td>2023-01-31T04:00:00+00:00</td>
      <td>-0.42</td>
      <td>4.87</td>
      <td>0.25</td>
      <td>-0.17</td>
      <td>1.13</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>3056</th>
      <td>0.58</td>
      <td>2023-01-31T08:00:00+00:00</td>
      <td>-0.43</td>
      <td>2.01</td>
      <td>-0.04</td>
      <td>-0.21</td>
      <td>0.48</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>3057</th>
      <td>0.13</td>
      <td>2023-01-31T12:00:00+00:00</td>
      <td>-0.32</td>
      <td>7.75</td>
      <td>0.30</td>
      <td>-0.20</td>
      <td>1.57</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>3058</th>
      <td>0.26</td>
      <td>2023-01-31T16:00:00+00:00</td>
      <td>-0.43</td>
      <td>5.88</td>
      <td>0.19</td>
      <td>-0.18</td>
      <td>1.17</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>3059</th>
      <td>0.84</td>
      <td>2023-01-31T20:00:00+00:00</td>
      <td>-0.40</td>
      <td>0.52</td>
      <td>-0.17</td>
      <td>-0.25</td>
      <td>0.18</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
  </tbody>
</table>
<p>3060 rows × 9 columns</p>

We can chart each of the iopaths and do a visual inspection. From the charts we see that if any of the input features had significant differences in the first two days which we can choose to inspect further. Here we choose to show 3 charts just to save space in this notebook.

```python
assay_results.chart_iopaths(labels=labels, selected_labels=['bedrooms', 'lat', 'sqft_living'])
```

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_44_0.png" width="800" label="png">}}
    

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_44_1.png" width="800" label="png">}}
    

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_44_2.png" width="800" label="png">}}
    

When we are comfortable with what alert threshold should be for our specific purposes we can create and save an assay that will be automatically run on a daily basis.

In this example we're create an assay that runs everyday against the baseline and has an alert threshold of 0.5.

Once we upload it it will be saved and scheduled for future data as well as run against past data.

```python
alert_threshold = 0.5
import string
import random

prefix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

assay_name = f"{prefix}example assay"
assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end).add_alert_threshold(alert_threshold)
assay_id = assay_builder.upload()
```

After a short while, we can get the assay results for further analysis.

When we get the assay results, we see that the assays analysis is similar to the interactive run we started with though the analysis for the third day does not exceed the new alert threshold we set. And since we called `upload` instead of `interactive_run` the assay was saved to the system and will continue to run automatically on schedule from now on.

## Scheduling Assays

By default assays are scheduled to run every 24 hours starting immediately after the baseline period ends.

However, you can control the start time by setting `start` and the frequency by setting `interval` on the window.

So to recap:

* The window width is the size of the window. The default is 24 hours.
* The interval is how often the analysis is run, how far the window is slid into the future based on the last run. The default is the window width.
* The window start is when the analysis should start. The default is the end of the baseline period.

For example to run an analysis every 12 hours on the previous 24 hours of data you'd set the window width to 24 (the default) and the interval to 12.

```python
assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end)
assay_builder = assay_builder.add_run_until(last_day)

assay_builder.window_builder().add_width(hours=24).add_interval(hours=12)

assay_config = assay_builder.build()

assay_results = assay_config.interactive_run()
print(f"Generated {len(assay_results)} analyses")
```

    Generated 59 analyses

```python
assay_results.chart_scores()
```

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_51_0.png" width="800" label="png">}}
    

To start a weekly analysis of the previous week on a specific day, set the start date (taking care to specify the desired timezone), and the width and interval to 1 week and of course an analysis won't be generated till a window is complete.

By default, assay start date is set to 24 hours from when the assay was created.  For this example, we will set the `assay.window_builder.add_start` to set the assay window to start at the beginning of our data, and `assay.add_run_until` to set the time period to stop gathering data from.

```python
report_start = datetime.datetime.fromisoformat('2022-01-03T00:00:00+00:00')

assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end)
assay_builder = assay_builder.add_run_until(last_day)

assay_builder.window_builder().add_width(weeks=1).add_interval(weeks=1).add_start(report_start)

assay_config = assay_builder.build()

assay_results = assay_config.interactive_run()
print(f"Generated {len(assay_results)} analyses")
```

    Generated 5 analyses

```python
assay_results.chart_scores()
```

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_54_0.png" width="800" label="png">}}
    

## Advanced Configuration

The assay can be configured in a variety of ways to help customize it to your particular needs. Specifically you can:
* change the BinMode to evenly spaced, quantile or user provided
* change the number of bins to use
* provide weights to use when scoring the bins
* calculate the score using the sum of differences, maximum difference or population stability index
* change the value aggregation for the bins to density, cumulative or edges

Lets take a look at these in turn.

### Default configuration

First lets look at the default configuration. This is a lot of information but much of it is useful to know where it is available.

We see that the assay is broken up into 4 sections. A top level meta data section, a section for the baseline specification, a section for the window specification and a section that specifies the summarization configuration.

In the meta section we see the name of the assay, that it runs on the first column of the first output `"outputs 0 0"` and that there is a default threshold of 0.25.

The summarizer section shows us the defaults of Quantile, Density and PSI on 5 bins.

The baseline section shows us that it is configured as a fixed baseline with the specified start and end date times.

And the window tells us what model in the pipeline we are analyzing and how often.

```python
assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
print(assay_builder.build().to_json())
```

    {
        "name": "onmyexample assay",
        "pipeline_id": 1,
        "pipeline_name": "housepricepipe",
        "active": true,
        "status": "created",
        "iopath": "output dense_2 0",
        "baseline": {
            "Fixed": {
                "pipeline": "housepricepipe",
                "model": "housepricemodel",
                "start_at": "2023-01-01T00:00:00+00:00",
                "end_at": "2023-01-02T00:00:00+00:00"
            }
        },
        "window": {
            "pipeline": "housepricepipe",
            "model": "housepricemodel",
            "width": "24 hours",
            "start": null,
            "interval": null
        },
        "summarizer": {
            "type": "UnivariateContinuous",
            "bin_mode": "Quantile",
            "aggregation": "Density",
            "metric": "PSI",
            "num_bins": 5,
            "bin_weights": null,
            "bin_width": null,
            "provided_edges": null,
            "add_outlier_edges": true
        },
        "warning_threshold": null,
        "alert_threshold": 0.25,
        "run_until": "2023-02-01T00:00:00+00:00",
        "workspace_id": 5
    }

## Defaults

We can run the assay interactively and review the first analysis. The method `compare_basic_stats` gives us a dataframe with basic stats for the baseline and window data.

```python
assay_results = assay_builder.build().interactive_run()
ar = assay_results[0]

ar.compare_basic_stats()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Baseline</th>
      <th>Window</th>
      <th>diff</th>
      <th>pct_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>182.00</td>
      <td>181.00</td>
      <td>-1.00</td>
      <td>-0.55</td>
    </tr>
    <tr>
      <th>min</th>
      <td>12.00</td>
      <td>12.05</td>
      <td>0.04</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.97</td>
      <td>14.71</td>
      <td>-0.26</td>
      <td>-1.71</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.94</td>
      <td>12.97</td>
      <td>0.03</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>median</th>
      <td>12.88</td>
      <td>12.90</td>
      <td>0.01</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.45</td>
      <td>0.48</td>
      <td>0.03</td>
      <td>5.68</td>
    </tr>
    <tr>
      <th>start</th>
      <td>2023-01-01T00:00:00+00:00</td>
      <td>2023-01-02T00:00:00+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>end</th>
      <td>2023-01-02T00:00:00+00:00</td>
      <td>2023-01-03T00:00:00+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>

The method `compare_bins` gives us a dataframe with the bin information. Such as the number of bins, the right edges, suggested bin/edge names and the values for each bin in the baseline and the window.

```python
assay_bins = ar.compare_bins()
display(assay_bins.loc[:, assay_bins.columns!='w_aggregation'])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b_edges</th>
      <th>b_edge_names</th>
      <th>b_aggregated_values</th>
      <th>b_aggregation</th>
      <th>w_edges</th>
      <th>w_edge_names</th>
      <th>w_aggregated_values</th>
      <th>diff_in_pcts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.00</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>Density</td>
      <td>12.00</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.55</td>
      <td>q_20</td>
      <td>0.20</td>
      <td>Density</td>
      <td>12.55</td>
      <td>e_1.26e1</td>
      <td>0.19</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.81</td>
      <td>q_40</td>
      <td>0.20</td>
      <td>Density</td>
      <td>12.81</td>
      <td>e_1.28e1</td>
      <td>0.21</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.98</td>
      <td>q_60</td>
      <td>0.20</td>
      <td>Density</td>
      <td>12.98</td>
      <td>e_1.30e1</td>
      <td>0.18</td>
      <td>-0.02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.33</td>
      <td>q_80</td>
      <td>0.20</td>
      <td>Density</td>
      <td>13.33</td>
      <td>e_1.33e1</td>
      <td>0.21</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>14.97</td>
      <td>q_100</td>
      <td>0.20</td>
      <td>Density</td>
      <td>14.97</td>
      <td>e_1.50e1</td>
      <td>0.21</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>right_outlier</td>
      <td>0.00</td>
      <td>Density</td>
      <td>NaN</td>
      <td>right_outlier</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>

We can also plot the chart to visualize the values of the bins.

```python
ar.chart()
```

    baseline mean = 12.940910643273655
    window mean = 12.969964654406132
    baseline median = 12.884286880493164
    window median = 12.899214744567873
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.0029273068646199748
    scores = [0.0, 0.000514261205558409, 0.0002139202456922972, 0.0012617897456473992, 0.0002139202456922972, 0.0007234154220295724, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_62_1.png" width="800" label="png">}}
    

## Binning Mode

We can change the bin mode algorithm to equal and see that the bins/edges are partitioned at different points and the bins have different values.

```python
prefix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

assay_name = f"{prefix}example assay"

assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_bin_mode(BinMode.EQUAL)
assay_results = assay_builder.build().interactive_run()
assay_results_df = assay_results[0].compare_bins()
display(assay_results_df.loc[:, ~assay_results_df.columns.isin(['b_aggregation', 'w_aggregation'])])
assay_results[0].chart()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b_edges</th>
      <th>b_edge_names</th>
      <th>b_aggregated_values</th>
      <th>w_edges</th>
      <th>w_edge_names</th>
      <th>w_aggregated_values</th>
      <th>diff_in_pcts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.00</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>12.00</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.60</td>
      <td>p_1.26e1</td>
      <td>0.24</td>
      <td>12.60</td>
      <td>e_1.26e1</td>
      <td>0.24</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.19</td>
      <td>p_1.32e1</td>
      <td>0.49</td>
      <td>13.19</td>
      <td>e_1.32e1</td>
      <td>0.48</td>
      <td>-0.02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13.78</td>
      <td>p_1.38e1</td>
      <td>0.22</td>
      <td>13.78</td>
      <td>e_1.38e1</td>
      <td>0.22</td>
      <td>-0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.38</td>
      <td>p_1.44e1</td>
      <td>0.04</td>
      <td>14.38</td>
      <td>e_1.44e1</td>
      <td>0.06</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>5</th>
      <td>14.97</td>
      <td>p_1.50e1</td>
      <td>0.01</td>
      <td>14.97</td>
      <td>e_1.50e1</td>
      <td>0.01</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>right_outlier</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>right_outlier</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>

    baseline mean = 12.940910643273655
    window mean = 12.969964654406132
    baseline median = 12.884286880493164
    window median = 12.899214744567873
    bin_mode = Equal
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.011074287819376092
    scores = [0.0, 7.3591419975306595e-06, 0.000773779195360713, 8.538514991838585e-05, 0.010207597078872246, 1.6725322721660374e-07, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_64_2.png" width="800" label="png">}}
    

## User Provided Bin Edges

The values in this dataset run from ~11.6 to ~15.81. And lets say we had a business reason to use specific bin edges.  We can specify them with the BinMode.PROVIDED and specifying a list of floats with the right hand / upper edge of each bin and optionally the lower edge of the smallest bin. If the lowest edge is not specified the threshold for left outliers is taken from the smallest value in the baseline dataset.

```python
edges = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_bin_mode(BinMode.PROVIDED, edges)
assay_results = assay_builder.build().interactive_run()
assay_results_df = assay_results[0].compare_bins()
display(assay_results_df.loc[:, ~assay_results_df.columns.isin(['b_aggregation', 'w_aggregation'])])
assay_results[0].chart()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b_edges</th>
      <th>b_edge_names</th>
      <th>b_aggregated_values</th>
      <th>w_edges</th>
      <th>w_edge_names</th>
      <th>w_aggregated_values</th>
      <th>diff_in_pcts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11.00</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>11.00</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.00</td>
      <td>e_1.20e1</td>
      <td>0.00</td>
      <td>12.00</td>
      <td>e_1.20e1</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.00</td>
      <td>e_1.30e1</td>
      <td>0.62</td>
      <td>13.00</td>
      <td>e_1.30e1</td>
      <td>0.59</td>
      <td>-0.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.00</td>
      <td>e_1.40e1</td>
      <td>0.36</td>
      <td>14.00</td>
      <td>e_1.40e1</td>
      <td>0.35</td>
      <td>-0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15.00</td>
      <td>e_1.50e1</td>
      <td>0.02</td>
      <td>15.00</td>
      <td>e_1.50e1</td>
      <td>0.06</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>5</th>
      <td>16.00</td>
      <td>e_1.60e1</td>
      <td>0.00</td>
      <td>16.00</td>
      <td>e_1.60e1</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>right_outlier</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>right_outlier</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>

    baseline mean = 12.940910643273655
    window mean = 12.969964654406132
    baseline median = 12.884286880493164
    window median = 12.899214744567873
    bin_mode = Provided
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.0321620386600679
    scores = [0.0, 0.0, 0.0014576920813015586, 3.549754401142936e-05, 0.030668849034754912, 0.0, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_66_2.png" width="800" label="png">}}
    

## Number of Bins

We could also choose to a different number of bins, lets say 10, which can be evenly spaced or based on the quantiles (deciles).

```python
assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_bin_mode(BinMode.QUANTILE).add_num_bins(10)
assay_results = assay_builder.build().interactive_run()
assay_results_df = assay_results[1].compare_bins()
display(assay_results_df.loc[:, ~assay_results_df.columns.isin(['b_aggregation', 'w_aggregation'])])
assay_results[1].chart()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b_edges</th>
      <th>b_edge_names</th>
      <th>b_aggregated_values</th>
      <th>w_edges</th>
      <th>w_edge_names</th>
      <th>w_aggregated_values</th>
      <th>diff_in_pcts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.00</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>12.00</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.41</td>
      <td>q_10</td>
      <td>0.10</td>
      <td>12.41</td>
      <td>e_1.24e1</td>
      <td>0.09</td>
      <td>-0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.55</td>
      <td>q_20</td>
      <td>0.10</td>
      <td>12.55</td>
      <td>e_1.26e1</td>
      <td>0.04</td>
      <td>-0.05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.72</td>
      <td>q_30</td>
      <td>0.10</td>
      <td>12.72</td>
      <td>e_1.27e1</td>
      <td>0.14</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.81</td>
      <td>q_40</td>
      <td>0.10</td>
      <td>12.81</td>
      <td>e_1.28e1</td>
      <td>0.05</td>
      <td>-0.05</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12.88</td>
      <td>q_50</td>
      <td>0.10</td>
      <td>12.88</td>
      <td>e_1.29e1</td>
      <td>0.12</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>6</th>
      <td>12.98</td>
      <td>q_60</td>
      <td>0.10</td>
      <td>12.98</td>
      <td>e_1.30e1</td>
      <td>0.09</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>7</th>
      <td>13.15</td>
      <td>q_70</td>
      <td>0.10</td>
      <td>13.15</td>
      <td>e_1.32e1</td>
      <td>0.18</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>8</th>
      <td>13.33</td>
      <td>q_80</td>
      <td>0.10</td>
      <td>13.33</td>
      <td>e_1.33e1</td>
      <td>0.14</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>9</th>
      <td>13.47</td>
      <td>q_90</td>
      <td>0.10</td>
      <td>13.47</td>
      <td>e_1.35e1</td>
      <td>0.07</td>
      <td>-0.03</td>
    </tr>
    <tr>
      <th>10</th>
      <td>14.97</td>
      <td>q_100</td>
      <td>0.10</td>
      <td>14.97</td>
      <td>e_1.50e1</td>
      <td>0.08</td>
      <td>-0.02</td>
    </tr>
    <tr>
      <th>11</th>
      <td>NaN</td>
      <td>right_outlier</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>right_outlier</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>

    baseline mean = 12.940910643273655
    window mean = 12.956829186961135
    baseline median = 12.884286880493164
    window median = 12.929338455200195
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.16591076620684958
    scores = [0.0, 0.0002571306027792045, 0.044058279699182114, 0.009441459631493015, 0.03381618572319047, 0.0027335446937028877, 0.0011792419836838435, 0.051023062424253904, 0.009441459631493015, 0.008662563542113508, 0.0052978382749576496, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_68_2.png" width="800" label="png">}}
    

## Bin Weights

Now lets say we only care about differences at the higher end of the range. We can use weights to specify that difference in the lower bins should not be counted in the score. 

If we stick with 10 bins we can provide 10 a vector of 12 weights. One weight each for the original bins plus one at the front for the left outlier bin and one at the end for the right outlier bin.

Note we still show the values for the bins but the scores for the lower 5 and left outlier are 0 and only the right half is counted and reflected in the score.

```python
weights = [0] * 6
weights.extend([1] * 6)
print("Using weights: ", weights)
assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_bin_mode(BinMode.QUANTILE).add_num_bins(10).add_bin_weights(weights)
assay_results = assay_builder.build().interactive_run()
assay_results_df = assay_results[1].compare_bins()
display(assay_results_df.loc[:, ~assay_results_df.columns.isin(['b_aggregation', 'w_aggregation'])])
assay_results[1].chart()
```

    Using weights:  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b_edges</th>
      <th>b_edge_names</th>
      <th>b_aggregated_values</th>
      <th>w_edges</th>
      <th>w_edge_names</th>
      <th>w_aggregated_values</th>
      <th>diff_in_pcts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.00</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>12.00</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.41</td>
      <td>q_10</td>
      <td>0.10</td>
      <td>12.41</td>
      <td>e_1.24e1</td>
      <td>0.09</td>
      <td>-0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.55</td>
      <td>q_20</td>
      <td>0.10</td>
      <td>12.55</td>
      <td>e_1.26e1</td>
      <td>0.04</td>
      <td>-0.05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.72</td>
      <td>q_30</td>
      <td>0.10</td>
      <td>12.72</td>
      <td>e_1.27e1</td>
      <td>0.14</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.81</td>
      <td>q_40</td>
      <td>0.10</td>
      <td>12.81</td>
      <td>e_1.28e1</td>
      <td>0.05</td>
      <td>-0.05</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12.88</td>
      <td>q_50</td>
      <td>0.10</td>
      <td>12.88</td>
      <td>e_1.29e1</td>
      <td>0.12</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>6</th>
      <td>12.98</td>
      <td>q_60</td>
      <td>0.10</td>
      <td>12.98</td>
      <td>e_1.30e1</td>
      <td>0.09</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>7</th>
      <td>13.15</td>
      <td>q_70</td>
      <td>0.10</td>
      <td>13.15</td>
      <td>e_1.32e1</td>
      <td>0.18</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>8</th>
      <td>13.33</td>
      <td>q_80</td>
      <td>0.10</td>
      <td>13.33</td>
      <td>e_1.33e1</td>
      <td>0.14</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>9</th>
      <td>13.47</td>
      <td>q_90</td>
      <td>0.10</td>
      <td>13.47</td>
      <td>e_1.35e1</td>
      <td>0.07</td>
      <td>-0.03</td>
    </tr>
    <tr>
      <th>10</th>
      <td>14.97</td>
      <td>q_100</td>
      <td>0.10</td>
      <td>14.97</td>
      <td>e_1.50e1</td>
      <td>0.08</td>
      <td>-0.02</td>
    </tr>
    <tr>
      <th>11</th>
      <td>NaN</td>
      <td>right_outlier</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>right_outlier</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>

    baseline mean = 12.940910643273655
    window mean = 12.956829186961135
    baseline median = 12.884286880493164
    window median = 12.929338455200195
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = True
    score = 0.012600694309416988
    scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00019654033061397393, 0.00850384373737565, 0.0015735766052488358, 0.0014437605903522511, 0.000882973045826275, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_70_3.png" width="800" label="png">}}
    

## Metrics

The `score` is a distance or dis-similarity measure. The larger it is the less similar the two distributions are. We currently support
summing the differences of each individual bin, taking the maximum difference and a modified Population Stability Index (PSI).

The following three charts use each of the metrics. Note how the scores change. The best one will depend on your particular use case.

```python
assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_results = assay_builder.build().interactive_run()
assay_results[0].chart()
```

    baseline mean = 12.940910643273655
    window mean = 12.969964654406132
    baseline median = 12.884286880493164
    window median = 12.899214744567873
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.0029273068646199748
    scores = [0.0, 0.000514261205558409, 0.0002139202456922972, 0.0012617897456473992, 0.0002139202456922972, 0.0007234154220295724, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_72_1.png" width="800" label="png">}}
    

```python
assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_metric(Metric.SUMDIFF)
assay_results = assay_builder.build().interactive_run()
assay_results[0].chart()
```

    baseline mean = 12.940910643273655
    window mean = 12.969964654406132
    baseline median = 12.884286880493164
    window median = 12.899214744567873
    bin_mode = Quantile
    aggregation = Density
    metric = SumDiff
    weighted = False
    score = 0.025438649748041997
    scores = [0.0, 0.009956893934794486, 0.006648048084512165, 0.01548175581324751, 0.006648048084512165, 0.012142553579017668, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_73_1.png" width="800" label="png">}}
    

```python
assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_metric(Metric.MAXDIFF)
assay_results = assay_builder.build().interactive_run()
assay_results[0].chart()
```

    baseline mean = 12.940910643273655
    window mean = 12.969964654406132
    baseline median = 12.884286880493164
    window median = 12.899214744567873
    bin_mode = Quantile
    aggregation = Density
    metric = MaxDiff
    weighted = False
    score = 0.01548175581324751
    scores = [0.0, 0.009956893934794486, 0.006648048084512165, 0.01548175581324751, 0.006648048084512165, 0.012142553579017668, 0.0]
    index = 3

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_74_1.png" width="800" label="png">}}
    

## Aggregation Options

Also, bin aggregation can be done in histogram `Aggregation.DENSITY` style (the default) where we count the number/percentage of values that fall in each bin or Empirical Cumulative Density Function style `Aggregation.CUMULATIVE` where we keep a cumulative count of the values/percentages that fall in each bin.

```python
assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_aggregation(Aggregation.DENSITY)
assay_results = assay_builder.build().interactive_run()
assay_results[0].chart()
```

    baseline mean = 12.940910643273655
    window mean = 12.969964654406132
    baseline median = 12.884286880493164
    window median = 12.899214744567873
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.0029273068646199748
    scores = [0.0, 0.000514261205558409, 0.0002139202456922972, 0.0012617897456473992, 0.0002139202456922972, 0.0007234154220295724, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_76_1.png" width="800" label="png">}}
    

```python
assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_aggregation(Aggregation.CUMULATIVE)
assay_results = assay_builder.build().interactive_run()
assay_results[0].chart()
```

    baseline mean = 12.940910643273655
    window mean = 12.969964654406132
    baseline median = 12.884286880493164
    window median = 12.899214744567873
    bin_mode = Quantile
    aggregation = Cumulative
    metric = PSI
    weighted = False
    score = 0.04419889502762442
    scores = [0.0, 0.009956893934794486, 0.0033088458502823492, 0.01879060166352986, 0.012142553579017725, 0.0, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_77_1.png" width="800" label="png">}}
    

