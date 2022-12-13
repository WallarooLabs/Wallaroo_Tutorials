This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_insights).

## Wallaroo Assays Tutorial

Select [Steps](#steps) to skip the introduction and jump right into the code sample.

## Introduction

The Wallaroo Assays feature lets you monitor how the environment that your model operates within changes in ways that affect the model's predictions.  This allows you to intervene (aka retrain a model) in an efficient and timely manner. Changes in the inputs, **data drift**, can occur due to errors in the data processing pipeline or due to changes in the environment such as user behavior.

The validation framework performs per inference range checks with count frequency based thresholds for alerts and is ideal for catching many errors in input and output data.

In complement to the validation framework, model insights examines the distribution of data within a specified window of time, and compares it to a baseline for a given pipeline. It can detect situations where values are still within the expected range, but the distribution has shifted. 

For example, for a model that predicts housing prices you might expect the predictions to be between \\$200,000 and \\$1,000,000 with a distribution centered around \\$400,000. Then your model suddenly starts predicting prices centered around \\$250,000 or \\$750,000.  The predictions may still be within the expected range but the shift may signal something has changed that should be investigated.

Ideally we'd also monitor the _quality_ of the predictions, aka **concept drift**. However this can be difficult as true labels are often not available or are severely delayed in practice. That is there may be a significant lag between the time the prediction is made and the true (sale price) value is observed.

Consequently, model insights uses data drift detection techniques on both inputs and outputs to detect changes in the distributions of the data.

There are many useful statistical tests for calculating the difference between distributions; however, they typically require assumptions about the underlying distributions or confusing and expensive calculations. We've implemented a data drift framework that is easy to understand, fast to compute, runs in an automated fashion and is extensible to many specific use cases.

The methodology currently revolves around calculating the specific percentile-based bins of the baseline distribution and measuring how future distributions fall into these bins. This approach is both visually intuitive and supports an easy to calculate difference score between distributions. Users can tune the scoring mechanism to emphasize different regions of the distribution: for example, you may only care if there is a change in the top 20th percentile of the distribution, compared to the baseline.

You can specify the inputs or outputs that you want to monitor and the data to use for your baselines. You can also specify how often you want to monitor distributions and set parameters to define what constitutes a meaningful change in a distribution for your application. 

Once you've set up a monitoring task, called an assay, comparisons against your baseline are then run automatically on a scheduled basis. You can be notified if the system notices any abnormally different behavior. The framework also allows you to quickly investigate the cause of any unexpected drifts in your predictions.

The rest of this tutorial shows how to create assays to monitor your pipelines.

## Steps

Model Insights has the capability to perform interactive assays so that you can explore the data from a pipeline and learn how the data is behaving. With this information and the knowledge of your particular business use case you can then choose appropriate thresholds for persistent automatic assays as desired.

**NOTE:** Model insights operates over time and is difficult to demo in a notebook without pre-canned data. **We assume you have an active pipeline that has been running and making predictions over time and show you the code you may use to analyze your pipeline.**  If this historical data is not available, the Model Insights Canned Data Loader included with this tutorial as `model-insights-load_canned_data.ipynb` is made to establish a sample workspace, pipeline and model into your Wallaroo instance with canned historical data that can be used for this tutorial.

### Load Libraries

To get started we import the libraries we'll need.

```python
import matplotlib.pyplot as plt
import pandas as pd 

import wallaroo
from wallaroo.assay_config import BinMode, Aggregation, Metric
from wallaroo.object import EntityNotFoundError
```

### Set Configuration

The following configuration is used to connect to the pipeline used, and display the graphs.  The `pipeline_name` and `model_name` shown here are from the [Model Insights Canned Data Loader](model-insights-load_canned_data.ipynb), so adjust them based on your own needs.

```python
plt.rcParams["figure.figsize"] = (12,6)
pd.options.display.float_format = '{:,.2f}'.format

workspace_name = 'housepricedrift'
pipeline_name = 'housepricepipe'
model_name = 'housepricemodel'
```

### Connect to Wallaroo

Connect to your Wallaroo instance.

```python
wl = wallaroo.Client()
```

### Connect to Workspace and Pipeline

Connect to the workspace, pipeline, and model listed above.  This code assumes that there are not multiple pipelines with the same name.

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
        pipeline = wl.pipelines_by_name(pipeline_name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(pipeline_name)
    return pipeline

workspace = get_workspace(workspace_name)
wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
pipeline
```

<table><tr><th>name</th> <td>housepricepipe</td></tr><tr><th>created</th> <td>2022-10-10 18:38:51.033867+00:00</td></tr><tr><th>last_updated</th> <td>2022-10-10 18:38:51.121077+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>housepricemodel</td></tr></table>

We assume the pipeline has been running for a while and there is a period of time that is free of errors that we'd like to use as the _baseline_. Let's note the start and end times. For this example we have 30 days of data from Jan 2022 and will use Jan 1 data as our baseline.

```python
import datetime
baseline_start = datetime.datetime.fromisoformat('2022-01-01T00:00:00+00:00')
baseline_end = datetime.datetime.fromisoformat('2022-01-02T00:00:00+00:00')
last_day = datetime.datetime.fromisoformat('2022-02-01T00:00:00+00:00')
```

Let's create an assay using that pipeline and the model in the pipeline. We also specify the baseline start and end.

```python
assay_name = "example assay"
assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end)
```

We don't know much about our baseline data yet so let's examine the data and create a couple of visual representations. First let's get some basic stats on the baseline data.

```python
baseline_run = assay_builder.build().interactive_baseline_run()
baseline_run.baseline_stats()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>1813</td>
    </tr>
    <tr>
      <th>min</th>
      <td>11.95</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.08</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.95</td>
    </tr>
    <tr>
      <th>median</th>
      <td>12.91</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.46</td>
    </tr>
    <tr>
      <th>start</th>
      <td>2022-01-01T00:00:00Z</td>
    </tr>
    <tr>
      <th>end</th>
      <td>2022-01-02T00:00:00Z</td>
    </tr>
  </tbody>
</table>
</div>

Now let's look at a histogram, kernel density estimate (KDE), and Empirical Cumulative Distribution (ecdf) charts of the baseline data. These will give us insights into the distributions of the predictions and features that the assay is configured for.

```python
assay_builder.baseline_histogram()
```

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_18_0.png)
    

```python
assay_builder.baseline_kde()
```

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_19_0.png)
    

```python
assay_builder.baseline_ecdf()
```

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_20_0.png)
    

### Interactive Baseline Runs
We can do an interactive run of just the baseline part to see how the baseline data will be put into bins. This assay uses quintiles so all 5 bins (not counting the outlier bins) have 20% of the predictions. We can see the bin boundaries along the x-axis.

```python
baseline_run.chart()
```

    baseline mean = 12.954393170120568
    baseline median = 12.913979530334473
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_22_1.png)
    

We can also get a dataframe with the bin/edge information.

```python
baseline_run.baseline_bins()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>11.95</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.56</td>
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
      <td>13.01</td>
      <td>q_60</td>
      <td>0.20</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.31</td>
      <td>q_80</td>
      <td>0.20</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15.08</td>
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
</div>

The previous assay used quintiles so all of the bins had the same percentage/count of samples.  To get bins that are divided equally along the range of values we can use `BinMode.EQUAL`.

```python
equal_bin_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end)
equal_bin_builder.summarizer_builder.add_bin_mode(BinMode.EQUAL)
equal_baseline = equal_bin_builder.build().interactive_baseline_run()
equal_baseline.chart()
```

    baseline mean = 12.954393170120568
    baseline median = 12.913979530334473
    bin_mode = Equal
    aggregation = Density
    metric = PSI
    weighted = False

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_26_1.png)
    

We now see very different bin edges and sample percentages per bin.

```python
equal_baseline.baseline_bins()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>11.95</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.57</td>
      <td>p_1.26e1</td>
      <td>0.21</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.20</td>
      <td>p_1.32e1</td>
      <td>0.54</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13.83</td>
      <td>p_1.38e1</td>
      <td>0.21</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.45</td>
      <td>p_1.45e1</td>
      <td>0.04</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15.08</td>
      <td>p_1.51e1</td>
      <td>0.00</td>
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
</div>

### Interactive Assay Runs

By default the assay builder creates an assay with some good starting parameters. In particular the assay is configured to run a new analysis for **every 24 hours starting at the end of the baseline period**. Additionally, it sets the **number of bins to 5** to create quintiles, and sets the target `iopath` to `"outputs 0 0"` which means we want to monitor the first column of the first output/prediction.

We can do an interactive run of just the baseline part to see how the baseline data will be put into bins. This assay uses quintiles so all 5 bins (not counting the outlier bins) have 20% of the predictions. We can see the bin boundaries along the x-axis.

We then run it with `interactive_run` and convert it to a dataframe for easy analysis with `to_dataframe`.

Now let's do an interactive run of the first assay as it is configured.  Interactive runs don't save the assay to the database (so they won't be scheduled in the future) nor do they save the assay results. Instead the results are returned after a short while for further analysis.

#### Configuration Notes

By default the distance measure used is a modified version of the *Population Stability Index*, a measure that's widely used in banking and finance, and is also known as the *Jeffreys divergence*, or the *Symmetrised Kullback-Leibler divergence*.

There is a handy rule of thumb for determining whether the PSI score is "large":

* PSI < 0.1: The distance is small; the distributions are about the same
* 0.1 <= PSI < 0.2: The distance is moderately large; the distributions are somewhat different, and there may be some data drift
* PSI >= 0.2: The distance is large; the distributions are different. A prolonged range of PSI > 0.2 can indicate the model is no longer in operating bounds and should be retrained.

Of course, this is only a rule of thumb; different thresholds may work better for a specific application, and this exploration can help you properly tune the threshold (or other parameters, like the binning scheme or difference metric) as needed.

The red dots in the above graph indicate distance scores larger than our threshold of 0.1. We see that the difference scores are low for a while and then jump up to indicate there is an issue. We can examine that particular window to help us decide if that threshold is set correctly or not.

We can also retrieve the above results as a data frame, for further analysis, if desired.

```python
assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end)
assay_config = assay_builder.add_run_until(last_day).build()
assay_results = assay_config.interactive_run()
assay_df = assay_results.to_dataframe()
assay_df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>assay_id</th>
      <th>name</th>
      <th>iopath</th>
      <th>score</th>
      <th>start</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>median</th>
      <th>std</th>
      <th>warning_threshold</th>
      <th>alert_threshold</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>0.01</td>
      <td>2022-01-02T00:00:00Z</td>
      <td>11.62</td>
      <td>15.11</td>
      <td>12.95</td>
      <td>12.91</td>
      <td>0.45</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>1</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>0.01</td>
      <td>2022-01-03T00:00:00Z</td>
      <td>11.87</td>
      <td>15.39</td>
      <td>12.95</td>
      <td>12.90</td>
      <td>0.45</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>2</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>0.00</td>
      <td>2022-01-04T00:00:00Z</td>
      <td>11.74</td>
      <td>14.79</td>
      <td>12.95</td>
      <td>12.93</td>
      <td>0.44</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>0.02</td>
      <td>2022-01-05T00:00:00Z</td>
      <td>11.89</td>
      <td>15.81</td>
      <td>12.95</td>
      <td>12.92</td>
      <td>0.44</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>0.01</td>
      <td>2022-01-06T00:00:00Z</td>
      <td>11.83</td>
      <td>14.94</td>
      <td>12.95</td>
      <td>12.92</td>
      <td>0.44</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>5</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>0.02</td>
      <td>2022-01-07T00:00:00Z</td>
      <td>11.83</td>
      <td>15.14</td>
      <td>12.96</td>
      <td>12.92</td>
      <td>0.44</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>6</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>0.01</td>
      <td>2022-01-08T00:00:00Z</td>
      <td>11.89</td>
      <td>15.48</td>
      <td>12.93</td>
      <td>12.90</td>
      <td>0.43</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>7</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>0.01</td>
      <td>2022-01-09T00:00:00Z</td>
      <td>11.80</td>
      <td>15.12</td>
      <td>12.95</td>
      <td>12.91</td>
      <td>0.45</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>8</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>0.01</td>
      <td>2022-01-10T00:00:00Z</td>
      <td>11.93</td>
      <td>14.79</td>
      <td>12.95</td>
      <td>12.90</td>
      <td>0.44</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>9</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>0.01</td>
      <td>2022-01-11T00:00:00Z</td>
      <td>11.86</td>
      <td>14.81</td>
      <td>12.96</td>
      <td>12.93</td>
      <td>0.44</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>10</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>0.00</td>
      <td>2022-01-12T00:00:00Z</td>
      <td>11.80</td>
      <td>14.87</td>
      <td>12.95</td>
      <td>12.91</td>
      <td>0.46</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>11</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>0.01</td>
      <td>2022-01-13T00:00:00Z</td>
      <td>11.99</td>
      <td>14.61</td>
      <td>12.92</td>
      <td>12.88</td>
      <td>0.43</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>12</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>0.11</td>
      <td>2022-01-14T00:00:00Z</td>
      <td>11.98</td>
      <td>15.31</td>
      <td>13.02</td>
      <td>13.00</td>
      <td>0.38</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>13</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>3.06</td>
      <td>2022-01-15T00:00:00Z</td>
      <td>12.74</td>
      <td>16.32</td>
      <td>14.01</td>
      <td>13.99</td>
      <td>0.57</td>
      <td>None</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>14</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>7.53</td>
      <td>2022-01-16T00:00:00Z</td>
      <td>14.37</td>
      <td>17.76</td>
      <td>15.90</td>
      <td>15.89</td>
      <td>0.63</td>
      <td>None</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>15</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>8.87</td>
      <td>2022-01-17T00:00:00Z</td>
      <td>16.59</td>
      <td>19.30</td>
      <td>17.92</td>
      <td>17.92</td>
      <td>0.63</td>
      <td>None</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>16</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>8.87</td>
      <td>2022-01-18T00:00:00Z</td>
      <td>18.65</td>
      <td>21.47</td>
      <td>20.01</td>
      <td>20.01</td>
      <td>0.64</td>
      <td>None</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>17</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>8.87</td>
      <td>2022-01-19T00:00:00Z</td>
      <td>20.72</td>
      <td>23.72</td>
      <td>22.14</td>
      <td>22.16</td>
      <td>0.66</td>
      <td>None</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>18</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>8.87</td>
      <td>2022-01-20T00:00:00Z</td>
      <td>22.57</td>
      <td>25.73</td>
      <td>24.30</td>
      <td>24.32</td>
      <td>0.67</td>
      <td>None</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>19</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>8.87</td>
      <td>2022-01-21T00:00:00Z</td>
      <td>24.63</td>
      <td>27.97</td>
      <td>26.47</td>
      <td>26.49</td>
      <td>0.66</td>
      <td>None</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>20</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>8.87</td>
      <td>2022-01-22T00:00:00Z</td>
      <td>26.91</td>
      <td>29.95</td>
      <td>28.63</td>
      <td>28.66</td>
      <td>0.66</td>
      <td>None</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>21</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>8.87</td>
      <td>2022-01-23T00:00:00Z</td>
      <td>29.07</td>
      <td>32.18</td>
      <td>30.81</td>
      <td>30.82</td>
      <td>0.66</td>
      <td>None</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>22</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>8.87</td>
      <td>2022-01-24T00:00:00Z</td>
      <td>31.45</td>
      <td>34.35</td>
      <td>32.97</td>
      <td>32.99</td>
      <td>0.66</td>
      <td>None</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>23</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>8.87</td>
      <td>2022-01-25T00:00:00Z</td>
      <td>33.64</td>
      <td>36.56</td>
      <td>35.14</td>
      <td>35.17</td>
      <td>0.65</td>
      <td>None</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>24</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>8.87</td>
      <td>2022-01-26T00:00:00Z</td>
      <td>35.84</td>
      <td>38.64</td>
      <td>37.31</td>
      <td>37.34</td>
      <td>0.65</td>
      <td>None</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>25</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>3.67</td>
      <td>2022-01-27T00:00:00Z</td>
      <td>12.00</td>
      <td>39.92</td>
      <td>29.38</td>
      <td>38.68</td>
      <td>12.63</td>
      <td>None</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>26</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>0.02</td>
      <td>2022-01-28T00:00:00Z</td>
      <td>11.87</td>
      <td>14.59</td>
      <td>12.93</td>
      <td>12.91</td>
      <td>0.39</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>27</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>0.03</td>
      <td>2022-01-29T00:00:00Z</td>
      <td>11.81</td>
      <td>14.78</td>
      <td>12.94</td>
      <td>12.92</td>
      <td>0.39</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>28</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>0.03</td>
      <td>2022-01-30T00:00:00Z</td>
      <td>11.97</td>
      <td>14.54</td>
      <td>12.95</td>
      <td>12.94</td>
      <td>0.38</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>29</th>
      <td>None</td>
      <td>example assay</td>
      <td>output 0 0</td>
      <td>0.03</td>
      <td>2022-01-31T00:00:00Z</td>
      <td>11.93</td>
      <td>14.64</td>
      <td>12.94</td>
      <td>12.94</td>
      <td>0.39</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
  </tbody>
</table>
</div>

Basic functionality for creating quick charts is included.

```python
assay_results.chart_scores()
```

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_32_0.png)
    

We see that the difference scores are low for a while and then jump up to indicate there is an issue. We can examine that particular window to help us decide if that threshold is set correctly or not.

We can generate a quick chart of the results. This chart shows the 5 quantile bins (quintiles) derived from the baseline data plus one for left outliers and one for right outliers.  We also see that the data from the window falls within the baseline quintiles but in a different proportion and is skewing higher. Whether this is an issue or not is specific to your use case.

First let's examine a day that is only slightly different than the baseline. We see that we do see some values that fall outside of the range from the baseline values, the left and right outliers, and that the bin values are different but similar.

```python
assay_results[0].chart()
```

    baseline mean = 12.954393170120568
    window mean = 12.952570220492534
    baseline median = 12.913979530334473
    window median = 12.905640125274658
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.013551035434377596
    scores = [0.0006959467613300823, 0.0004941766212731371, 0.0003452027689633905, 0.0014095463411471284, 0.0007957390027837054, 7.341649894282799e-06, 0.00980308228898587]
    index = None

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_34_1.png)
    

Other days, however are significantly different.

```python
assay_results[12].chart()
```

    baseline mean = 12.954393170120568
    window mean = 13.018988957205092
    baseline median = 12.913979530334473
    window median = 12.995140552520752
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.1062324578888069
    scores = [0.0, 0.06790765360198812, 0.0003893727578237944, 0.0037302373887164895, 0.02434412838052893, 5.798347076369716e-05, 0.00980308228898587]
    index = None

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_36_1.png)
    

```python
assay_results[13].chart()
```

    baseline mean = 12.954393170120568
    window mean = 14.013120903347765
    baseline median = 12.913979530334473
    window median = 13.991220951080322
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 3.055506158911697
    scores = [0.0, 0.7203606043304971, 0.8049360069588025, 0.4504317335378006, 0.0820473282443674, 0.9698478211538909, 0.027882664686338928]
    index = None

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_37_1.png)
    

If we want to investigate further, we can run interactive assays on each of the inputs to see if any of them show anything abnormal. In this example we'll provide the feature labels to create more understandable titles.

The current assay expects continuous data. Sometimes categorical data is encoded as 1 or 0 in a feature and sometimes in a limited number of values such as 1, 2, 3. If one value has high a percentage the analysis emits a warning so that we know the scores for that feature may not behave as we expect.

```python
labels = ['bedrooms', 'bathrooms', 'lat', 'long', 'waterfront', 'sqft_living', 'sqft_lot', 'floors', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']

# get the raw inference logs
topic = wl.get_topic_name(pipeline.id())
baseline_inferences = wl.get_raw_pipeline_inference_logs(topic, baseline_start, baseline_end, model_name)

# feed the inference logs into the assay so we can examine the inputs
assay_results = assay_config.interactive_input_run(baseline_inferences, labels)
iadf = assay_results.to_dataframe()
```

    input column distinct_vals label           largest_pct
        0     0             12 bedrooms        0.4567 
        0     1             26 bathrooms       0.2421 
        0     2           1492 lat             0.0022 
        0     3            502 long            0.0077 
        0     4              2 waterfront      0.9928 *** May not be continuous feature
        0     5            425 sqft_living     0.0094 
        0     6           1407 sqft_lot        0.0188 
        0     7              6 floors          0.5036 
        0     8              5 view            0.9068 *** May not be continuous feature
        0     9              5 condition       0.6293 
        0    10             10 grade           0.4242 
        0    11            393 sqft_above      0.0121 
        0    12            162 sqft_basement   0.6034 
        0    13            116 yr_built        0.0237 
        0    14             37 yr_renovated    0.9597 *** May not be continuous feature
        0    15            340 sqft_living15   0.0132 
        0    16           1347 sqft_lot15      0.0177 

We can chart each of the iopaths and do a visual inspection. From the charts we see that if any of the input features had significant differences in the first two days which we can choose to inspect further. Here we choose to show 3 charts just to save space in this notebook.

```python
assay_results.chart_iopaths(labels=labels, selected_labels=['bedrooms', 'lat', 'sqft_living'])
```

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_41_0.png)
    

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_41_1.png)
    

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_41_2.png)
    

When we are comfortable with what alert threshold should be for our specific purposes we can create and save an assay that will be automatically run on a daily basis.

In this example we're create an assay that runs everyday against the baseline and has an alert threshold of 0.5.

Once we upload it it will be saved and scheduled for future data as well as run against past data.

```python
alert_threshold = 0.1
assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end).add_alert_threshold(alert_threshold)
assay_id = assay_builder.upload()
```

To retrieve the results after enough time has passed, collect them with the method `get_assay_results_logs` specifying the time to collect and the `assay_id`.  This will take additional inference history so can not be demonstrated here, but the command to collect the data is listed below.

```python
assay_results = wl.get_assay_results_logs(baseline_end,datetime.datetime.now(), assay_id=assay_id)
```

## Scheduling Assays

By default assays are scheduled to run **every 24 hours** starting immediately after the baseline period ends.

However, you can control the start time by setting `start` and the frequency by setting `interval` on the window.

So to recap:

* The window width is the **size** of the window. The default is 24 hours.
* The interval is:
  * How often the analysis is run.
  * How far the window is slid into the future based on the last run.
  * The default is the window width.
* The window start is when the analysis should start. The default is the end of the baseline period.

For example to run an analysis every 12 hours on the previous 24 hours of data, you'd set the window width to 24 (the default) and the interval to 12.

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

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_48_0.png)
    

To start a weekly analysis of the previous week on a specific day, set the start date (taking care to specify the desired timezone), and the width and interval to 1 week.  The analysis will be generated when the window is complete.

```python
report_start = datetime.datetime.fromisoformat('2022-01-03T00:00:00+00:00')

assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end)
assay_builder = assay_builder.add_run_until(last_day)

assay_builder.window_builder().add_width(weeks=1).add_interval(weeks=1).add_start(report_start)

assay_config = assay_builder.build()

assay_results = assay_config.interactive_run()
print(f"Generated {len(assay_results)} analyses")
```

    Generated 4 analyses

```python
assay_results.chart_scores()
```

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_51_0.png)
    

## Advanced Configuration

The assay can be configured in a variety of ways to help customize it to your particular needs. Specifically you can:
* change the `BinMode` to evenly spaced, quantile or user provided
* change the number of bins to use
* provide weights to use when scoring the bins
* calculate the score using the sum of differences, maximum difference or population stability index
* change the value aggregation for the bins to density, cumulative or edges

Let's take a look at these in turn.

### Default configuration

First let's look at the default configuration. This is a lot of information but much of it is useful to know where it is available.

We see that the assay is broken up into 4 sections: 
  
* Top level meta data section
* Baseline specification
* Window specification
* The summarization configuration.

In the meta section we see the name of the assay, that it runs on the first column of the first output `"outputs 0 0"` and that there is a default threshold of 0.25.

The summarizer section shows us the defaults of Quantile, Density and PSI on 5 bins.

The baseline section shows us that it is configured as a fixed baseline with the specified start and end date times.

And the window tells us what model in the pipeline we are analyzing and how often.

```python
assay_builder = wl.build_assay("Test Assay", pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
print(assay_builder.build().to_json())
```

    {
        "name": "Test Assay",
        "pipeline_id": 4,
        "pipeline_name": "housepricepipe",
        "active": true,
        "status": "created",
        "iopath": "output 0 0",
        "baseline": {
            "Fixed": {
                "pipeline": "housepricepipe",
                "model": "housepricemodel",
                "start_at": "2022-01-01T00:00:00+00:00",
                "end_at": "2022-01-02T00:00:00+00:00"
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
        "run_until": "2022-02-01T00:00:00+00:00",
        "workspace_id": 4,
        "model_insights_url": "http://model-insights:5150"
    }

## Defaults

We can run the assay interactively and review the first analysis. The method `compare_basic_stats` gives us a dataframe with basic stats for the baseline and window data.

```python
assay_results = assay_builder.build().interactive_run()
ar = assay_results[0]

ar.compare_basic_stats()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>1,813.00</td>
      <td>1,812.00</td>
      <td>-1.00</td>
      <td>-0.06</td>
    </tr>
    <tr>
      <th>min</th>
      <td>11.95</td>
      <td>11.62</td>
      <td>-0.33</td>
      <td>-2.72</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.08</td>
      <td>15.11</td>
      <td>0.03</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.95</td>
      <td>12.95</td>
      <td>-0.00</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>median</th>
      <td>12.91</td>
      <td>12.91</td>
      <td>-0.01</td>
      <td>-0.06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.46</td>
      <td>0.45</td>
      <td>-0.01</td>
      <td>-2.75</td>
    </tr>
    <tr>
      <th>start</th>
      <td>2022-01-01T00:00:00Z</td>
      <td>2022-01-02T00:00:00Z</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>end</th>
      <td>2022-01-02T00:00:00Z</td>
      <td>2022-01-03T00:00:00Z</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

The method `compare_bins` gives us a dataframe with the bin information. Such as the number of bins, the right edges, suggested bin/edge names and the values for each bin in the baseline and the window.

```python
ar.compare_bins()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>w_aggregation</th>
      <th>diff_in_pcts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11.95</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>Density</td>
      <td>11.95</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>Density</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.56</td>
      <td>q_20</td>
      <td>0.20</td>
      <td>Density</td>
      <td>12.56</td>
      <td>e_1.26e1</td>
      <td>0.19</td>
      <td>Density</td>
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
      <td>Density</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13.01</td>
      <td>q_60</td>
      <td>0.20</td>
      <td>Density</td>
      <td>13.01</td>
      <td>e_1.30e1</td>
      <td>0.18</td>
      <td>Density</td>
      <td>-0.02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.31</td>
      <td>q_80</td>
      <td>0.20</td>
      <td>Density</td>
      <td>13.31</td>
      <td>e_1.33e1</td>
      <td>0.21</td>
      <td>Density</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15.08</td>
      <td>q_100</td>
      <td>0.20</td>
      <td>Density</td>
      <td>15.08</td>
      <td>e_1.51e1</td>
      <td>0.20</td>
      <td>Density</td>
      <td>0.00</td>
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
      <td>Density</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>

We can also plot the chart to visualize the values of the bins.

```python
ar.chart()
```

    baseline mean = 12.954393170120568
    window mean = 12.952570220492534
    baseline median = 12.913979530334473
    window median = 12.905640125274658
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.013551035434377596
    scores = [0.0006959467613300823, 0.0004941766212731371, 0.0003452027689633905, 0.0014095463411471284, 0.0007957390027837054, 7.341649894282799e-06, 0.00980308228898587]
    index = None

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_59_1.png)
    

## Binning Mode

We can change the bin mode algorithm to equal and see that the bins/edges are partitioned at different points and the bins have different values.

```python
assay_builder = wl.build_assay("Test Assay", pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_bin_mode(BinMode.EQUAL)
assay_results = assay_builder.build().interactive_run()
display(display(assay_results[0].compare_bins()))
assay_results[0].chart()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>w_aggregation</th>
      <th>diff_in_pcts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11.95</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>Density</td>
      <td>11.95</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>Density</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.57</td>
      <td>p_1.26e1</td>
      <td>0.21</td>
      <td>Density</td>
      <td>12.57</td>
      <td>e_1.26e1</td>
      <td>0.20</td>
      <td>Density</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.20</td>
      <td>p_1.32e1</td>
      <td>0.54</td>
      <td>Density</td>
      <td>13.20</td>
      <td>e_1.32e1</td>
      <td>0.53</td>
      <td>Density</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13.83</td>
      <td>p_1.38e1</td>
      <td>0.21</td>
      <td>Density</td>
      <td>13.83</td>
      <td>e_1.38e1</td>
      <td>0.24</td>
      <td>Density</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.45</td>
      <td>p_1.45e1</td>
      <td>0.04</td>
      <td>Density</td>
      <td>14.45</td>
      <td>e_1.45e1</td>
      <td>0.03</td>
      <td>Density</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15.08</td>
      <td>p_1.51e1</td>
      <td>0.00</td>
      <td>Density</td>
      <td>15.08</td>
      <td>e_1.51e1</td>
      <td>0.00</td>
      <td>Density</td>
      <td>-0.00</td>
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
      <td>Density</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>

    None

    baseline mean = 12.954393170120568
    window mean = 12.952570220492534
    baseline median = 12.913979530334473
    window median = 12.905640125274658
    bin_mode = Equal
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.016463316496701866
    scores = [0.0006959467613300823, 0.00028622745636607417, 0.000136940329536975, 0.0024190313632530313, 0.0028459952590805006, 0.0002760930381493355, 0.00980308228898587]
    index = None

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_61_3.png)
    

## User Provided Bin Edges

The values in this dataset run from ~11.6 to ~15.81. And let's say we had a business reason to use specific bin edges.  We can specify them with the BinMode.PROVIDED and specifying a list of floats with the right hand / upper edge of each bin and optionally the lower edge of the smallest bin. If the lowest edge is not specified the threshold for left outliers is taken from the smallest value in the baseline dataset.

```python
edges = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
assay_builder = wl.build_assay("Test Assay", pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_bin_mode(BinMode.PROVIDED, edges)
assay_results = assay_builder.build().interactive_run()
display(display(assay_results[0].compare_bins()))
assay_results[0].chart()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>w_aggregation</th>
      <th>diff_in_pcts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11.00</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>Density</td>
      <td>11.00</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>Density</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.00</td>
      <td>e_1.20e1</td>
      <td>0.00</td>
      <td>Density</td>
      <td>12.00</td>
      <td>e_1.20e1</td>
      <td>0.00</td>
      <td>Density</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.00</td>
      <td>e_1.30e1</td>
      <td>0.59</td>
      <td>Density</td>
      <td>13.00</td>
      <td>e_1.30e1</td>
      <td>0.58</td>
      <td>Density</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.00</td>
      <td>e_1.40e1</td>
      <td>0.39</td>
      <td>Density</td>
      <td>14.00</td>
      <td>e_1.40e1</td>
      <td>0.40</td>
      <td>Density</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15.00</td>
      <td>e_1.50e1</td>
      <td>0.02</td>
      <td>Density</td>
      <td>15.00</td>
      <td>e_1.50e1</td>
      <td>0.02</td>
      <td>Density</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>16.00</td>
      <td>e_1.60e1</td>
      <td>0.00</td>
      <td>Density</td>
      <td>16.00</td>
      <td>e_1.60e1</td>
      <td>0.00</td>
      <td>Density</td>
      <td>0.00</td>
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
      <td>Density</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>

    None

    baseline mean = 12.954393170120568
    window mean = 12.952570220492534
    baseline median = 12.913979530334473
    window median = 12.905640125274658
    bin_mode = Provided
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.005831639113611392
    scores = [0.0, 0.002708901099649454, 0.00015914496208737885, 0.0004215024577886459, 0.002159043392325224, 0.00038304720176068804, 0.0]
    index = None

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_63_3.png)
    

## Number of Bins

We could also choose to a different number of bins, let's say 10, which can be evenly spaced or based on the quantiles (deciles).

```python
assay_builder = wl.build_assay("Test Assay", pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_bin_mode(BinMode.QUANTILE).add_num_bins(10)
assay_results = assay_builder.build().interactive_run()
display(display(assay_results[1].compare_bins()))
assay_results[1].chart()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>w_aggregation</th>
      <th>diff_in_pcts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11.95</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>Density</td>
      <td>11.95</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>Density</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.40</td>
      <td>q_10</td>
      <td>0.10</td>
      <td>Density</td>
      <td>12.40</td>
      <td>e_1.24e1</td>
      <td>0.10</td>
      <td>Density</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.56</td>
      <td>q_20</td>
      <td>0.10</td>
      <td>Density</td>
      <td>12.56</td>
      <td>e_1.26e1</td>
      <td>0.09</td>
      <td>Density</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.70</td>
      <td>q_30</td>
      <td>0.10</td>
      <td>Density</td>
      <td>12.70</td>
      <td>e_1.27e1</td>
      <td>0.09</td>
      <td>Density</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.81</td>
      <td>q_40</td>
      <td>0.10</td>
      <td>Density</td>
      <td>12.81</td>
      <td>e_1.28e1</td>
      <td>0.10</td>
      <td>Density</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12.91</td>
      <td>q_50</td>
      <td>0.10</td>
      <td>Density</td>
      <td>12.91</td>
      <td>e_1.29e1</td>
      <td>0.12</td>
      <td>Density</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>6</th>
      <td>13.01</td>
      <td>q_60</td>
      <td>0.10</td>
      <td>Density</td>
      <td>13.01</td>
      <td>e_1.30e1</td>
      <td>0.08</td>
      <td>Density</td>
      <td>-0.02</td>
    </tr>
    <tr>
      <th>7</th>
      <td>13.15</td>
      <td>q_70</td>
      <td>0.10</td>
      <td>Density</td>
      <td>13.15</td>
      <td>e_1.31e1</td>
      <td>0.12</td>
      <td>Density</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>8</th>
      <td>13.31</td>
      <td>q_80</td>
      <td>0.10</td>
      <td>Density</td>
      <td>13.31</td>
      <td>e_1.33e1</td>
      <td>0.09</td>
      <td>Density</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>9</th>
      <td>13.56</td>
      <td>q_90</td>
      <td>0.10</td>
      <td>Density</td>
      <td>13.56</td>
      <td>e_1.36e1</td>
      <td>0.11</td>
      <td>Density</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>10</th>
      <td>15.08</td>
      <td>q_100</td>
      <td>0.10</td>
      <td>Density</td>
      <td>15.08</td>
      <td>e_1.51e1</td>
      <td>0.09</td>
      <td>Density</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>11</th>
      <td>NaN</td>
      <td>right_outlier</td>
      <td>0.00</td>
      <td>Density</td>
      <td>NaN</td>
      <td>right_outlier</td>
      <td>0.00</td>
      <td>Density</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>

    None

    baseline mean = 12.954393170120568
    window mean = 12.94535461693147
    baseline median = 12.913979530334473
    window median = 12.903773307800293
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.021364617672033626
    scores = [0.0013318933239185415, 0.0001508387888967812, 0.0014077319940240033, 0.00044689056669365687, 0.0001508387888967812, 0.002879132738274895, 0.002579185308688176, 0.002722796821458902, 0.0011510010089298668, 0.0009475972030849906, 0.001710564941068633, 0.005886146188098397]
    index = None

    /opt/conda/lib/python3.9/site-packages/wallaroo/assay.py:318: UserWarning: FixedFormatter should only be used together with FixedLocator
      ax.set_xticklabels(labels=edge_names, rotation=45)

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_65_4.png)
    

## Bin Weights

Now let's say we only care about differences at the higher end of the range. We can use weights to specify that difference in the lower bins should not be counted in the score. 

If we stick with 10 bins we can provide 10 a vector of 12 weights. One weight each for the original bins plus one at the front for the left outlier bin and one at the end for the right outlier bin.

Note we still show the values for the bins but the scores for the lower 5 and left outlier are 0 and only the right half is counted and reflected in the score.

```python
weights = [0] * 6
weights.extend([1] * 6)
print("Using weights: ", weights)
assay_builder = wl.build_assay("Test Assay", pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_bin_mode(BinMode.QUANTILE).add_num_bins(10).add_bin_weights(weights)
assay_results = assay_builder.build().interactive_run()
display(display(assay_results[1].compare_bins()))
assay_results[1].chart()
```

    Using weights:  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>w_aggregation</th>
      <th>diff_in_pcts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11.95</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>Density</td>
      <td>11.95</td>
      <td>left_outlier</td>
      <td>0.00</td>
      <td>Density</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.40</td>
      <td>q_10</td>
      <td>0.10</td>
      <td>Density</td>
      <td>12.40</td>
      <td>e_1.24e1</td>
      <td>0.10</td>
      <td>Density</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.56</td>
      <td>q_20</td>
      <td>0.10</td>
      <td>Density</td>
      <td>12.56</td>
      <td>e_1.26e1</td>
      <td>0.09</td>
      <td>Density</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.70</td>
      <td>q_30</td>
      <td>0.10</td>
      <td>Density</td>
      <td>12.70</td>
      <td>e_1.27e1</td>
      <td>0.09</td>
      <td>Density</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.81</td>
      <td>q_40</td>
      <td>0.10</td>
      <td>Density</td>
      <td>12.81</td>
      <td>e_1.28e1</td>
      <td>0.10</td>
      <td>Density</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12.91</td>
      <td>q_50</td>
      <td>0.10</td>
      <td>Density</td>
      <td>12.91</td>
      <td>e_1.29e1</td>
      <td>0.12</td>
      <td>Density</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>6</th>
      <td>13.01</td>
      <td>q_60</td>
      <td>0.10</td>
      <td>Density</td>
      <td>13.01</td>
      <td>e_1.30e1</td>
      <td>0.08</td>
      <td>Density</td>
      <td>-0.02</td>
    </tr>
    <tr>
      <th>7</th>
      <td>13.15</td>
      <td>q_70</td>
      <td>0.10</td>
      <td>Density</td>
      <td>13.15</td>
      <td>e_1.31e1</td>
      <td>0.12</td>
      <td>Density</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>8</th>
      <td>13.31</td>
      <td>q_80</td>
      <td>0.10</td>
      <td>Density</td>
      <td>13.31</td>
      <td>e_1.33e1</td>
      <td>0.09</td>
      <td>Density</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>9</th>
      <td>13.56</td>
      <td>q_90</td>
      <td>0.10</td>
      <td>Density</td>
      <td>13.56</td>
      <td>e_1.36e1</td>
      <td>0.11</td>
      <td>Density</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>10</th>
      <td>15.08</td>
      <td>q_100</td>
      <td>0.10</td>
      <td>Density</td>
      <td>15.08</td>
      <td>e_1.51e1</td>
      <td>0.09</td>
      <td>Density</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>11</th>
      <td>NaN</td>
      <td>right_outlier</td>
      <td>0.00</td>
      <td>Density</td>
      <td>NaN</td>
      <td>right_outlier</td>
      <td>0.00</td>
      <td>Density</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>

    None

    baseline mean = 12.954393170120568
    window mean = 12.94535461693147
    baseline median = 12.913979530334473
    window median = 12.903773307800293
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = True
    score = 0.0024995485785548276
    scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000429864218114696, 0.00045379947024315036, 0.00019183350148831114, 0.00015793286718083176, 0.0002850941568447722, 0.0009810243646830661]
    index = None

    /opt/conda/lib/python3.9/site-packages/wallaroo/assay.py:318: UserWarning: FixedFormatter should only be used together with FixedLocator
      ax.set_xticklabels(labels=edge_names, rotation=45)

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_67_5.png)
    

## Metrics

The `score` is a distance or dis-similarity measure. The larger it is the less similar the two distributions are. We currently support
summing the differences of each individual bin, taking the maximum difference and a modified Population Stability Index (PSI).

The following three charts use each of the metrics. Note how the scores change. The best one will depend on your particular use case.

```python
assay_builder = wl.build_assay("Test Assay", pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_results = assay_builder.build().interactive_run()
assay_results[0].chart()
```

    baseline mean = 12.954393170120568
    window mean = 12.952570220492534
    baseline median = 12.913979530334473
    window median = 12.905640125274658
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.013551035434377596
    scores = [0.0006959467613300823, 0.0004941766212731371, 0.0003452027689633905, 0.0014095463411471284, 0.0007957390027837054, 7.341649894282799e-06, 0.00980308228898587]
    index = None

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_69_1.png)
    

```python
assay_builder = wl.build_assay("Test Assay", pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_metric(Metric.SUMDIFF)
assay_results = assay_builder.build().interactive_run()
assay_results[0].chart()
```

    baseline mean = 12.954393170120568
    window mean = 12.952570220492534
    baseline median = 12.913979530334473
    window median = 12.905640125274658
    bin_mode = Quantile
    aggregation = Density
    metric = SumDiff
    weighted = False
    score = 0.02626907215365116
    scores = [0.0033112582781456954, 0.009823277798679891, 0.008388338331573902, 0.016445794354971288, 0.012803349369101491, 0.001214249795139094, 0.0005518763796909492]
    index = None

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_70_1.png)
    

```python
assay_builder = wl.build_assay("Test Assay", pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_metric(Metric.MAXDIFF)
assay_results = assay_builder.build().interactive_run()
assay_results[0].chart()
```

    baseline mean = 12.954393170120568
    window mean = 12.952570220492534
    baseline median = 12.913979530334473
    window median = 12.905640125274658
    bin_mode = Quantile
    aggregation = Density
    metric = MaxDiff
    weighted = False
    score = 0.016445794354971288
    scores = [0.0033112582781456954, 0.009823277798679891, 0.008388338331573902, 0.016445794354971288, 0.012803349369101491, 0.001214249795139094, 0.0005518763796909492]
    index = 3

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_71_1.png)
    

## Aggregation Options

Also, bin aggregation can be done in histogram `Aggregation.DENSITY` style (the default) where we count the number/percentage of values that fall in each bin or Empirical Cumulative Density Function style `Aggregation.CUMULATIVE` where we keep a cumulative count of the values/percentages that fall in each bin.

```python
assay_builder = wl.build_assay("Test Assay", pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_aggregation(Aggregation.DENSITY)
assay_results = assay_builder.build().interactive_run()
assay_results[0].chart()
```

    baseline mean = 12.954393170120568
    window mean = 12.952570220492534
    baseline median = 12.913979530334473
    window median = 12.905640125274658
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.013551035434377596
    scores = [0.0006959467613300823, 0.0004941766212731371, 0.0003452027689633905, 0.0014095463411471284, 0.0007957390027837054, 7.341649894282799e-06, 0.00980308228898587]
    index = None

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_73_1.png)
    

```python
assay_builder = wl.build_assay("Test Assay", pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_aggregation(Aggregation.CUMULATIVE)
assay_results = assay_builder.build().interactive_run()
assay_results[0].chart()
```

    baseline mean = 12.954393170120568
    window mean = 12.952570220492534
    baseline median = 12.913979530334473
    window median = 12.905640125274658
    bin_mode = Quantile
    aggregation = Cumulative
    metric = PSI
    weighted = False
    score = 0.028587074708172105
    scores = [0.0033112582781456954, 0.006512019520534207, 0.0018763188110397233, 0.01456947554393151, 0.0017661261748300738, 0.0005518763796908965, 0.0]
    index = None

    
![png](/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files/wallaroo-model-insights-reference_74_1.png)
    

