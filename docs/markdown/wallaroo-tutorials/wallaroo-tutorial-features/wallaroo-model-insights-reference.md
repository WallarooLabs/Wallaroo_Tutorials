This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/model_insights).

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

assay_name = "example assay jch"
```

### Connect to Wallaroo

Connect to your Wallaroo instance.

```python
# Login through local Wallaroo instance

# wl = wallaroo.Client()

# SSO login through keycloak

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR SUFFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")
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

We assume the pipeline has been running for a while and there is a period of time that is free of errors that we'd like to use as the _baseline_. Let's note the start and end times. For this example we have 30 days of data from Jan 2022 and will use Jan 1 data as our baseline.

```python
import datetime
baseline_start = datetime.datetime.fromisoformat('2022-01-01T00:00:00+00:00')
baseline_end = datetime.datetime.fromisoformat('2022-01-02T00:00:00+00:00')
last_day = datetime.datetime.fromisoformat('2022-02-01T00:00:00+00:00')
```

Let's create an assay using that pipeline and the model in the pipeline. We also specify the baseline start and end.

```python

assay_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end)
```

We don't know much about our baseline data yet so let's examine the data and create a couple of visual representations. First let's get some basic stats on the baseline data.

```python
baseline_run = assay_builder.build().interactive_baseline_run()
baseline_run.baseline_stats()
```

Now let's look at a histogram, kernel density estimate (KDE), and Empirical Cumulative Distribution (ecdf) charts of the baseline data. These will give us insights into the distributions of the predictions and features that the assay is configured for.

```python
assay_builder.baseline_histogram()
```

```python
assay_builder.baseline_kde()
```

```python
assay_builder.baseline_ecdf()
```

### Interactive Baseline Runs
We can do an interactive run of just the baseline part to see how the baseline data will be put into bins. This assay uses quintiles so all 5 bins (not counting the outlier bins) have 20% of the predictions. We can see the bin boundaries along the x-axis.

```python
baseline_run.chart()
```

We can also get a dataframe with the bin/edge information.

```python
baseline_run.baseline_bins()
```

The previous assay used quintiles so all of the bins had the same percentage/count of samples.  To get bins that are divided equally along the range of values we can use `BinMode.EQUAL`.

```python
equal_bin_builder = wl.build_assay(assay_name, pipeline, model_name, baseline_start, baseline_end)
equal_bin_builder.summarizer_builder.add_bin_mode(BinMode.EQUAL)
equal_baseline = equal_bin_builder.build().interactive_baseline_run()
equal_baseline.chart()
```

We now see very different bin edges and sample percentages per bin.

```python
equal_baseline.baseline_bins()
```

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

Basic functionality for creating quick charts is included.

```python
assay_results.chart_scores()
```

We see that the difference scores are low for a while and then jump up to indicate there is an issue. We can examine that particular window to help us decide if that threshold is set correctly or not.

We can generate a quick chart of the results. This chart shows the 5 quantile bins (quintiles) derived from the baseline data plus one for left outliers and one for right outliers.  We also see that the data from the window falls within the baseline quintiles but in a different proportion and is skewing higher. Whether this is an issue or not is specific to your use case.

First let's examine a day that is only slightly different than the baseline. We see that we do see some values that fall outside of the range from the baseline values, the left and right outliers, and that the bin values are different but similar.

```python
assay_results[0].chart()
```

Other days, however are significantly different.

```python
assay_results[12].chart()
```

```python
assay_results[13].chart()
```

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

We can chart each of the iopaths and do a visual inspection. From the charts we see that if any of the input features had significant differences in the first two days which we can choose to inspect further. Here we choose to show 3 charts just to save space in this notebook.

```python
assay_results.chart_iopaths(labels=labels, selected_labels=['bedrooms', 'lat', 'sqft_living'])
```

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

```python
assay_results.chart_scores()
```

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

```python
assay_results.chart_scores()
```

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

## Defaults

We can run the assay interactively and review the first analysis. The method `compare_basic_stats` gives us a dataframe with basic stats for the baseline and window data.

```python
assay_results = assay_builder.build().interactive_run()
ar = assay_results[0]

ar.compare_basic_stats()
```

The method `compare_bins` gives us a dataframe with the bin information. Such as the number of bins, the right edges, suggested bin/edge names and the values for each bin in the baseline and the window.

```python
ar.compare_bins()
```

We can also plot the chart to visualize the values of the bins.

```python
ar.chart()
```

## Binning Mode

We can change the bin mode algorithm to equal and see that the bins/edges are partitioned at different points and the bins have different values.

```python
assay_builder = wl.build_assay("Test Assay", pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_bin_mode(BinMode.EQUAL)
assay_results = assay_builder.build().interactive_run()
display(display(assay_results[0].compare_bins()))
assay_results[0].chart()
```

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

## Number of Bins

We could also choose to a different number of bins, let's say 10, which can be evenly spaced or based on the quantiles (deciles).

```python
assay_builder = wl.build_assay("Test Assay", pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_bin_mode(BinMode.QUANTILE).add_num_bins(10)
assay_results = assay_builder.build().interactive_run()
display(display(assay_results[1].compare_bins()))
assay_results[1].chart()
```

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

## Metrics

The `score` is a distance or dis-similarity measure. The larger it is the less similar the two distributions are. We currently support
summing the differences of each individual bin, taking the maximum difference and a modified Population Stability Index (PSI).

The following three charts use each of the metrics. Note how the scores change. The best one will depend on your particular use case.

```python
assay_builder = wl.build_assay("Test Assay", pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_results = assay_builder.build().interactive_run()
assay_results[0].chart()
```

```python
assay_builder = wl.build_assay("Test Assay", pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_metric(Metric.SUMDIFF)
assay_results = assay_builder.build().interactive_run()
assay_results[0].chart()
```

```python
assay_builder = wl.build_assay("Test Assay", pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_metric(Metric.MAXDIFF)
assay_results = assay_builder.build().interactive_run()
assay_results[0].chart()
```

## Aggregation Options

Also, bin aggregation can be done in histogram `Aggregation.DENSITY` style (the default) where we count the number/percentage of values that fall in each bin or Empirical Cumulative Density Function style `Aggregation.CUMULATIVE` where we keep a cumulative count of the values/percentages that fall in each bin.

```python
assay_builder = wl.build_assay("Test Assay", pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_aggregation(Aggregation.DENSITY)
assay_results = assay_builder.build().interactive_run()
assay_results[0].chart()
```

```python
assay_builder = wl.build_assay("Test Assay", pipeline, model_name, baseline_start, baseline_end).add_run_until(last_day)
assay_builder.summarizer_builder.add_aggregation(Aggregation.CUMULATIVE)
assay_results = assay_builder.build().interactive_run()
assay_results[0].chart()
```
