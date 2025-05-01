# Checklist

[ ] Remove spaces - regex `\n\n\n` with `\n\n`
[ ] Image paths - Remove all `(./images` - replace with `(/images`.
[ ] Replace `docs/markdown/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference.md` `(wallaroo-model-insights-reference_files` with `(/images/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-insights-reference_files`.
[ ] Remove:
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
```
    With:

{{<table "table table-bordered">}}
<table>

    Remove:

</table>
{{</table>}}
    

[ ] Verify images are replaced into /images where needed.
[ ] Notebooks in prod:  Look for `/opt/conda/lib` and `<style>` to remove them.
[ ] Notebooks in prod:  Replace `(01_notebooks-in-prod_explore_and_train-reference_files` with `(/images/wallaroo-tutorials/notebooks-in-prod/01_notebooks-in-prod_explore_and_train-reference_files`
[ ] Notebooks in prod: Replace `(02_notebooks-in-prod_automated_training_process-reference_files` with `(/images/wallaroo-tutorials/notebooks-in-prod/02_notebooks-in-prod_automated_training_process-reference_files`
[ ] Replace:

Check notebooks in prod - something still fishy there.

anomaly detection - error there.  

Make sure the assays demo is fully run.


</table> with:

</table>
{{</table>}}