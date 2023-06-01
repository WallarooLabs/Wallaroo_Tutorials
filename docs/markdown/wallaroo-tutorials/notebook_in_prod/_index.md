# Using Jupyter Notebooks in Production

The following tutorials are available from the [Wallaroo Tutorials Repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/notebooks_in_prod).

The following tutorials provide an example of an organization moving from experimentation to deployment in production using Jupyter Notebooks as the basis for code research and use.  For this example, we can assume to main actors performing the following tasks.

| Number | Notebook Sample | Task | Actor | Description |
|---|---|---|---|---|
|01| `01_explore_and_train.ipynb` | Data Exploration and Model Selection | Data Scientist | The data scientist evaluates the data and determines the best model to use to solve the proposed problems. |
|02| `02_automated_training_process.ipynd` | Training Process Automation Setup | Data Scientist | The data scientist has selected the model and tested how to train it.  In this phase, the data scientist tests automating the training process based on a data store. |
|03| `03_deploy_model.ipynb` | Deploy the Model in Wallaroo | MLOps Engineer | The MLOps takes the trained model and deploys a Wallaroo pipeline with it to perform inferences on by feeding it data from a data store. |
|04| `04_regular_batch_inferences.ipynb` | Regular Batch Inference | MLOps Engineer | With the pipeline deployed, regular inferences can be made and the results reported to a data store. |

Each Jupyter Notebook is arranged to demonstrate each step of the process.

## Resources

The following resources are provided as part of this tutorial:

* **data**
  * `data/seattle_housing_col_description.txt`: Describes the columns used as part data analysis.
  * `data/seattle_housing.csv`: Sample data of the Seattle, Washington housing market between 2014 and 2015.
* **code**
  * `postprocess.py`: Formats the data after inference by the model is complete.
  * `preprocess.py`: Formats the incoming data for the model.
  * `simdb.py`: A simulated database to demonstrate sending and receiving queries.
  * `wallaroo_client.py`: Additional methods used with the Wallaroo instance to create workspaces, etc.
