# import libraries
from pathlib import Path
from mac.config.inference import CustomInferenceConfig
from byop.byop import BYOPInferenceBuilder
import pandas as pd
import pyarrow as pa

# create the input dataframe and convert to dictionary for testing
input_df = pd.DataFrame({"input_number": [1,2,3],
                             "id": [20000000004093819,20012684296980773,481562342]
                            })

input_dictionary = {
        col: input_df[col].to_numpy() for col in input_df.columns
    }

print(input_dictionary)


# prepare the BYOP and import any modules
builder = BYOPInferenceBuilder()
config = CustomInferenceConfig(
    framework="custom", 
    model_path=Path("./byop/"), modules_to_include={"*.py"}
)

# create the BYOP object
inference = builder.create(config)

# run a simulated inference
results = inference.predict(input_data=input_dictionary)
print(results)

# Schema Generation

# convert results into a dataframe
results_df = pd.DataFrame({
    key : value.tolist() for key, value in results.items()
    })
input_schema = pa.Schema.from_pandas(input_df).remove_metadata()
output_schema = pa.Schema.from_pandas(results_df).remove_metadata()
print(input_schema)
print(output_schema)

