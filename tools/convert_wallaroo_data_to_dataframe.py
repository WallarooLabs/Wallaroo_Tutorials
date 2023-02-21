import pandas as pd
import pyarrow as pa
import json

# setting arrow on or off
#import os
#os.environ["ARROW_ENABLED"]="False"

# List of files to convert
jsonFileList = [
    # {
    #     "inputFile": "wallaroo-101/smoke_test.json",
    #     "dataframeOutputFile": "wallaroo-101/smoke_test.df.json",
    #     "arrowOutputFile": "wallaroo-101/smoke_test.arrow"
    # },
    # {
    #     "inputFile": "wallaroo-101/high_fraud.json",
    #     "dataframeOutputFile": "wallaroo-101/high_fraud.df.json",
    #     "arrowOutputFile": "wallaroo-101/high_fraud.arrow"
    # },
    # {
    #     "inputFile": "wallaroo-101/cc_data_1k.json",
    #     "dataframeOutputFile": "wallaroo-101/cc_data_1k.df.json",
    #     "arrowOutputFile": "wallaroo-101/cc_data_1k.arrow"
    # },
    # {
    #     "inputFile": "wallaroo-101/cc_data_10k.json",
    #     "dataframeOutputFile": "wallaroo-101/cc_data_10k.df.json",
    #     "arrowOutputFile": "wallaroo-101/cc_data_10k.arrow"
    # },
    # {
    #     "inputFile": "wallaroo-model-cookbooks/aloha/data_1.json",
    #     "dataframeOutputFile": "wallaroo-model-cookbooks/aloha/data_1.df.json",
    #     "arrowOutputFile": "wallaroo-model-cookbooks/aloha/data_1.arrow"
    # },
    # {
    #     "inputFile": "wallaroo-model-cookbooks/aloha/data_1k.json",
    #     "dataframeOutputFile": "wallaroo-model-cookbooks/aloha/data_1k.df.json",
    #     "arrowOutputFile": "wallaroo-model-cookbooks/aloha/data_1k.arrow"
    # },
    # {
    #     "inputFile": "wallaroo-model-cookbooks/aloha/data_25k.json",
    #     "dataframeOutputFile": "wallaroo-model-cookbooks/aloha/data_25k.df.json",
    #     "arrowOutputFile": "wallaroo-model-cookbooks/aloha/data_25k.arrow"
    # },
    # {
    #     "inputFile": "wallaroo-model-cookbooks/imdb/data/singleton.json",
    #     "dataframeOutputFile": "wallaroo-model-cookbooks/imdb/data/singleton.df.json",
    #     "arrowOutputFile": "wallaroo-model-cookbooks/imdb/data/singleton.arrow"
    # },
    # {
    #     "inputFile": "wallaroo-model-cookbooks/imdb/data/test_data.json",
    #     "dataframeOutputFile": "wallaroo-model-cookbooks/imdb/data/test_data.df.json",
    #     "arrowOutputFile": "wallaroo-model-cookbooks/imdb/data/test_data.arrow"
    # },
    # {
    #     "inputFile": "wallaroo-model-cookbooks/imdb/data/test_data_50K.json",
    #     "dataframeOutputFile": "wallaroo-model-cookbooks/imdb/data/test_data_50K.df.json",
    #     "arrowOutputFile": "wallaroo-model-cookbooks/imdb/data/test_data_50K.arrow"
    # },
    # {
    #     "inputFile": "wallaroo-testing-tutorials/abtesting/data/data-1.json",
    #     "dataframeOutputFile": "wallaroo-testing-tutorials/abtesting/data/data-1.df.json",
    #     "arrowOutputFile": "wallaroo-testing-tutorials/abtesting/data/data-1.arrow"
    # },
    # {
    #     "inputFile": "wallaroo-testing-tutorials/abtesting/data/data-1k.json",
    #     "dataframeOutputFile": "wallaroo-testing-tutorials/abtesting/data/data-1k.df.json",
    #     "arrowOutputFile": "wallaroo-testing-tutorials/abtesting/data/data-1k.arrow"
    # },
    # {
    #     "inputFile": "wallaroo-testing-tutorials/abtesting/data/data-25k.json",
    #     "dataframeOutputFile": "wallaroo-testing-tutorials/abtesting/data/data-25k.df.json",
    #     "arrowOutputFile": "wallaroo-testing-tutorials/abtesting/data/data-25k.arrow"
    # }
    # {
    #     "inputFile": "wallaroo-101/data/cc_data_1k.json",
    #     "dataframeOutputFile": "wallaroo-101/data/cc_data_1k.df.json",
    #     "arrowOutputFile": "wallaroo-101/data/cc_data_1k.arrow"
    # },
    # {
    #     "inputFile": "wallaroo-101/data/high_fraud.json",
    #     "dataframeOutputFile": "wallaroo-101/data/high_fraud.df.json",
    #     "arrowOutputFile": "wallaroo-101/data/high_fraud.arrow"
    # },
    {
        "inputFile": "model_conversion/sklearn-classification-to-onnx/isolet_test_data.json",
        "dataframeOutputFile": "model_conversion/sklearn-classification-to-onnx/isolet_test_data.df.json",
        "arrowOutputFile": "model_conversion/sklearn-classification-to-onnx/isolet_test_data.arrow"
    },
    # {
    #     "inputFile": "wallaroo-101/data/cc_data_10k.json",
    #     "dataframeOutputFile": "wallaroo-101/data/cc_data_10k.df.json",
    #     "arrowOutputFile": "wallaroo-101/data/cc_data_10k.arrow"
    # }
   
]

def ConvertJSONtoDataframe(inputFile, outputDataFrameFile):
    # read the file and convert it to a dataframe
    data = pd.read_json(inputFile, orient="records")
    # data = pd.read_json(inputFile)
    resultJson = pd.DataFrame.to_json(data, indent=4, orient="records")
    with open(outputDataFrameFile, "w") as outfile:
        outfile.write(resultJson)
    # return the dataframe as the result
    return data

# input is a dataframe
def ConvertJSONtoArrow(inputFile, outputArrowFile):
    # starting with the original input file
    data =  pd.read_json(inputFile, orient="records")
    data_table = pa.Table.from_pandas(data)
    data_schema = pa.Schema.from_pandas(data)
    fields = []
    for i in data_table.column_names:
        if pa.types.is_fixed_size_list(data_table[i].type):
            fields.append(pa.field(i, data_table[i].type))
        else:
            #print(data_table[i])
            inner_size = len(data_table[i][0])
            tensor_type = {"shape": [inner_size]}
            tensor_meta_type = {"tensor_type": json.dumps(tensor_type)}
            tensor_arrow_type = pa.list_(data_table[i][0][0].type, inner_size)
            fields.append(pa.field(i, tensor_arrow_type, metadata=tensor_meta_type))
        
    schema = pa.schema(fields)
    final_table = pa.Table.from_pandas(data, schema=schema).cast(target_schema=schema)
    #print(final_table.schema)
    arrow_file_name = outputArrowFile
    with pa.OSFile(arrow_file_name, 'wb') as sink:
        with pa.ipc.new_file(sink, final_table.schema) as arrow_ipc:
            arrow_ipc.write(final_table)
            arrow_ipc.close()
    return final_table

def main():
    # convert all of the JSON files to dataframe and arrow
    for currentFile in jsonFileList:
        newDataFrame = ConvertJSONtoDataframe(currentFile["inputFile"], currentFile["dataframeOutputFile"])
        #print(newDataFrame)
        # newArrow = ConvertJSONtoArrow(currentFile["inputFile"], currentFile["arrowOutputFile"])
        # print("table:")
        # print(newArrow)
        # print("Schema:")
        # print(newArrow.schema)
    


if __name__ == '__main__':
    main()