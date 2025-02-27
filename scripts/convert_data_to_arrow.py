import pandas as pd
import pyarrow as pa
import json
import convert_wallaroo_data

# List of files to convert
jsonFileList = [
    # {
    #     "data_type_dict": {"text_input": pa.float32()},
    #     "inputFile": "wallaroo-model-cookbooks/aloha/data/data_25k.df.json",
    #     "arrowOutputFile": "wallaroo-model-cookbooks/aloha/data/data_25k.arrow"
    # }
    {
        "data_type_dict": None,
        "inputFile": "wallaroo-features/pipeline-log-tutorial/data/xtest-1.df.json",
        "arrowOutputFile": "wallaroo-features/pipeline-log-tutorial/data/xtest-1.arrow"
    }
]

def main():
    # convert all of the JSON files to dataframe and arrow
    for currentFile in jsonFileList:
        currentDataFrame = pd.read_json(currentFile['inputFile'], orient="records")
        newArrowTable = convert_wallaroo_data.convert_pandas_to_arrow(currentDataFrame, currentFile['data_type_dict'])
        with pa.OSFile(currentFile['arrowOutputFile'], 'wb') as sink:
            with pa.ipc.new_file(sink, newArrowTable.schema) as arrow_ipc:
                arrow_ipc.write(newArrowTable)
                arrow_ipc.close()
        #print(newDataFrame)
        # newArrow = ConvertJSONtoArrow(currentFile["inputFile"], currentFile["arrowOutputFile"])
        # print("table:")
        # print(newArrow)
        # print("Schema:")
        # print(newArrow.schema)



if __name__ == '__main__':
    main()