import pandas as pd
import pyarrow as pa
import json
import convert_wallaroo_data

# setting arrow on or off
#import os
#os.environ["ARROW_ENABLED"]="False"

# List of files to convert
jsonFileList = [
    {
        "inputFile": "wallaroo-101/data/cc_data_10k.df.json",
        "arrowOutputFile": "wallaroo-101/data/cc_data_10k.arrow"
    }

]

def main():
    # convert all of the JSON files to dataframe and arrow
    for currentFile in jsonFileList:
        currentDataFrame = pd.read_json(currentFile['inputFile'], orient="records")
        newArrowTable = convert_wallaroo_data.convert_pandas_to_arrow(currentDataFrame, None)
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