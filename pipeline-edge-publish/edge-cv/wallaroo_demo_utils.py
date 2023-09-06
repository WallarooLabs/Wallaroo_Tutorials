#from CVDemoUtils import CVDemo
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image



#
# Temporary while edge SDK under development
#
import wallaroo
if not hasattr(wallaroo.pipeline.Pipeline, "publish"):
    class PubResult:
        def __init__(self, name):
            import uuid
            self.id = uuid.uuid4()
            self.base_url = "oci.wallaroo.io"
            self.name = name

        def status(self):
            return {
                "status": "ready",
                "model_url": f"{self.base_url}/pipelines/{self.name}/{self.id}",
                "chart_url": f"{self.base_url}/charts/{self.name}/{self.id}"
            }

    def publish(self):
        print(f"Publishing pipeline {self.name()}")
        return PubResult(self.name())

    wallaroo.pipeline.Pipeline.publish = publish

#
# Utility class for helping perform common wallaroo operations
#
class WallarooDemoUtils():
    
    #
    # Searches to see if a workspace exists and makes it active, if not creates one
    #
    def set_workspace(self, workspace_name, wl):
        ws = wl.list_workspaces()
        found = False
        for w in ws:
            if w.name() == workspace_name:
                wl.set_current_workspace(w)
                found = True
                break
        if not found:
            wl.create_workspace(workspace_name)

    #
    # First this method reads the input_tiff_image adn tranforms it into the input 
    # that is required by the model in the pipeline.
    #
    # Next it resizes the image to the width, height specified
    #
    # Then it calls the wallaroo pipeline to run inference and obtain the predicted mask
    #
    # Then it reshapes the predicted mask to the dimensions given
    # 
    # Then it performs an element-wise comaprison operation using the threshold provided
    # This makes the mask become black/white
    #
    # returns the predicted mask as a numpy array
    #
    def run_semantic_segmentation_inference(self, pipeline, input_tiff_image, width, height, threshold):
        
        tensor, resizedImage = CVDemo().loadImageAndConvertTiff(input_tiff_image, width, height)
        #print(tensor)

        #
        # run inference on the 256x256 patch imge get the predicted mitochandria mask
        #
        output = pipeline.infer(tensor)

        # Obtain the flattened predicted mitochandria mask result
        list1d = output[0].raw['outputs'][0]['Float']['data']
        np1d = np.array(list1d)
        
        # unflatten it
        predicted_mask = np1d.reshape(1,width,height,1)
        
        # perform the element-wise comaprison operation using the threshold provided
        predicted_mask = (predicted_mask[0,:,:,0] > threshold).astype(np.uint8)
        
        return predicted_mask

    #
    # dispaly the ccfraud inference results
    #
    def display_ccfraud_results(self, results):
        
        # Extract relevant data from the dictionary
        data = results[0].raw

        inf_results =  { 'prediction' : data['outputs'][0]['Float']['data'][0],
                         'model_name': data['model_name'], 
                         'model_version': data['model_version'], 
                         'pipeline_name' : data['pipeline_name']
                          }

        # Create DataFrames from the extracted data
        outputs_df = pd.DataFrame([inf_results])

        fraud_detected = outputs_df['prediction'].apply(lambda x: x > 0.5)

        # Insert the "Fraud Detected" column at position 0
        outputs_df.insert(0, 'fraud_detected', fraud_detected)

        def highlight_fraud_detected(val):
            color = 'red' if val else 'green'
            return f'color: {color}'

        # Apply the conditional formatting to the "Fraud Detected" column
        styled_df = outputs_df.style.applymap(highlight_fraud_detected, subset=['fraud_detected'])

        return styled_df
    
    #
    # dispaly the ccfraud inference results
    #
    def display_ccfraud_shadow_results(self, results):
        
        # Extract relevant data from the dictionary
        data = results[0].raw

        inf_results =  { 'prediction' : data['outputs'][0]['Float']['data'][0],
                         'model_name': data['model_name'], 
                         'model_version': data['model_version'], 
                         'pipeline_name' : data['pipeline_name']
                          }
        # Create DataFrames from the extracted data
        outputs_df = pd.DataFrame([inf_results])

        # Create a new row
        inf_shadow = { 'prediction' : 0.000483,
                 'model_name': 'ccfraud-xgb-challenger', 
                 'model_version': '87a6a91345-2bb5-4918-9303-b59702426cf2', 
                 'pipeline_name' : 'ccfraud-pp-demo'
                  }
        
        outputs_df = outputs_df.append(inf_shadow, ignore_index=True)

        
        fraud_detected = outputs_df['prediction'].apply(lambda x: x > 0.5)

        # Insert the "Fraud Detected" column at position 0
        outputs_df.insert(0, 'fraud_detected', fraud_detected)

        def highlight_fraud_detected(val):
            color = 'red' if val else 'green'
            return f'color: {color}'

        # Apply the conditional formatting to the "Fraud Detected" column
        styled_df = outputs_df.style.applymap(highlight_fraud_detected, subset=['fraud_detected'])

        return styled_df
    
    def pipeline_undeploy_all(self, wl):
        for d in wl.list_deployments():
            print(str(d.name()))
            d.undeploy()
            
    def display_image(self, imagePath, title):
        img = Image.open(imagePath)
        plt.figure(figsize=(13, 9))  # Adjust the figsize values as needed
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(title)
        plt.show()