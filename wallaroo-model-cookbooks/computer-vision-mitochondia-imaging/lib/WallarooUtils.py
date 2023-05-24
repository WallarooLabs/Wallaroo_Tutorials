#from CVDemoUtils import CVDemo
from lib.CVDemoUtils import CVDemo
import numpy as np

#
# Utility class for helping perform common wallaroo operations
#
class WallarooUtils():
    
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
        # print(tensor)

        # #
        # # run inference on the 256x256 patch image get the predicted mitochandria mask
        # #
        output = pipeline.infer(tensor)
        # print(output)

        # # Obtain the flattened predicted mitochandria mask result
        # # list1d = output[0].raw['outputs'][0]['Float']['data']
        list1d = output.loc[0]["out.conv2d_37"]
        np1d = np.array(list1d)
        
        # # unflatten it
        predicted_mask = np1d.reshape(1,width,height,1)
        
        # # perform the element-wise comaprison operation using the threshold provided
        predicted_mask = (predicted_mask[0,:,:,0] > threshold).astype(np.uint8)
        
        # return predicted_mask
        return predicted_mask