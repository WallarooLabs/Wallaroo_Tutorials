# https://youtu.be/7IL7LKSLb9I
"""
pip install patchify
"""

from IPython.display import clear_output, display
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
import piexif
import os
import base64
from PIL import Image
import io
import urllib.parse
import cv2
import requests
from keras.utils import normalize
import random

#
# Utility class for helping with processing large Tiff Files
#
class TiffUtils():

    #
    # Prints the attributes of a tiff volume
    #
    def inspect_tiff(self, filePath):
        with tiff.TiffFile(filePath) as tif:
            num_images = len(tif.pages)
            image_sizes = []
            
            print(f"filename:{tif.filename}")
            print(f"page cnt:{len(tif.pages)}")
            
            tiff_files = tif.series
            print(f"image series cnt:{len(tiff_files)}")
            print(f"shape:{tiff_files[0].asarray().shape}")
            print(f"dtype:{tiff_files[0].dtype}")
            
    
    #
    # Extracts the images from the filePath tiff and writes it to a folder called tiffdir
    #
    def extract_images_from_tiff(self, filePath, tiffdir):
        with tiff.TiffFile(filePath) as tif:
            print(f"filename:{tif.filename}")
           
            if not os.path.exists(tiffdir):
                tiffdir = os.mkdir(tiffdir)
        
            # Open the TIFF file
            # Iterate over each page in the file
            for i, page in enumerate(tif.pages):
                # Construct the output filename for this page
                filename = f'page_{i}.tif'
                filename = f"{tiffdir}/{filename}"
                # Write the page to a new TIFF file
                with tiff.TiffWriter(filename) as tif_writer:
                    #print(filename)
                    tif_writer.save(page.asarray())
            print(f"created dir {tiffdir} with {i} files")
          
    #
    # Decode the base64-encoded TIFF image 
    #
    def decode_tiff_image_stream(self, encoded_image):
        img_data = base64.b64decode(encoded_image)
        img_buffer = io.BytesIO(img_data)
        return Image.open(img_buffer)

    #
    # display an base64-encoded TIFF iamge in a plot
    #
    def display_tiff_image_stream(self, encoded_image):
        img_data = base64.b64decode(encoded_image)
        img = Image.open(io.BytesIO(img_data))
        plt.imshow(img)
        plt.show()
        
    #
    # display an base64-encoded TIFF iamge in a plot
    #
    def display_tiff_image(self, tiff_image_path):
        img = tiff.imread(tiff_image_path)
        plt.imshow(img)
        plt.show()
        
    def read_image_resize(self,file_path, width, height):
        image = cv2.imread(file_path, 0)
        image = Image.fromarray(image)
        image = image.resize((width, height))
        return image
    
    def read_tiff_from_file(self, tiff_file):
        with urllib.request.urlopen(tiff_file) as tif:
                image_data = url.read()
                # Iterate over each page in the TIFF file
                for page in tif.pages:

                    # Extract the pixel data from the current page
                    page_array = page.asarray()

                    # Append the page array to the list of page arrays
                    page_arrays.append(page_array)

        # Combine all page arrays into a single NumPy array
        tiff_array =  np.concatenate(page_arrays, axis=0)
        return cv2.imdecode(tiff_array, cv2.IMREAD_UNCHANGED)
    
    
    #
    # Reads a the url content as bytes and use tifffile to convert to a
    # multi dimensional array
    #
    def read_tiff_from_url(self, url: str) -> np.ndarray:
        # Download the TIFF file from the public URL
        response = requests.get(url)

        # Raise an exception if the request failed
        response.raise_for_status()

        # Read the TIFF data and convert it into a NumPy array
        with tiff.TiffFile(io.BytesIO(response.content)) as tif:
            images = tif.asarray()

        return images
    
    
    def get_all_patches(self,patch_path):
        patches = {}
        SIZE = 256
        image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
        images_path = patch_path +"/images"
        patch_img_list = os.listdir(images_path)
        #print(patch_img_list)
        images = sorted(patch_img_list)
        patches['image_files'] = images

        for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
            if (image_name.split('.')[1] == 'tif'):
                #print(f"image_name={image_name}")
                image = cv2.imread(images_path + "/" + image_name, 0)
                image = Image.fromarray(image)
                image = image.resize((SIZE, SIZE))
                image_dataset.append(np.array(image))
        mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.
        masks_path = patch_path +"/masks"

        patch_mask_list = os.listdir(masks_path)
        masks = sorted(patch_mask_list)
        patches['mask_files'] = masks
        for i, image_name in enumerate(masks):
            if (image_name.split('.')[1] == 'tif'):
                image = cv2.imread(masks_path + "/" + image_name, 0)
                image = Image.fromarray(image)
                image = image.resize((SIZE, SIZE))
                mask_dataset.append(np.array(image))

        image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),3)
        #D not normalize masks, just rescale to 0 to 1.
        mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.
        
        patches['image_dataset'] = image_dataset       
        patches['mask_dataset'] = mask_dataset       

        return patches
                                     
    def get_random_patch_sample(self, patches):
         rand_idx = random.randint(0, len(patches['image_dataset']))

         return {'index':rand_idx, \
                 'patch_image': patches['image_dataset'][rand_idx], \
                 'patch_mask' : patches['mask_dataset'][rand_idx], \
                 'patch_image_file' : patches['image_files'][rand_idx], \
                 'patch_mask_file' : patches['mask_files'][rand_idx] }                        
                                     
    #
    # Build patches
    # directory - dir to store all images in
    # dimensions - patch size represented as a tuble (w,h) 
    # step - how far to move over before getting next patch
    # image_file_name - tiff image with time series
    # mask_file_name - tiff mask image with time series
    # step - space between
    #
    def build_patches(self, directory, dimensions, step, image_file_name, mask_file_name = None):
        patches = {}
        
        #
        # handle the images
        #       
        is_url = bool(urllib.parse.urlparse(image_file_name).scheme)
        if not is_url:
            #print(f"large_image_stack={image_file_name}")
            large_image_stack = tiff.imread(image_file_name)
        else:
            large_image_stack = self.read_tiff_from_url(image_file_name)
        
        #print(f"large_image_stack.shape = {large_image_stack.shape}")
        #print("Data type:", large_image_stack.dtype)

        # get the filename without ext or path
        filename_with_ext = os.path.basename(image_file_name)
        filename_without_ext, _ = os.path.splitext(filename_with_ext)
        #print(f"filename_with_ext={filename_with_ext}")
        #print(f"filename_without_ext={filename_without_ext}")

        #print(f"directory={directory}")
        #print(f"filename_without_ext={filename_without_ext}")

        if not os.path.exists(directory):
            os.mkdir(directory)
        # create directory for using filename
        patches_dir = directory+"/"+filename_without_ext
        if not os.path.exists(patches_dir):
            os.mkdir(patches_dir)
        print(f"created dir {patches_dir}")
              
        # read the filename and save it to the patches dir
        pipeline_file = patches_dir+"/"+filename_without_ext+".tiff"
        with tiff.TiffWriter(pipeline_file) as tif:
            tif.save(large_image_stack)
        print(f"saving file {pipeline_file}")
        
        # create the images dir
        patches_images_dir = patches_dir+"/images"
        if not os.path.exists(patches_images_dir):
            os.mkdir(patches_images_dir)
        
        # iterate through all the time series images in the tiff
        patches_img_list = []
        for img in range(large_image_stack.shape[0]):
            large_image = large_image_stack[img]

            #
            # for each time series image create patch images of 256x256 squares and save to images dir
            #
            patches_img = patchify(large_image, dimensions, step=step)  #Step=256 for 256 patches means no overlap
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):           
                    single_patch_img = patches_img[i,j,:,:]
                    file_name = patches_images_dir + '/image_' + str(img) + '_' + str(i)+str(j)+ ".tif"
                    #tiff.imwrite(fileName, single_patch_img,metadata={'spacing': 0.56, 'unit': 'um', 'axes': 'TZYX'})
                    with tiff.TiffWriter(file_name) as tif:
                        tif.save(single_patch_img)
                    patches_img_list.append(file_name)
        #
        # handle the masks
        #
        
        # create the masks dir
        patches_masks_dir = patches_dir+"/masks"
        if not os.path.exists(patches_masks_dir):
            os.mkdir(patches_masks_dir)
            
        # if provided, copy the masks file
        patches_mask_list = []
        if mask_file_name is not None:
            #print(f"mask_file_name:{mask_file_name}")

            is_url = bool(urllib.parse.urlparse(mask_file_name).scheme)
            if not is_url:
                large_mask_stack = tiff.imread(mask_file_name)
            else:
                large_mask_stack = self.read_tiff_from_url(mask_file_name)
                
            pipeline_file = patches_dir+"/"+filename_without_ext+"-masks.tiff"
            with tiff.TiffWriter(pipeline_file) as tif:
                tif.save(large_mask_stack)
            #print(f"saving file {pipeline_file}")
            
            # iterate through all the time series masks in the tiff
            
            for img in range(large_mask_stack.shape[0]):
                large_image = large_mask_stack[img]

                #
                # for each time series image create patch images of 256x256 squares and save to images dir
                #
                patches_mask = patchify(large_image, dimensions, step=step)  #Step=256 for 256 patches means no overlap
                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):           
                        single_patch_mask = patches_mask[i,j,:,:]
                        file_name = patches_masks_dir + '/mask_' + str(img) + '_' + str(i)+str(j)+ ".tif"
                        #tiff.imwrite(fileName, single_patch_img,metadata={'spacing': 0.56, 'unit': 'um', 'axes': 'TZYX'})
                        with tiff.TiffWriter(file_name) as tif:
                            tif.save(single_patch_mask)
                        patches_mask_list.append(file_name)
                        
        patches['patches_images_dir'] = patches_images_dir
        patches['patches_img_list'] = patches_img_list
        patches['patches_masks_dir'] = patches_masks_dir
        patches['patches_mask_list'] = patches_mask_list
        
        return patches

    
    def displayImage(self, imagePath, title):
        img = Image.open(imagePath)
        plt.figure(figsize=(16, 16))  # Adjust the figsize values as needed
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(title)
        plt.show()
        
    def displayMicrospyTimeSeries(self, trainingImagePath, trainTitle, groundTruthPath, groundTruthTitle):
        # Read the multi-slice TIFF images
        with tiff.TiffFile(trainingImagePath) as tif:
            training_image_stack = tif.asarray()

        with tiff.TiffFile(groundTruthPath) as tif:
            additional_image_stack = tif.asarray()

        # Display the image stacks as animations side by side
        #%matplotlib inline

        # Determine the number of slices in the image stacks (assuming both have the same number of slices)
        num_slices = training_image_stack.shape[0]

        # Iterate through the slices, updating the display
        for i in range(num_slices):
            clear_output(wait=True)  # Clear the previous plot before displaying the next one

            fig, axes = plt.subplots(1, 2, figsize=(20, 7.5))  # Adjust the figsize values as needed

            axes[0].imshow(training_image_stack[i], cmap='gray', vmin=0, vmax=255)
            axes[0].axis('off')
            axes[0].set_title(f"{trainTitle} - Slice {i+1}/{num_slices} - 1024x768 pixels - 8 bit")

            axes[1].imshow(additional_image_stack[i], cmap='gray', vmin=0, vmax=255)
            axes[1].axis('off')
            axes[1].set_title(f"{groundTruthTitle} - Slice {i+1}/{num_slices} - 1024x768 pixels - 8 bit")

            plt.show()
            #time.sleep(0.01)  # Add a delay between frames (0.1 seconds in this case)
    
    def draw_squares(self, image, square_size=256):
        img_height, img_width, _ = image.shape

        for y in range(0, img_height, square_size):
            for x in range(0, img_width, square_size):
                top_left = (x, y)
                bottom_right = (x + square_size, y + square_size)
                color = (0, 0, 255)  # Red color in BGR format
                thickness = 2
                cv2.rectangle(image, top_left, bottom_right, color, thickness)

    def plot_test_results(self, test_image, test_image_title, \
                        ground_truth_image, ground_truth_image_title, \
                        predicted_mask, predicted_mask_title):

        plt.figure(figsize=(16, 8))
        # plot the test image
        plt.subplot(231)
        plt.title(test_image_title)
        plt.imshow(test_image, cmap='gray')

        # plot the test ground truth mask
        plt.subplot(232)
        plt.title(ground_truth_image_title)
        plt.imshow(ground_truth_image, cmap='gray')

        plt.subplot(233)
        plt.title(predicted_mask_title)
        plt.imshow(predicted_mask, cmap='gray')
        
    def plot_inferenced_results(self, input_image, input_image_title, \
                        predicted_mask, predicted_mask_title):

        plt.figure(figsize=(12, 8))
        
        # plot the input image
        plt.subplot(231)
        plt.title(input_image_title)
        plt.imshow(input_image, cmap='gray')

        plt.subplot(233)
        plt.title(predicted_mask_title)
        plt.imshow(predicted_mask, cmap='gray')
        plt.show()    