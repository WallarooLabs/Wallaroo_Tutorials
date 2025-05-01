# Wallaroo SDK
import wallaroo

# Data Tools
import pandas as pd
import numpy as np
#import logging
import json

# S3 Bucket Tools
import os
#import boto3
#from boto3 import session
#import botocore
#from botocore.client import Config
#from botocore.exceptions import ClientError

class Util():

    def convert_data(self,tensor,name):
        # get npArray from the tensorFloat
        npArray = tensor.cpu().numpy()
        pd.set_option('display.max_colwidth', None)
        #print(npArray.shape)

        #creates a dictionary with the wallaroo "tensor" key and the numpy ndim array representing image as the value.
        dictData = {name:[npArray]}  # Scott Edit
        dataframedata = pd.DataFrame(dictData)
        return(dataframedata)
    
    def convert_to_json(self,tensor):
        # get npArray from the tensorFloat
        npArray = tensor.cpu().numpy()
        pd.set_option('display.max_colwidth', None)
        #print(npArray.shape)

        #creates a dictionary with the wallaroo "tensor" key and the numpy ndim array representing image as the value.
        dictData = {"tensor":[npArray]}
        dataframedata = pd.DataFrame(dictData)
        jsonData = dataframedata.to_json(orient='records')
        return(jsonData)
    
    
    # methods for s3 commands (object storage)
    def uploadFile_s3(self,client,bucket,filename, newFilename):
        try:
            client.upload_file(filename, bucket, newFilename)
        except ClientError as e:
            logging.error(e)

    def downloadFile_s3(self,client,bucket,filename, newFilename):
        print(newFilename)
        try:
            client.download_file(bucket, filename,newFilename)
        except ClientError as e:
            logging.error(e)

    def createBucket_s3(self,client,bucketName):
        try:
            client.create_bucket(Bucket=bucketName)
        except ClientError as e:
            logging.error(e)

    def deleteFile_s3(self,client,bucket,filename):
        try:
            client.delete_object(Bucket=bucket,Key=filename)
        except ClientError as e:
            logging.error(e)

    def getListObjects_s3(self,s3_type,client,bucket,folder):
        fileList = []
        
        if s3_type == 'minio':
            objectDict = client.list_objects(Bucket=bucket, Prefix=folder)

            if 'Content' not in objectDict:
                print('Reading Contents...')
                print('Contents do not exist.')
                return False
            else:
                try:
                    for key in client.list_objects(Bucket=bucket, Prefix=folder):#['Contents']:
                        fullWord=key['Key']
                        splitWord = "/"
                        newFilename = fullWord.split(splitWord)[1]
                        if newFilename != '':
                            fileList.append(newFilename)            
                except ClientError as e:
                    logging.error(e)
                return fileList

        if s3_type == 's3':
            if client.list_objects(Bucket=bucket, Prefix=folder)['Contents']:
                try:
                    for key in client.list_objects(Bucket=bucket, Prefix=folder):#['Contents']:
                        fullWord=key['Key']
                        splitWord = "/"
                        newFilename = fullWord.split(splitWord)[1]
                        if newFilename != '':
                            fileList.append(newFilename)            
                except ClientError as e:
                    logging.error(e)
                return fileList

        else:
            print(f'{s3_type} not found.')
            return False

    def createFolder_s3(self,client,bucket_name, folder_name):
        try:
            #client.delete_object(Bucket=bucket,Key=filename)
            client.put_object(Bucket=bucket_name, Key=(folder_name + '/'))
        except ClientError as e:
            logging.error(e)

    def uploadFolderContents_s3(self,client,source_directory,dest_directory,fileExtension,current_bucket):
        numberFiles = 0
        for filename in os.listdir(source_directory):
            if filename.endswith(fileExtension):
                numberFiles += 1
                # Loop through the /images directory
                sourcePath = source_directory+"/"+filename
                destPath = dest_directory+"/"+filename
                print(f"{sourcePath} --> {destPath}")
                #uploadFile('images/checkout.jpeg',current_bucket,'checkout.jpeg')
                client.uploadFile(sourcePath,current_bucket,destPath)
        print(f"{numberFiles} uploaded to {current_bucket}")
        
    def deleteAllinBucket_s3(self,client,current_bucket):
        try:
        #print(response)
            for each in response:
                filename=(each['Key'])
                client.delete_object(Bucket=current_bucket,Key=filename)        
        except ClientError as e:
            logging.error(e)
            response = client.list_objects(Bucket=current_bucket)['Contents']
        except ClientError as e:
            logging.error(e)