#!/usr/bin/env python
# coding: utf-8

#Introduction

# This notebook has been derived from the P8_Spark_Local notebook, with a few adjustements due to the configuration of the Cloud environment itself, such as:
# - working with S3,
# - downgrade of a few packages leading to a slight different approach for initial DataFrame building.<br/>
# 
# The core part, i.e. featurizer and reducer, still provide results we can compare with our local execution.

import pyspark
from pyspark import SparkContext
pyspark.__version__

# usefull packages
import pandas as pd
import numpy as np
import time
from PIL import Image
from io import BytesIO
from io import StringIO
# context & session
from pyspark.sql import SparkSession
import pyarrow
pyarrow.__version__
# data handling
from pyspark.sql.functions import input_file_name, split
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import udf
from pyspark.sql.types import *
# ml tasks
from pyspark.ml.image import ImageSchema
from pyspark.ml.feature import PCA
# transform
from pyspark.ml.linalg import Vectors, VectorUDT
# core featurizer
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
# Spark session
import sagemaker
from sagemaker import get_execution_role
import sagemaker_pyspark
role = get_execution_role()
# Configure Spark to use the SageMaker Spark dependency jars
jars = sagemaker_pyspark.classpath_jars()
classpath = ":".join(sagemaker_pyspark.classpath_jars())
# See the SageMaker Spark Github to learn how to connect to EMR from a notebook instance
spark = SparkSession.builder.config("spark.driver.extraClassPath", classpath)    .master("local[*]").getOrCreate()
spark
spark.conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')

# # S3 as data source
import boto3
# Get resources stored in AWS S3 service
s3 = boto3.resource('s3')
# Print all existing buckets names (only one in this case)
for bucket in s3.buckets.all():
    print(bucket.name)
# Print n first files present in the bucket 'ocproject-fruits'
fruits_bucket = s3.Bucket('ocproject-fruits')
for file in fruits_bucket.objects.limit(3):
    label = file.key.split('/')[-2]
    print(label, file.key)

# # Load data & Featurizer
# ### Use of a CNN as feature extractor
# model for featurization, last layers truncated.
conv_base = VGG16(
    include_top=False,
    weights=None,
    pooling='max',
    input_shape=(100, 100, 3))
conv_base.summary()
# get the 2134 first cnn_features
list_path_img = []
for file in fruits_bucket.objects.limit(2134):
    obj = fruits_bucket.Object(file.key)
    label = file.key.split('/')[-2]
    response = obj.get()
    file_stream = response['Body']
    img = Image.open(file_stream)
    # convert image to flatten array
    flat_array = np.array(img).ravel().tolist()
    tensor = np.array(flat_array).reshape(1, 100, 100, 3).astype(np.uint8)
    # preprocess input
    prep_tensor = preprocess_input(tensor)
    features = conv_base.predict(prep_tensor).ravel().tolist()
    # Store file key and features
    list_path_img.append((file.key, label, features))


# Create spark dataframe from previous list of tuples
df_img = spark.createDataFrame(list_path_img, ['origin', 'label', 'cnn_features'])
# # Reducer
# from Array to Vectors for PCA
array_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
df_img = df_img.withColumn('cnn_vectors', array_to_vector_udf('cnn_features'))
df_img = df_img.select('origin', 'label', 'cnn_vectors')
# reduce with PCA - k=20 arbitrary setting
pca = PCA(k=20, inputCol='cnn_vectors', outputCol='pca_vectors')
model = pca.fit(df_img)
# apply pca reduction
df_img = model.transform(df_img)
# from Vector to Array
vector_to_array_udf = udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))
df_img = df_img.withColumn('arrays', vector_to_array_udf('pca_vectors'))
df_img = df_img.select('origin', 'label', 'arrays')
# Turn spark dataframe into pandas dataframe
results_df = df_img.toPandas()
results_df
# store the results into S3 Bucket, using boto3
# buffer
csv_buffer = StringIO()
results_df.to_csv(csv_buffer)
# boto client
client = boto3.client('s3')
# put the object
response = client.put_object(
    Body=csv_buffer.getvalue(),
    Bucket='ocproject-fruits',
    Key='results.csv')
# overview of the upload
response
# stop spark
spark.stop()

