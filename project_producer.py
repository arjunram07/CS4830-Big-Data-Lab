#from kafka import KafkaProducer
#import json
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql import Row
from pyspark.sql.types import *


sc = SparkContext()
spark = SparkSession(sc)

data_path = "gs://arunprak_finalproject/data/"

column_names = ['Summons Number', 'Plate ID', 'Registration State', 'Plate Type',
                   'Issue Date', 'Violation Code', 'Vehicle Body Type', 'Vehicle Make',
                   'Issuing Agency', 'Street Code1', 'Street Code2', 'Street Code3',
                   'Vehicle Expiration Date', 'Issuer Code', 'Issuer Command',
                   'Issuer Squad', 'Violation Time', 'Time First Observed',
                   'Violation_County', 'Violation In Front Of Or Opposite', 'House Number',
                   'Street Name', 'Intersecting Street', 'Date First Observed',
                   'Law Section', 'Sub Division', 'Violation Legal Code',
                   'Days Parking In Effect', 'From Hours In Effect', 'To Hours In Effect',
                   'Vehicle Color', 'Unregistered Vehicle?', 'Vehicle Year',
                   'Meter Number', 'Feet From Curb', 'Violation Post Code',
                   'Violation Description', 'No Standing or Stopping Violation',
                   'Hydrant Violation', 'Double Parking Violation', 'Latitude',
                   'Longitude', 'Community Board', 'Community Council', 'Census Tract',
                   'BIN', 'BBL', 'NTA']

userSchema = StructType([StructField(colname, StringType(), True) for colname in column_names])

raw_data = spark.readStream.option("sep", ",").schema(userSchema).csv(data_path)

host_ip, port, topic_name = '10.150.0.2', '9092', 'nycdata'

# Writing data to Kafka topic
kafka_stream = raw_data \
                  .withColumn("value", f.to_json(f.struct([raw_data[x] for x in raw_data.columns]))) \
                  .writeStream \
                  .format("kafka") \
                  .option("checkpointLocation", "gs://arunprak_finalproject/") \
                  .option("kafka.bootstrap.servers", "{}:{}".format(host_ip, port)) \
                  .option("topic", topic_name) \
                  .start()


kafka_stream.awaitTermination()
