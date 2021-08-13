from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from itertools import chain
from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql import Row
from pyspark.sql.types import *

spark = SparkSession.builder.appName("NYC_Data_Streaming").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

host_ip, port, topic_name = '10.150.0.2', '9092', 'nycdata'

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

# Subscribing to the topic
test_data = spark \
                .readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", "{}:{}".format(host_ip, port)) \
                .option("subscribe", topic_name) \
                .load()


################################
# Preprocessing the data
################################
# Parsing the json message
raw_data = test_data.withColumn("value", f.from_json(f.col("value").cast("string"), userSchema)).select(f.col("value.*"))

# Selecting the columns required for the model
raw_data = raw_data.select('Summons Number', 'Registration State','Plate Type','Violation Code','Vehicle Body Type','Vehicle Make','Issuer Command','Issuer Squad','Violation_County')
raw_data = raw_data.na.drop()

# Converting string columns to indices
preprocess_model = PipelineModel.load('gs://arunprak_finalproject/model/Preprocessing_Model_Fulldata_2/') 
new_data = preprocess_model.transform(raw_data)

##########################################
# Using the saved model for prediction
##########################################
trained_model = PipelineModel.load('gs://arunprak_finalproject/model/Trained_Model_Fulldata/') 
labels_preds = trained_model.transform(new_data).select(f.col('Summons Number'), f.col("label"), f.col("prediction"))

# Using the indices to retrieve the true names for labels and predictions
index_label_map = dict(zip([0.0, 1.0, 2.0, 3.0], ['NY', 'K', 'Q', 'BX']))
mapp_expr = f.create_map([f.lit(x) for x in chain(*index_label_map.items())])
labels_preds = labels_preds \
                    .withColumn("true_label_names", mapp_expr[f.col("label")]) \
                    .withColumn("prediction_names", mapp_expr[f.col("prediction")])

##########################################
# Computing the evaluation metrics 
##########################################
# Accuracy
labels_preds = labels_preds.withColumn('correct', f.when((f.col('label') == f.col('prediction')), 1).otherwise(0))
df_accuracy = labels_preds.select(f.format_number(f.avg('correct'), 4).alias('accuracy'))

################################
#  Printing the output
################################
# True labels and predictions
q1 = labels_preds \
        .select("Summons Number", "true_label_names", "prediction_names") \
        .writeStream \
        .queryName("Real-time Predictions") \
        .outputMode('update') \
        .format('console') \
        .start()

q2 = df_accuracy.writeStream.queryName("accuracy").outputMode('complete').format('console').start()

q1.awaitTermination()
q2.awaitTermination()
