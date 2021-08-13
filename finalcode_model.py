from __future__ import print_function
from pyspark.context import SparkContext
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as sqlf
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression, OneVsRest, LinearSVC, DecisionTreeClassifier,RandomForestClassifier
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StandardScaler
from pyspark.sql.types import StructType, StructField, DoubleType, StringType

sc = SparkContext()
spark = SparkSession(sc)

######################################################################
#Loading data from bucket
######################################################################

raw_data = spark.read.csv('gs://arunprak_finalproject/data/*', header=True, inferSchema=True)

#Selecting only those columns which are useful after preprocessing 

raw_data = raw_data.select('Registration State','Plate Type','Violation Code',
						 'Vehicle Body Type','Vehicle Make','Issuer Command',
						 'Issuer Squad','Violation_County')

raw_data = raw_data.na.drop()
######################################################################
#Label Encoding the Categorical Columns
######################################################################

indexers = [StringIndexer(inputCol ="Violation_County", outputCol = "label", handleInvalid='keep'), 
			StringIndexer(inputCol="Registration State", outputCol="RegistrationStateindex", handleInvalid='keep'), 
			StringIndexer(inputCol="Plate Type", outputCol="PlateTypeindex", handleInvalid='keep'), 
			StringIndexer(inputCol="Violation Code", outputCol="ViolationCodeindex", handleInvalid='keep'), 
			StringIndexer(inputCol="Vehicle Body Type", outputCol="VehicleBodyTypeindex", handleInvalid='keep'), 
			StringIndexer(inputCol="Vehicle Make", outputCol="VehicleMakeindex", handleInvalid='keep'), 
			StringIndexer(inputCol="Issuer Command", outputCol="IssuerCommandindex", handleInvalid='keep'), 
			StringIndexer(inputCol="Issuer Squad", outputCol="IssuerSquadindex", handleInvalid='keep')]

pipeline = Pipeline(stages = indexers) 
pipeline_mod = pipeline.fit(raw_data)
 
#################################################################
#Saving the indexers  
#################################################################

pipeline_mod.save('gs://arunprak_finalproject/model/Preprocessing_Model_Fulldata_2')

new_data = pipeline_mod.transform(raw_data)
 
train, val = new_data.randomSplit([0.95, 0.05])  #Splitting the data into 95-5 

#################################################################
#Assembling and Scaling the data
#################################################################

assembler = VectorAssembler(inputCols = ["RegistrationStateindex","PlateTypeindex","ViolationCodeindex",
										 "VehicleBodyTypeindex","VehicleMakeindex","IssuerCommandindex",
										 "IssuerSquadindex"], outputCol = "features")

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd = True, withMean = False)

#################################################################
#Model Selection and Parameter Search
#################################################################

rf = RandomForestClassifier(labelCol="label", featuresCol="scaledFeatures",maxBins=1000)

p = Pipeline(stages = [assembler,scaler,rf])

evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol="prediction",metricName = "accuracy")

paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [10,15,20,25]).build()

crossval = CrossValidator(estimator = p, estimatorParamMaps = paramGrid, evaluator = evaluator ,numFolds = 3)
cvmodel = crossval.fit(train)
prediction = cvmodel.transform(val)

model_accuracy_best = evaluator.evaluate(prediction)

print("-"*50)
print("Best_model valid performance (LR): "+ str(model_accuracy_best))
print("-"*50)

################################################################
#Saving the best model
################################################################

best_Model = cvmodel.bestModel            

best_Model.save('gs://arunprak_finalproject/model/Trained_Model_Fulldata')
