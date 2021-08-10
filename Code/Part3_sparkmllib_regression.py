import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil import parser

from pyspark.sql.functions import *
from pyspark.sql.types import *

from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.feature import RFormula
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# WoW_Churn_Data(wow 게임 이탈 사용자 데이터, 가공 데이터)
df = spark.read.load('abfss://aaafilesystem@aaatraing.dfs.core.windows.net/sparkpooldata/churnersdf_yj3.csv', format='csv', header=True)

df = df.withColumn('churn', regexp_replace('churn', 'FALSE', '0')).withColumn('churn', regexp_replace('churn', 'TRUE', '1'))

df1=df.withColumn('level', col('level').cast(IntegerType()))\
      .withColumn('Size', col('Size').cast(IntegerType()))\
      .withColumn('Min_req_level', col('Min_req_level').cast(IntegerType()))\
      .withColumn('Min_rec_level', col('Min_rec_level').cast(IntegerType()))\
      .withColumn('Max_rec_level', col('Max_rec_level').cast(IntegerType()))\
      .withColumn('Min_bot_level', col('Min_bot_level').cast(IntegerType()))\
      .withColumn('Max_bot_level', col('Max_bot_level').cast(IntegerType()))\
      .withColumn('churn', col('churn').cast(IntegerType()))

df2= df1.withColumn('log_timestamp', to_timestamp(unix_timestamp(col('log_timestamp'), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")))
df3= df2.withColumn('churn_timestamp', to_timestamp(unix_timestamp(col('churn_timestamp'), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")))

df3 = df3.withColumn('weekday_num', date_format('churn_timestamp', 'u')).withColumn('weekday', date_format('churn_timestamp', 'E'))\
         .withColumn('churn_month', date_format('churn_timestamp','MM'))
df4 = df3.withColumn('weekday_num', col('weekday_num').cast(IntegerType())).withColumn('churn_month', col('churn_month').cast(IntegerType()))

print(df4.printSchema())

sampled_df = df4.sample(True, 0.0025, seed=1234)
print(sampled_df.count()) #21,549

display(sampled_df.groupBy('churn').count())

display(sampled_df.groupBy('churn_month').agg(countDistinct('IdentifierId')))

# Because the sample uses an algorithm that works only with numeric features, convert them so they can be consumed
sI1 = StringIndexer(inputCol="race", outputCol="raceIndex")
en1 = OneHotEncoder(dropLast=False, inputCol="raceIndex", outputCol="raceVec")
sI2 = StringIndexer(inputCol="charclass", outputCol="charclassIndex")
en2 = OneHotEncoder(dropLast=False, inputCol="charclassIndex", outputCol="charclassVec")
sI3 = StringIndexer(inputCol="Continent", outputCol="ContinentIndex")
en3 = OneHotEncoder(dropLast=False, inputCol="ContinentIndex", outputCol="ContinentVec")
#'Area'
#'Type'
#'Controlled'

# Create a new DataFrame that has had the encodings applied
encoded_final_df = Pipeline(stages=[sI1, en1, sI2, en2, sI3, en3]).fit(sampled_df).transform(sampled_df)

encoded_final_df.printSchema()

pd_df=encoded_final_df.toPandas() #Spark DataFrame → #Pandas DataFrame

#pd.set_option('display.max_columns', 100)
#pd.set_option('display.max_rows', 100)
pd_df.describe()

pd_df_num=pd_df[['level','Size', 'Min_req_level', 'Min_rec_level', 'Max_rec_level', 'Min_bot_level', 'Max_bot_level','weekday_num', 'churn_month', 'raceIndex','charclassIndex','ContinentIndex','churn']]

plt.rcParams["figure.figsize"] = [7, 5]
heat_map = sns.heatmap(pd_df_num.corr(method='pearson'), annot=True, fmt='.1f', linewidths=2, cmap='YlGnBu')
heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=45);
plt.show()

# How many passengers tipped by various amounts 
plt.rcParams["figure.figsize"] = [7, 4]
ax = pd_df_num.boxplot(column=['level'], by=['churn'])
#ax.set_title('Churn by level \n')
ax.set_xlabel('Churn')
ax.set_ylabel('level')
plt.show()

plt.rcParams["figure.figsize"] = [7, 3]
ax = sns.boxplot(x="churn", y="weekday_num",data=pd_df_num, showfliers = False)
ax.set_title('churn distribution by weekday')
ax.set_xlabel('weekday')
ax.set_ylabel('churn')
plt.show()

# Look at the relationship between fare and tip amounts
plt.rcParams["figure.figsize"] = [3, 3]
ax1 = pd_df_num.plot(kind='scatter', x= 'weekday_num', y = 'level', c='red', alpha = 0.10, s=2.5*(pd_df_num['churn']==1))
ax2 = pd_df_num.plot(kind='scatter', x= 'weekday_num', y = 'level', c='blue', alpha = 0.10, s=2.5*(pd_df_num['churn']==0))
ax1.set_title('churn by weekday_num \n')
ax1.set_xlabel('weekday_num')
ax1.set_ylabel('level')
plt.show(ax1,ax2)

# trainingFraction, testingFraction = (1-trainingFraction), seed
train_df, test_df = encoded_final_df.randomSplit([0.7, 0.3], 1234)

## Create a new logistic regression object for the model
logReg = LogisticRegression(maxIter=10, regParam=0.3, labelCol = 'churn')

## The formula for the model
classFormula = RFormula(formula='churn ~ level +  race +  charclass +  Continent + Type +  Size +  Controlled +  Min_req_level +  Min_rec_level + \
 Max_rec_level +  Min_bot_level +  Max_bot_level +  weekday_num + raceVec +  charclassVec + ContinentVec')

## Undertake training and create a logistic regression model
lrModel = Pipeline(stages=[classFormula, logReg]).fit(train_df)

## Saving the model is optional, but it's another form of inter-session cache
datestamp = datetime.now().strftime('%m-%d-%Y-%s')
fileName = "wow_lrModel_" + datestamp
logRegDirfilename = fileName
lrModel.save(logRegDirfilename)

## Predict tip 1/0 (yes/no) on the test dataset; evaluation using area under ROC
predictions = lrModel.transform(test_df)
predictionAndLabels = predictions.select("label","prediction").rdd
metrics = BinaryClassificationMetrics(predictionAndLabels)
print("Area under ROC = %s" % metrics.areaUnderROC)

## Plot the ROC curve; no need for pandas, because this uses the modelSummary object
plt.rcParams["figure.figsize"] = [3, 3]
modelSummary = lrModel.stages[-1].summary

plt.plot([0, 1], [0, 1], 'r--')
plt.plot(modelSummary.roc.select('FPR').collect(),
         modelSummary.roc.select('TPR').collect())
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
