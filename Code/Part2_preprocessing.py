from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd

# abfss://{filesystem}@{storage acount}.{domain}/{path}/{filename}.{fileformat}
# Character_Log_Data(사용자 캐릭터 로그 데이터)
wow_log_df = spark.read.load('abfss://aaafilesystem@aaatraing.dfs.core.windows.net/sparkpooldata/WoW_Logs.csv', format='csv', header=True)

# InGame_Log_Data(인게임 캐릭터 위치 로그 데이터)
zones_df = spark.read.load('abfss://aaafilesystem@aaatraing.dfs.core.windows.net/sparkpooldata/zones.csv', format='csv', header=True)

# Churn_Label_Log_Data(이탈 여부 태그 로그 데이터)
churners_df = spark.read.load('abfss://aaafilesystem@aaatraing.dfs.core.windows.net/sparkpooldata/churners.csv', format='csv', header=True)

#spark dataframe to pandas dataframe 변환작업 
#churner_pandas_df = churners_df.toPandas()
#wow_log_pandas_df = wow_log_df.toPandas()
#zones_pandas_df = zones_df.toPandas()

#연습용 셀

#wow_log_df.printschema()
#print(wow_log_df.describe().show()) #컬럼정보들 표시
#display(wow_log_df)
#print(wow_log_df.agg({"level": "max"}).collect()) #표현식 계산
#print(wow_log_df.columns) #컬럼명
#print(wow_log_df.count()) #row수 반환
#print(wow_log_df.drop('level').printSchema()) #컬럼삭제
#print(wow_log_df.dropDuplicates(['IdentifierId']).count()) #중복제거


# Character_Log_Data(사용자 캐릭터 로그 데이터)
wow_log_df = wow_log_df.withColumnRenamed('char', 'IdentifierId').withColumnRenamed('zone', 'zoneId')\
            .withColumnRenamed('timestamp', 'log_timestamp')

# InGame_Log_Data(인게임 캐릭터 위치 로그 데이터)
zones_df = zones_df.withColumnRenamed('Zone_Name', 'zoneId')

# Churn_Label_Log_Data(이탈 여부 태그 로그 데이터)
churners_df = churners_df.withColumnRenamed('char', 'IdentifierId')\
                         .withColumnRenamed('timestamp', 'churn_timestamp')

churner_pandas_df = churner_pandas_df.rename(columns={'char': 'IdentifierId','timestamp': 'churn_timestamp'})
wow_log_pandas_df = wow_log_pandas_df.rename(columns={'char': 'IdentifierId','zone': 'zoneId','timestamp': 'log_timestamp'})
zones_pandas_df =zones_pandas_df.rename(columns={'Zone_Name': 'zoneId'})
churner_pandas_df.columns

#연습용 셀
#printschema()

wow_log_df = wow_log_df.withColumn('log_timestamp',unix_timestamp(col('log_timestamp'),'E MMM dd HH:mm:ss z yyyy'))
wow_log_df = wow_log_df.withColumn("log_timestamp",to_timestamp(col('log_timestamp')))
display(wow_log_df)

churners_df = churners_df.withColumn('churn_timestamp',unix_timestamp(col('churn_timestamp'),'E MMM dd HH:mm:ss z yyyy'))
churners_df = churners_df.withColumn('churn_timestamp',to_timestamp(col('churn_timestamp')))

display(churners_df)

#연습용 셀

# Character_Log_Data(사용자 캐릭터 로그 데이터)
wow_log_df = wow_log_df.withColumn('IdentifierId', regexp_extract(col('IdentifierId'), r"\(([^()]+)\)", 1))\
.withColumn('zoneId', regexp_extract(col('zoneId'), r"\(([^()]+)\)", 1)) #8,654,936
# InGame_Log_Data(인게임 캐릭터 위치 로그 데이터)
zones_df = zones_df.withColumn('zoneId', regexp_extract(col('zoneId'), r"\(([^()]+)\)", 1)) #160

# Churn_Label_Log_Data(이탈 여부 태그 로그 데이터)
churners_df = churners_df.withColumn('IdentifierId', regexp_extract(col('IdentifierId'), r"\(([^()]+)\)", 1)) #14,579

#연습용 셀

# Character_Log_Data(사용자 캐릭터 로그 데이터) ∩ InGame_Log_Data(인게임 캐릭터 위치 로그 데이터)
wow_log_join = wow_log_df.join(zones_df, wow_log_df.zoneId == zones_df.zoneId).drop(zones_df.zoneId)
print('Character ∩ InGame =', wow_log_join.count(), len(wow_log_join.columns))

# Character_Log_Data(사용자 캐릭터 로그 데이터) ∩ InGame_Log_Data(인게임 캐릭터 위치 로그 데이터) ∩ Churn_Label_Log_Data(이탈 여부 태그 로그 데이터)
wow_log_result_join = wow_log_join.join(churners_df, wow_log_join.IdentifierId == churners_df.IdentifierId).drop(churners_df.IdentifierId)
print('Character ∩ InGame ∩ Churn_Label =', wow_log_result_join.count(), len(wow_log_result_join.columns))
display(wow_log_result_join)

#pandas dataframe으로 하는 join

#df_pandas_INNER_JOIN = pd.merge(wow_log_pandas_df, churner_pandas_df, left_on='IdentifierId', right_on='IdentifierId', how='inner')
#df_pandas_INNER_JOIN_result = pd.merge(df_pandas_INNER_JOIN, zones_pandas_df, left_on='zoneId', right_on='zoneId', how='inner')
#print(df_pandas_INNER_JOIN_result)


#연습용 셀

wow_log_result_join.createOrReplaceTempView("pysparkdftemptable") #temptable에 저장
wow_log_result_join.write.mode("overwrite").saveAsTable('sparkwowdataset') # sparkpool에 저장
wow_log_result_join.write.mode("overwrite").save('abfss://aaafilesystem@aaatraing.dfs.core.windows.net/sparkpooldata/churnersdf_yj3.csv', format='csv', header=True) #ADLS에 저장

%%spark
val scala_df = spark.sqlContext.sql ("select * from pysparkdftemptable")
scala_df.write.mode("overwrite").synapsesql("aaasqlpool.dbo.notebooksqlwowdata", Constants.INTERNAL) //Sql pool에 저장


