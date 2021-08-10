%%spark

val df = spark.read.sqlanalytics("aaasqlpool.dbo.dataflow_table") 

df.write.mode("overwrite").saveAsTable("default.dataflowsparkwowdataset")


