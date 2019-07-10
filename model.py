import preprocess as pp

import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import length

from pyspark.ml.feature import Tokenizer,StopWordsRemover,CountVectorizer,IDF,StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline


training_data = 'train.iob'

output,labels = pp.preprocess(training_data)
df = pd.DataFrame.from_dict(output, orient='columns')


spark = SparkSession.builder.appName('nlp').getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

spark_df = spark.createDataFrame(df)
spark_df = spark_df.drop("tags")
spark_df = spark_df.withColumn('length',length(spark_df['text']))


tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
idf = IDF(inputCol="c_vec", outputCol="tf_idf")
to_num = StringIndexer(inputCol='intent',outputCol='label', handleInvalid='keep')
clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')

data_prep_pipe = Pipeline(stages=[to_num,tokenizer,stopremove,count_vec,idf,clean_up])
cleaner = data_prep_pipe.fit(spark_df)

clean_data = cleaner.transform(spark_df)
clean_data = clean_data.select(['label','features'])

nb = NaiveBayes()

nb_model = nb.fit(clean_data)
nb_model.save("AtisIntentModel")