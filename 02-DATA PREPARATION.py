# Databricks notebook source
# MAGIC %run ./RESOURCES/VARIABLES

# COMMAND ----------

import huggingface_hub
# hf_token = dbutils.secrets.get(f"{secrets_scope}", f"{secrets_hf_key_name}")
from huggingface_hub import login
login(token=hf_token)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T
from transformers import AutoTokenizer, LlamaTokenizer
from typing import Iterator
import pandas as pd
from langchain.text_splitter import TokenTextSplitter


# COMMAND ----------

df = spark.table(f"{catalog}.{schema}.scouting_reports")
display(df)

# COMMAND ----------

df = df.withColumnsRenamed({"PlayerFBref": "player_name", "UrlFBref": "fbref_url", "TmPos": "player_position"})
display(df)

# COMMAND ----------

df = df.withColumn("document", F.concat_ws("\n\n", F.concat_ws(" ", F.lit("player name:"), F.col("player_name")), F.concat_ws(" ", F.lit("player scouting report:"), F.col("scouting_report")), F.concat_ws(" ", F.lit("player position:"), F.col("player_position"))))
                   
display(df) 

# COMMAND ----------

print((df.count(), len(df.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### 2A. DATA EXPLORATORY ANALYSIS
# MAGIC
# MAGIC Let's explore the data and determine the average number of tokens per document. This is important to understand because LLMs have token input limits; and in this RAGs architecture we will be passing plot summaries as context. Because we are going to be using [Llama-2-7b-chat from hugging face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), we should use the tokenizer used in that model. You can access all Llama models through Hugging Face through [this](https://huggingface.co/meta-llama) link. 

# COMMAND ----------

tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat')

# COMMAND ----------

tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

# COMMAND ----------

alphonso_davies_example = """
player name: Alphonso Davies
player scouting report: Here is the scouting report for Alphonso Davies:
Scouting Report for Alphonso Davies
Position: Left Wingback/Left Midfielder
Age: 21
Team: Bayern Munich
League: Bundesliga
Overall Rating: 8.5/10
Strengths:
Alphonso Davies is an explosive and dynamic player who possesses electrifying pace, agility, and acceleration. He is a nightmare to defend against, with the ability to beat opponents with his quick feet, speed, and clever movement off the ball. Additionally, Davies is an exceptional 1v1 defender, using his quick reactions and anticipation to win the ball back and start counter-attacks. His attacking prowess is also noteworthy, as he consistently delivers high-quality crosses and passes into the box, creating scoring opportunities for his teammates.
Weaknesses:
While Davies is an elite athlete, he can sometimes be prone to reckless decision-making, leading to turnovers and counter-attacks against his team. Additionally, his defensive positioning and awareness can be suspect at times, leaving him exposed to opposition attacks. Finally, Davies can struggle with consistency, having the occasional off-game where his usually electric pace and agility are not on display.
Summary:
Alphonso Davies is an elite talent with the potential to dominate games on both the attacking and defensive ends. His pace, agility, and attacking prowess make him a nightmare to defend against, and his defensive skills are rapidly improving. While he has some areas for growth, Davies is an excellent addition to any team and would be a valuable asset to our squad. His ability to play both left wingback and left midfield makes him a versatile option, and his ceiling is incredibly high.
player position: Left-Back
"""

# COMMAND ----------

print(f"length of Alphonso Davies document page: {len(tokenizer.encode(alphonso_davies_example))}")

# COMMAND ----------

# UDF to determine the number of tokens using the llama-2-7b tokenizer

@F.pandas_udf("long")
def num_tokens_llama(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    login(token=hf_token)
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    try:
        for x in batch_iter:
            yield x.apply(lambda s: len(tokenizer.encode(s)))
    finally:
        pass

# COMMAND ----------

df = (
    df
    .withColumn("document_num_chars", F.length("document"))
    .withColumn("document_num_words", F.size(F.split("document", "\\s")))
    .withColumn("document_num_tokens_llama", num_tokens_llama("document"))
)

# COMMAND ----------

display(df)

# COMMAND ----------

df.createOrReplaceTempView("documents")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   AVG(document_num_tokens_llama) AS mean_tokens,
# MAGIC   MAX(document_num_tokens_llama) AS max_tokens,
# MAGIC   MIN(document_num_tokens_llama) AS min_tokens,
# MAGIC   SUM(CASE WHEN document_num_tokens_llama>3500 THEN 1 ELSE 0 END) AS documents_3500
# MAGIC FROM documents

# COMMAND ----------

#to keep things simple for this workshop we are going to remove all documents with a token limit about 2000. This is because Llama-2 has a token input limit of 4096 tokens. 

df=df.filter(df.document_num_tokens_llama <=2000)

# COMMAND ----------

#write to delta table
df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.player_documents_silver")

# COMMAND ----------

# delta table to use
df1 = spark.sql(
    f"""SELECT fbref_url AS player_id, document, player_name, document_num_tokens_llama, document_num_chars, 
document_num_words FROM {catalog}.{schema}.player_documents_silver;"""
)

# COMMAND ----------

#creating a subset of data for vector search delta sync
df1.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{catalog}.{schema}.{sync_table_name}")

# COMMAND ----------

spark.sql(f'''
          ALTER TABLE {catalog}.{schema}.{sync_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
          ''')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3. DATA CHUNKING
# MAGIC
# MAGIC We won't be using chunking in this rag bot, but I wanted to include how you would do this. This is a good strategy if you need extra control over token input. 

# COMMAND ----------

print(f"chunk_size: {chunk_size}")
print(f"chunk_overlap: {chunk_overlap}")

# COMMAND ----------

def split_documents(dfs: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    fn = lambda s: text_splitter.split_text(s)
    for df in dfs:
        df.loc[:, "text"] = df.loc[:, "scout_report"].apply(fn)
        df = df.loc[:, ["player_id","text"]]
        df = df.explode("text").reset_index().rename(columns={'index' : 'chunk_index'})
        df['chunk_index'] = df.groupby('chunk_index').cumcount()
        yield df.loc[:, ["player_id", "chunk_index", "text"]]

# COMMAND ----------

player_df=spark.table(f"{catalog}.{schema}.player_documents_silver")

metadata_df = (
    player_df.select([
        F.col("fbref_url").alias("player_id"),
        "player_name",
        "document_num_tokens_llama", 
        "document_num_chars", 
        "document_num_words",
        "document"]))

# COMMAND ----------

results_schema = T.StructType([
    T.StructField("player_id", T.StringType()),
    T.StructField("chunk_index", T.LongType()),
    T.StructField("text", T.StringType())
])

results = (
    player_df.mapInPandas(split_documents, results_schema)
    .withColumn("id", F.concat_ws("_", 
        F.col("player_id"), 
        F.col("chunk_index").cast("string")))
    .join(metadata_df, "player_id")
)

display(results)

# COMMAND ----------

display(results)

# COMMAND ----------

results.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.player_documents_silver_chunked")
