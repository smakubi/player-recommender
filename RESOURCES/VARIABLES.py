# Databricks notebook source
# MAGIC %md
# MAGIC # Rename this file variables.py after you fill in your variables.

# COMMAND ----------

hf_token = "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"

# COMMAND ----------

#GENERAL
user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get().split("@")[0]

managed_storage_path = "s3://databricks-workspace-stack-b2583-bucket/unity-catalog/1981868249518092"
catalog = 'llm_workshop'
schema = f'{user_name}_genai_scout'
volume_name = 'scouting'

secrets_scope = ''
secrets_hf_key_name = ''

workspace_url = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
base_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog} MANAGED LOCATION '{managed_storage_path}/llm'")
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
spark.sql(f"USE SCHEMA {schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_name}") 

# COMMAND ----------

#DATA PREP
volume_path=f"/Volumes/{catalog}/{schema}/{volume_name}"

chunk_size=200
chunk_overlap=50

sync_table_name = 'player_documents_for_sync'

# COMMAND ----------

#EMBEDDING MODEL
embedding_model_name=f'{user_name}-e5-small-v2'
registered_embedding_model_name = f'{catalog}.{schema}.{embedding_model_name}'
embedding_endpoint_name = f'{user_name}-e5-small-v2' 

# COMMAND ----------

#VECTOR SEARCH
vs_endpoint_name='playerscout-report-endpoint'

vs_index = f'{user_name}_playerscout_report_index'
vs_index_fullname = f"{catalog}.{schema}.{vs_index}"

sync_table_fullname = f"{catalog}.{schema}.{sync_table_name}"

# COMMAND ----------

#LLM SERVING
llm_model_name=f'{user_name}-llama-2-7b-hf-chat'
registered_llm_model_name=f'{catalog}.{schema}.{llm_model_name}'
llm_endpoint_name = f'{user_name}-llama-2-7b-hf-chat'

