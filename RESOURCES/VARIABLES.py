# Databricks notebook source
# MAGIC %md
# MAGIC # Rename this file variables.py after you fill in your variables.

# COMMAND ----------

hf_token = ""

# COMMAND ----------

#GENERAL
catalog='smakubi'
schema='ai_scout'

secrets_scope=''
secrets_hf_key_name=''

workspace_url = "https://" + spark.conf.get("spark.databricks.workspaceUrl") 
base_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

# COMMAND ----------

#DATA PREP
volume_path="/Volumes/smakubi/ai_scout/scouting"

chunk_size=200
chunk_overlap=50

sync_table_name = 'player_documents_for_sync'

# COMMAND ----------

#EMBEDDING MODEL
embedding_model_name='semmanuel-e5-small-v2'
registered_embedding_model_name = f'{catalog}.{schema}.{embedding_model_name}'
embedding_endpoint_name = 'semmanuel-e5-small-v2' 

# COMMAND ----------

#VECTOR SEARCH
vs_endpoint_name='playerscout-report-endpoint'

vs_index = 'semmanuel_playerscout_report_index'
vs_index_fullname = f"{catalog}.{schema}.{vs_index}"

sync_table_fullname = f"{catalog}.{schema}.{sync_table_name}"

# COMMAND ----------

#LLM SERVING
llm_model_name='semmanuel-llama-2-7b-hf-chat'
registered_llm_model_name=f'{catalog}.{schema}.{llm_model_name}'
llm_endpoint_name = 'semmanuel-llama-2-7b-hf-chat'

