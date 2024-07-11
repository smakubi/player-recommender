# Databricks notebook source
# MAGIC %run ./RESOURCES/VARIABLES

# COMMAND ----------

install.packages("devtools")
install.packages("glue")
devtools::install_github("JaseZiv/worldfootballR")
library(worldfootballR) 

# COMMAND ----------

mapped_players <- player_dictionary_mapping()
dplyr::glimpse(mapped_players)

# COMMAND ----------

# change samakubi to your name from your email: name@domain.com
volume_path <- "/Volumes/llm_workshop/samakubi_genai_scout/scouting/mapped_players.csv"

# COMMAND ----------

write.csv(mapped_players, volume_path, row.names = FALSE)

# COMMAND ----------

# NOW MANUALLY GO IN AND CREATE TABLE NAMED mapped_players FROM THE VOLUME LOCATION
# Make Sure Your Cluster Is Selected. Be Sure to Select Your Catalog and Schema
