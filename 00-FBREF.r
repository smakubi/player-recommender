# Databricks notebook source
# MAGIC %run ./RESOURCES/VARIABLES

# COMMAND ----------

install.packages("devtools")
devtools::install_github("JaseZiv/worldfootballR")
library(worldfootballR) 

# COMMAND ----------

mapped_players <- player_dictionary_mapping()
dplyr::glimpse(mapped_players)

# COMMAND ----------


write.csv(players, "/Volumes/smakubi/ai_scout/scouting/mapped_players.csv", row.names = FALSE)
