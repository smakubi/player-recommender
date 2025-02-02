# Databricks notebook source
# MAGIC %run ./RESOURCES/VARIABLES

# COMMAND ----------

# MAGIC %pip install openai tabulate --upgrade typing_extensions
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import requests
import pandas as pd
from datetime import datetime
from openai import OpenAI
from bs4 import BeautifulSoup
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, lit
from pyspark.sql.types import StringType, StructType, StructField, ArrayType, IntegerType

# COMMAND ----------

# Fetch data from Spark
df1 = spark.sql(f"""
                SELECT PlayerFBref, UrlFBref, TmPos FROM {catalog}.{schema}.mapped_players LIMIT 1000
                """)

# COMMAND ----------

DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

# Define the API client
client = OpenAI(api_key=DATABRICKS_TOKEN,
                base_url=f"{workspace_url}/serving-endpoints") # PUT YOUR API KEY here

# COMMAND ----------

# Function to fetch and process data
def get_scouting_report(player_name, url, attrs):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Get the player's age, team, league, and position
    try:
        position = (
            soup.select_one('p:-soup-contains("Position")')
            .text.split(":")[-2]
            .split(" ")[0]
            .strip()
        )
    except AttributeError:
        position = "Unknown"

    try:
        birthday = soup.select_one('span[id="necro-birth"]').text.strip()
    except AttributeError:
        birthday = "Unknown"

    # Calculate the player's age
    try:
        age = (datetime.now() - datetime.strptime(birthday, "%B %d, %Y")).days // 365
    except:
        age = "Unknown"

    try:
        team = soup.select_one('p:-soup-contains("Club")').text.split(":")[-1].strip()
    except:
        team = "Unknown"

    # Read the scouting data table
    try:
        df = pd.read_html(url, attrs={"id": attrs})[0]
        df = df.dropna(subset=["Statistic"])
    except:
        df = pd.DataFrame()

    # Create the prompt for the AI model
    prompt = f"""
                Do not include your disclaimer such as "I apologize, but with the limited information provided, I'll have to make some assumptions." Just give me the result I want. Be as objective and as accurate as possible. This is extremely important.I need you to create a scouting report on {player_name}. Provide me with their strengths and weaknesses and summary recommendations for the team.
                Here is the data I have on this player:
                Player: {player_name}.
                Position: {position}. 
                Age: {age}.
                Club: {team}.
                Statistics:{df.to_markdown()}
                Return the scouting report in the following format (Do not skip lines):
                Scouting Report for {player_name}
                Position: {position}
                Age:{age}
                Current Club: {team}
                Current League: 
                Overall Rating (Out of 10):
                Strengths: < a paragraph of 1 to 3 strengths. Include hard metrics and statistics from {df.to_markdown()} above>
                Weaknesses: < a paragraph of 1 to 3 weaknesses. Include hard metrics and statistics from {df.to_markdown()} above>
                Summary: < a summary of the player's overall performance and whether or not he would be beneficial to the team>
            """

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a professional football (soccer) scout.",
            },
            {
                "role": "user", 
                "content": prompt},
        ],
        model="databricks-meta-llama-3-70b-instruct",
    )

    return response.choices[0].message.content

# COMMAND ----------

# Collect data to driver
data = df1.collect()

# Process each row and get scouting reports
results = []
for row in data:
    player_name = row.PlayerFBref
    pos = row.TmPos
    url = row.UrlFBref
    report = get_scouting_report(player_name, url, "scout_summary_AM")
    results.append((player_name, pos, url, report))

# Create a new DataFrame with the results
schema = StructType([
    StructField("PlayerFBref", StringType(), True),
    StructField("TmPos", StringType(), True),
    StructField("UrlFBref", StringType(), True),
    StructField("scouting_report", StringType(), True)
])

result_df = spark.createDataFrame(results, schema)

# COMMAND ----------

# Show the result
display(result_df)

# COMMAND ----------

table_name = "scouting_reports"
result_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(table_name)
