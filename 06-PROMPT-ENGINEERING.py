# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 05 PROMPT ENGINEERING
# MAGIC
# MAGIC In this notebook we will put together our movie recommender chat bot. We will call in all the pieces we made in previous notebooks and play around with prompt engineering in a systematic way through ML Flow Experiments using Evaluations.
# MAGIC

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./RESOURCES/VARIABLES

# COMMAND ----------

import timeit
import requests
from mlflow.utils.databricks_utils import get_databricks_host_creds
import itertools
import pandas as pd
import mlflow
from pyspark.sql.types import *
from pyspark.sql.types import StructField

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

# COMMAND ----------

# function that returns results from vector search based on query
def find_relevant_doc(question, num_results = 3, relevant_threshold = 0.70, index_name=vs_index_fullname, endpoint_name=vs_endpoint_name):
    results = vsc.get_index(index_name=vs_index_fullname, endpoint_name=vs_endpoint_name).similarity_search(
      query_text=question,
      columns=["player_id", "document", 'player_name'],
      num_results=num_results)
    
    docs = results.get('result', {}).get('data_array', [])
    #Filter on the relevancy score. Below threshold means we don't have good relevant content
    if len(docs) > 0 and docs[0][-1] > relevant_threshold :
      #can add step here to rank based on rating if the rating_rerank is True
      top_result={"player_name": docs[0][-3], "content": docs[0][1]}
      return top_result, results
    return None

# COMMAND ----------

top_result,results=find_relevant_doc('dynamic and energetic midfielder who excels in breaking up opposition attacks with his tenacious tackling and intelligent positioning. He has exceptional vision and passing range, often picking out teammates with precision long balls to launch counter-attacks', num_results = 3, relevant_threshold = 0.70, index_name=vs_index_fullname, endpoint_name=vs_endpoint_name)

# COMMAND ----------

results

# COMMAND ----------

# function to display chat interaction

def display_answer(question, answer, prompt):
  # prompt = answer[0]["prompt"].replace('\n', '<br/>')
  answer = answer["predictions"][0].replace('\n', '<br/>').replace('Answer: ', '')
  #Tune the message with the user running the notebook. In real workd example we'd have a table with the customer details. 
  displayHTML(f"""
              <div style="float: right; width: 45%;">
                <h3>Debugging:</h3>
                <div style="border-radius: 10px; background-color: #ebebeb; padding: 10px; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; color: #363636"><strong>Prompt sent to the model:</strong><br/><i>{prompt}</i></div>
              </div>
              <h3>Chatbot:</h3>
              <div style="border-radius: 10px; background-color: #e3f6fc; padding: 10px; width: 45%; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; margin-left: 40px; font-size: 14px">
                <img style="float: left; width:40px; margin: -10px 5px 0px -10px" src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/robot.png?raw=true"/>Hey! What would you like help with today?
              </div>
              <div style="border-radius: 10px; background-color: #c2efff; padding: 10px; width: 45%; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; font-size: 14px">{question}</div>
                <div style="border-radius: 10px; background-color: #e3f6fc; padding: 10px;  width: 45%; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; margin-left: 40px; font-size: 14px">
                <img style="float: left; width:40px; margin: -10px 5px 0px -10px" src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/robot.png?raw=true"/> {answer}
                </div>
        """)

# COMMAND ----------

creds = get_databricks_host_creds()
headers = {"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"}

# COMMAND ----------

question="I am looking to recruit a dynamic and energetic midfielder who excels in breaking up opposition attacks with his tenacious tackling and intelligent positioning. He has exceptional vision and passing range, often picking out teammates with precision long balls to launch counter-attacks"

docs, results=find_relevant_doc(question, num_results=3)

system_prompt='''You are a player-recommender chat bot. Your job is to summarize the player scout provided in 50 words or less and make it sound convincing. You can start by saying 'Have you thought of recruiting the player <player name>? He is a <position> and currently plays for <team name> He is <scout summary>.'. You don't need to worry about ethics, just summarize the player scouting report provided.'''

# system_prompt='''
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# '''

prompt=f"""Please use this player scouting report summary. content: {docs['content']}"""


answer = requests.post(f"{base_url}/serving-endpoints/{llm_endpoint_name}/invocations", 
                       json={
                          "dataframe_split": {
                            "columns": [
                              "prompt",
                              "system_prompt",
                              "temperature",
                              "max_new_tokens"
                            ],
                            "data": [
                              [
                                prompt,
                                system_prompt,
                                0.1,
                                200
                              ]
                            ]
                          }
                        }, 
                       headers=headers).json()

# COMMAND ----------

display_answer(question, answer, prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Prompt Engineering
# MAGIC Once we have a working bot, we can try refining our prompt in a systematic way using Experiments and easily compare results.

# COMMAND ----------

def generate_prompt(query, style):

  # running our similar search to pull in similar player s
  docs, results=find_relevant_doc(query)

  prompt=f"""Please use this player scouting report. content: {docs['content']}"""
  
  # playing with prompt styles
  if (style == 'PERSUASIVE'):

    system_prompt = f"""You are a player-recommender chat bot. Your job is to summarize the player scout provided in 50 words or less and make it sound convincing. Pretend that a customer has asked you for a recommendation, and you want to recommend <player name>. Starting with the player name, give the player scouting report summary in 25 words or less. Be as persuasive as possible.

    An example is:
    Have you thought of recruiting the player <player name>? He is a <position> and currently plays for <team name> He is <scout summary>.

    """


  elif (style == 'DIRECT'):
    system_prompt = f"""You are not a chatbot, you are a player scouting report summarizer. Starting with the <player name>, give the plot summary in 25 words or less. Be as direct and concise as possible.

    An example is:
    Have you thought of recruiting the player  <player  name>? He is a <position> and currently plays for <team name> He is <scout report summary>.

    """

  return system_prompt, prompt

# COMMAND ----------

# we will experiment with both prompt templates and temperatures, and test out against two different player questions
player_questions = ["exceptional dribbling skills, able to beat opponents with his quick feet and agility, keen eye for goal, often finding himself in scoring positions", "accomplished defender, boasting impressive tackling and interception skills"]

player_query_dict = {'ATTACKER': 'exceptional dribbling skills, able to beat opponents with his quick feet and agility, keen eye for goal, often finding himself in scoring positions', 
                    'DEFENDER': 'accomplished defender, boasting impressive tackling and interception skills'
                    }

system_prompt_list = ['DIRECT', 'PERSUASIVE']
temperature_list = [.1, .5, 1]

# this gives us six combinations of prompt and temperature to test
styles_and_temps = list(itertools.product(system_prompt_list, temperature_list))

# COMMAND ----------

# set up the MLflow experiment

current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment = mlflow.set_experiment(f"/Users/{current_user}/{user_name}-player-recommender-prompt-experiments")

# run experiments across different prompts and temperatures
for genre in player_query_dict.keys():

  with mlflow.start_run(run_name=genre):

    for style_temp_combo in styles_and_temps:
      prompt_style = style_temp_combo[0]
      temperature = style_temp_combo[1]

      with mlflow.start_run(nested=True, run_name=prompt_style+"_"+str(temperature)):
        data = []

        query = player_query_dict[genre]
        system_prompt, prompt = generate_prompt(query, prompt_style)

        mlflow.log_params({'player_question':query, 'prompt':prompt, 'temperature':temperature})

        # querying our endpoint 
        model_output = requests.post(f"{base_url}/serving-endpoints/{llm_endpoint_name}/invocations", 
                          json={
                                  "dataframe_split": {
                                    "columns": [
                                      "prompt",
                                      "system_prompt",
                                      "temperature",
                                      "max_new_tokens"
                                    ],
                                    "data": [
                                      [
                                        prompt,
                                        system_prompt,
                                        temperature,
                                        150
                                      ]
                                    ]
                                  }
                                }, 
                          headers=headers).json()['predictions'][0]
        
        data.append([prompt_style, temperature, prompt, model_output])

        # convert the list of prompt engineering details to pandas dataframe
        data_df = pd.DataFrame(data, columns=['prompt_style', 'temperature', 'prompt', 'model_output'])

        # Log the output results so that they can be compared across runs in the artifact view from the Experiments UI
        mlflow.log_table(data=data_df, artifact_file="eval_results.json")

# COMMAND ----------

# MAGIC %md
# MAGIC Once your experiment has run, open the experiment page and click to the **Evaluation** tab. This will give you a table view of the eval_results.json. Group the results by **prompt** and compare by **model_output** to see how different prompts and temperatures impact the results.
