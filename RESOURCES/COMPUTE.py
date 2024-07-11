# Databricks notebook source
import requests

API_URL = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
EMAIL = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
USER_NAME = EMAIL.split('@')[0]
URL = f"{API_URL}/api/2.0/clusters/create"


CLUSTER_CONFIG = {
    "num_workers": 0,
    "cluster_name": f"{USER_NAME}'s LLM Cluster",
    "spark_version": "15.2.x-cpu-ml-scala2.12",
    "spark_conf": {
        "spark.master": "local[*, 4]",
        "spark.databricks.cluster.profile": "singleNode"
    },
    "aws_attributes": {
        "first_on_demand": 1,
        "availability": "ON_DEMAND",
        "zone_id": "auto",
        "spot_bid_price_percent": 100,
        "ebs_volume_count": 0
    },
    "node_type_id": "r5d.24xlarge",
    "driver_node_type_id": "r5d.24xlarge",
    "ssh_public_keys": [],
    "custom_tags": {
        "ResourceClass": "SingleNode"
    },
    "spark_env_vars": {},
    "autotermination_minutes": 120,
    "init_scripts": [],
    "single_user_name": f"{EMAIL}",
    "enable_local_disk_encryption": False,
    "data_security_mode": "SINGLE_USER",
    "runtime_engine": "STANDARD"
}



response = requests.post(URL, json=CLUSTER_CONFIG, headers={"Authorization": f"Bearer {TOKEN}"})


if response.status_code == 200:
    print("Cluster created successfully")
else:
    print("Failed to create cluster:", response.json())
