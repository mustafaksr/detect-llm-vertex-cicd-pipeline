"""Lightweight component data extraction function."""
from kfp.v2.dsl import component

@component(
    base_image="python:3.11",
    output_component_file="detect_llm_kfp_data_extract.yaml",
    packages_to_install=["google-cloud-bigquery"],
)
def process_data( project: str ):
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
    PROJECT_ID = project
    dataset_id = "detect_llm_ds_bq"
    bigquery_client = bigquery.Client(project=PROJECT_ID)
    def preprocess_text(bigquery_client, PROJECT_ID,dataset_id = dataset_id,table_name="test"):
        job_config = bigquery.QueryJobConfig(
            destination=f"{PROJECT_ID}.{dataset_id}.{table_name}",
            write_disposition="WRITE_TRUNCATE",
        )
        special_chars = r"[\\.\,\?\!\"\#\$\%\(\)\*\+\:\;\<\=\>\@\^\_\`\|\~\'\{\}]"
        sql = f"""SELECT LOWER(REGEXP_REPLACE(text, r"{special_chars}", ' ')) AS text, label FROM `detect-llm-cicd.{dataset_id}.{table_name}`"""    
        query_job = bigquery_client.query(sql, job_config=job_config)
        query_job.result()
        print(f"Preprocess Complete, {table_name}")

    for table_name in ["training","validation","test"]:
        preprocess_text(bigquery_client, PROJECT_ID,dataset_id = dataset_id,table_name=table_name)


    
    
