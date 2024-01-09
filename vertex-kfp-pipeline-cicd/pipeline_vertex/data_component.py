"""Lightweight component data extraction function."""
from kfp.v2.dsl import component

@component(
    base_image="python:3.11",
    output_component_file="detect_llm_kfp_data_extract.yaml",
    packages_to_install=["google-cloud-bigquery"],
)
def extract_data(project: str):
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
    PROJECT_ID = project
    dataset_id = "detect_llm_ds_bq"
    def shuffle_and_split_data(bigquery_client, PROJECT_ID,dataset_id = dataset_id):
        # Shuffle Data
        job_config = bigquery.QueryJobConfig(
            destination=f"{PROJECT_ID}.{dataset_id}.shuffle_raw",
            write_disposition="WRITE_TRUNCATE",
        )
        sql = f'SELECT * \
                FROM `{PROJECT_ID}.{dataset_id}.raw_data` ORDER BY RAND()'
        query_job = bigquery_client.query(sql, job_config=job_config)
        query_job.result()
        print("0.Data Shuffling Done")

        # Create Split Data
        split_queries = [
            (f"{PROJECT_ID}.{dataset_id}.training", [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            (f"{PROJECT_ID}.{dataset_id}.test", [1, 12]),
            (f"{PROJECT_ID}.{dataset_id}.validation", [11, 12]),
            (f"{PROJECT_ID}.{dataset_id}.training", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ]

        for table_id, mod_values in split_queries:
            job_config = bigquery.QueryJobConfig(
                destination=table_id, write_disposition="WRITE_TRUNCATE"
            )
            sql = f'SELECT * \
                    FROM `{PROJECT_ID}.{dataset_id}.shuffle_raw` AS train \
                    WHERE MOD(ABS(FARM_FINGERPRINT(TO_JSON_STRING(train))), 12) IN ({",".join(map(str, mod_values))})'
            query_job = bigquery_client.query(sql, job_config=job_config)
            query_job.result()
            print(f"{split_queries.index((table_id, mod_values)) + 1}.Step Done")

    def extract_data_from_table(bigquery_client, PROJECT_ID, table_id, destination_uri,dataset_id):
        extract_job = bigquery_client.extract_table(
            f"{PROJECT_ID}.{dataset_id}.{table_id}",
            destination_uri,
            location="US",
        )
        extract_job.result()
        print(f"{table_id} Data Extracted")

    def extract_data_in(PROJECT_ID,dataset_id):
        ARTIFACT_STORE = f"gs://{PROJECT_ID}/detect-llm"
        DATA_ROOT = f"{ARTIFACT_STORE}/data"
        TRAINING_FILE_PATH = f"{DATA_ROOT}/training/train_df.csv"
        VALIDATION_FILE_PATH = f"{DATA_ROOT}/validation/validation_df.csv"
        TEST_FILE_PATH = f"{DATA_ROOT}/test/test_df.csv"

        bigquery_client = bigquery.Client(project=PROJECT_ID)

        

        try:
            bigquery_client.get_dataset(dataset_id)
        except NotFound:
            print(f"Dataset {dataset_id} not found. Please make sure it exists.")

            dataset = bigquery.Dataset(dataset_id)
            dataset.location = "US"
            dataset = bigquery_client.create_dataset(dataset, timeout=30) 
            print(f"Created dataset, id:{dataset.dataset_id}")


        shuffle_and_split_data(bigquery_client, PROJECT_ID)

        extract_data_from_table(bigquery_client, PROJECT_ID, "training", TRAINING_FILE_PATH, dataset_id)
        extract_data_from_table(bigquery_client, PROJECT_ID, "validation", VALIDATION_FILE_PATH, dataset_id)
        extract_data_from_table(bigquery_client, PROJECT_ID, "test", TEST_FILE_PATH, dataset_id)

    extract_data_in(PROJECT_ID,dataset_id)
