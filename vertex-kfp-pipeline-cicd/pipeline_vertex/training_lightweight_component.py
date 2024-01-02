# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""Lightweight component training function."""
from kfp.v2.dsl import component


@component(
    base_image="python:3.11",
    output_component_file="detect_llm_kfp_train_and_deploy.yaml",
    packages_to_install=["google-cloud-aiplatform"],
)
def train_and_deploy(
    project: str,
    location: str,
    container_uri: str,
    serving_container_uri: str,
    training_file_path: str,
    validation_file_path: str,
    test_file_path: str,
    staging_bucket: str,
    dropout :float,
    embedding_dim  :int,
    hidden_dim : int,
    max_features :int,
    sequence_length :int 
):

    # pylint: disable-next=import-outside-toplevel
    from google.cloud import aiplatform

    aiplatform.init(
        project=project, location=location, staging_bucket=staging_bucket
    )
    job = aiplatform.CustomContainerTrainingJob(
        display_name="detect_llm_kfp_training",
        container_uri=container_uri,
        command=[
            "python",
            "train.py",
            f"--training_file_path={training_file_path}",
            f"--validation_file_path={validation_file_path}",
            f"--test_file_path={test_file_path}",
            f"--dropout={dropout}",
            f"--embedding_dim={embedding_dim}",
            f"--hidden_dim={hidden_dim}",
            f"--max_features={max_features}",
            f"--sequence_length={sequence_length}",
            "--nohptune",
        ],
        staging_bucket=staging_bucket,
        model_serving_container_image_uri=serving_container_uri,
    )

    model = job.run(replica_count=1, model_display_name="detect_llm_kfp_model")
    
    endpoint = model.deploy(  # pylint: disable=unused-variable
        traffic_split={"0": 100},
        machine_type="n1-standard-2",
    )
