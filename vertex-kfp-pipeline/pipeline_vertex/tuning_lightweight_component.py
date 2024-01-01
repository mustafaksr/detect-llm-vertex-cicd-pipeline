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
"""Lightweight component tuning function."""
from typing import NamedTuple

from kfp.v2.dsl import component


@component(
    base_image="python:3.11",
    output_component_file="detect_llm_kfp_tune_hyperparameters.yaml",
    packages_to_install=["google-cloud-aiplatform"],
)
def tune_hyperparameters(
    project: str,
    location: str,
    container_uri: str,
    training_file_path: str,
    validation_file_path: str,
    test_file_path: str,

    staging_bucket: str,
    max_trial_count: int,
    parallel_trial_count: int,
) -> NamedTuple(
    "Outputs",
    [ 
     ("best_dropout", float), 
     ("best_embedding_dim", int),
     ("best_hidden_dim", int),
     ("best_max_features", int),
     ("best_roc_auc", float), 
     ("best_sequence_length", int),],
):

    # pylint: disable=import-outside-toplevel
    from google.cloud import aiplatform
    from google.cloud.aiplatform import hyperparameter_tuning as hpt

    aiplatform.init(
        project=project, location=location, staging_bucket=staging_bucket
    )

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",

            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": container_uri,
                "args": [
                    f"--training_file_path={training_file_path}",
                    f"--validation_file_path={validation_file_path}",
                    f"--test_file_path={test_file_path}",
                    "--hptune",
                ],
            },
        }
    ]

    custom_job = aiplatform.CustomJob(
        display_name="detect_llm_kfp_trial_job",
        worker_pool_specs=worker_pool_specs,
    )

    hp_job = aiplatform.HyperparameterTuningJob(
        display_name="detect_llm_kfp_tuning_job",
        custom_job=custom_job,
        metric_spec={
            "roc_auc": "maximize",
        },
        parameter_spec={
            "hidden_dim": hpt.DiscreteParameterSpec(    
                values=[32,64,96], scale="linear"
            ),
            "dropout": hpt.DiscreteParameterSpec(
                values=[0.2, 0.5], scale="linear"
            ),
            "embedding_dim": hpt.DiscreteParameterSpec(
                values=[32,64,96], scale="linear"
            ),
            "sequence_length": hpt.DiscreteParameterSpec(
                values=[32,64,96], scale="linear"
            ),
            "max_features": hpt.DiscreteParameterSpec(
                values=[5000,6000], scale="linear"
            ),




        },
        max_trial_count=max_trial_count,
        parallel_trial_count=parallel_trial_count,
    )

    hp_job.run()

    metrics = [
        trial.final_measurement.metrics[0].value for trial in hp_job.trials
    ]
    best_trial = hp_job.trials[metrics.index(max(metrics))]
    best_roc_auc= float(best_trial.final_measurement.metrics[0].value)

    best_dropout = float(best_trial.parameters[0].value)
    best_embedding_dim = int(best_trial.parameters[1].value)
    best_hidden_dim = int(best_trial.parameters[2].value)
    best_max_features = int(best_trial.parameters[3].value)
    best_sequence_length = int(best_trial.parameters[4].value)


    return  best_dropout, best_embedding_dim, best_hidden_dim, best_max_features, best_roc_auc, best_sequence_length
