# Detect-LLM-Vertex-CICD-Pipeline

## Overview

This repository contains code and resources for training and evaluating a deep learning model for text classification using TensorFlow and Keras. The model is specifically designed to detect whether student articles are written by humans or by a language model (LLM). The entire pipeline is orchestrated using Kubeflow Pipelines, with a continuous integration and deployment (CI/CD) setup using Google Cloud Build.

## Credits

### Kaggle Competition
- This project makes use of data from the [LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text) Kaggle competition.
  - The [DAIGT V2 Train Dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset) was employed within the machine learning code.

### Google Cloud Advanced Solutions Lab
- The organizational structure and implementation of this repository were inspired by the [Advanced Solutions Lab repository](https://github.com/GoogleCloudPlatform/asl-ml-immersion.git) from Google Cloud.
  - Code examples from the [Kubeflow Pipelines section](https://github.com/GoogleCloudPlatform/asl-ml-immersion/tree/master/notebooks/kubeflow_pipelines) of the Advanced Solutions Lab repository were referenced.

The contributions of the Kaggle community and Google Cloud Advanced Solutions Lab are acknowledged and appreciated for providing valuable resources and insights that significantly contributed to the development of this project.


## Repository Structure

The repository is organized into the following directories:

- **manuel-deployment:** Contains a Jupyter notebook (`text-classification-from-scratch.ipynb`) demonstrating manual deployment steps and a directory (`training_app_trees`) with a Dockerfile and training script for manual deployment.

- **vertex-kfp-pipeline:** Includes a Jupyter notebook (`detect-llm-kfp-pipeline.ipynb`) explaining the Kubeflow Pipeline setup. The `detect_llm_vertex_trainer` directory contains the Dockerfile and training script for the Kubeflow pipeline. The `pipeline_vertex` directory contains Python scripts defining the pipeline components.

- **vertex-kfp-pipeline-cicd:** Encompasses the CI/CD setup. It includes a Jupyter notebook (`detect_llm_kfp_cicd_vertex.ipynb`) explaining the CI/CD process. The `cloudbuild_vertex.yaml` file defines the Cloud Build configuration. The `detect_llm_vertex_trainer` directory contains the Dockerfile and training script for the CI/CD pipeline. The `kfp-cli-vertex` directory includes a Dockerfile and a script (`run_pipeline.py`) for running the pipeline using Kubeflow CLI.

```
.
├── LICENSE
├── manuel-deployment
│   ├── config.yaml
│   ├── requirements.txt
│   ├── text-classification-from-scratch.ipynb
│   └── training_app_trees
│       ├── Dockerfile
│       └── train.py
├── README.md
├── vertex-kfp-pipeline
│   ├── detect-llm-kfp-pipeline.ipynb
│   ├── detect_llm_vertex_trainer
│   │   ├── Dockerfile
│   │   └── train.py
│   ├── pipeline_vertex
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── training_lightweight_component.py
│   │   └── tuning_lightweight_component.py
│   └── requirements.txt
└── vertex-kfp-pipeline-cicd
    ├── cloudbuild_vertex.yaml
    ├── detect_llm_kfp_cicd_vertex.ipynb
    ├── detect_llm_vertex_trainer
    │   ├── Dockerfile
    │   └── train.py
    ├── kfp-cli-vertex
    │   ├── Dockerfile
    │   └── run_pipeline.py
    ├── pipeline_vertex
    │   ├── data_component.py
    │   ├── __init__.py
    │   ├── pipeline.py
    │   ├── preprocess_component.py
    │   ├── training_lightweight_component.py
    │   └── tuning_lightweight_component.py
    └── requirements.txt
```

## CI/CD Steps (cloudbuild_vertex.yaml)

### Step 0: Build the trainer image
Builds the Docker image for the trainer using the `gcr.io/cloud-builders/docker` image.

### Step 1: Push the trainer image
Pushes the built trainer image to Google Container Registry.

### Step 2: Compile the pipeline
Uses the `gcr.io/$PROJECT_ID/kfp-cli-vertex` image to compile the Kubeflow pipeline defined in `pipeline.py`. The compiled pipeline is saved as `detect_llm_kfp_pipeline.json`.

### Step 3: Run the pipeline
Executes the compiled Kubeflow pipeline using the `gcr.io/$PROJECT_ID/kfp-cli-vertex` image. The pipeline is run with specified parameters such as project ID, template path, display name, and region.

### Step 4: Push the images to Container Registry
Pushes the trainer image (`gcr.io/$PROJECT_ID/detect_llm_trainer_image_vertex:latest`) to Google Container Registry.

### Timeout
This is required since the pipeline run may exceed the default timeout, set to 10800 seconds.

## Notes
- The data used for training and validation is stored in Google Cloud Storage.
- The serving container image URI (`us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-13:latest`) and training container image URI (`gcr.io/$PROJECT_ID/detect_llm_trainer_image_vertex:latest`) are specified.
- The script utilizes the Fire library for command-line interface functionality.

## Deployment Walkthroughs

### Manual Deployment
The `manuel-deployment` directory provides a Jupyter notebook (`text-classification-from-scratch.ipynb`) guiding users through the manual deployment process.

### Kubeflow Pipeline Version of Manual Deployment
The `vertex-kfp-pipeline` directory includes a Jupyter notebook (`detect-llm-kfp-pipeline.ipynb`) explaining how to deploy the model using Kubeflow Pipelines.

### CI/CD Version of Kubeflow Pipelines
The `vertex-kfp-pipeline-cicd` directory contains a Jupyter notebook (`detect_llm_kfp_cicd_vertex.ipynb`) explaining the CI/CD setup for Kubeflow Pipelines. The CI/CD process triggers on a new release of the repository using Google Cloud Build.

## Monitoring

### Wandb
Wandb [project](https://wandb.ai/mustafakeser/detect-llm-gcp/workspace?workspace=user-).

## Releases

### Initial Pipeline Release

#### Release Version 0.0.0
The initial release introduces a straightforward pipeline with the following steps:
- Hyperparameter tuning
- Deployment decision
- Training and deployment 

using Cloud Build with trigger when release created for repository.

This release establishes the foundation for the project, focusing on key stages of hyperparameter tuning, deployment decision and model training and deployment.


## Future Developments

This pipeline is designed to evolve and adapt to future requirements. Here are some potential areas for improvement and future developments:

- **Enhanced Data Processing:** Explore more advanced data processing techniques, including automated feature engineering and dynamic data labeling strategies.

- **Advanced Model Explainability:** Integrate more advanced model explainability techniques to provide clearer insights into the model's decision-making process.

- **Continuous Monitoring:** Implement a robust continuous monitoring system to proactively identify model degradation and ensure ongoing performance excellence.

- **Scaling to Larger Datasets:** Optimize the pipeline for scalability, enabling it to handle larger datasets efficiently.

- **Integration with Additional Data Sources:** Explore integration with additional data sources to enrich the model with diverse information.

- **Exploration of New Architectures:** Investigate new model architectures or advanced deep learning techniques to enhance predictive capabilities.

- **Automated Model Retraining:** Implement automated model retraining mechanisms based on specific triggers or schedules to keep the model up-to-date.

- **Model Deployment Strategies:** Explore alternative deployment strategies and policies based on various performance metrics beyond ROC AUC, considering precision, recall, or F1 score.

These potential enhancements aim to make the pipeline more versatile, robust, and adaptable to emerging challenges and opportunities in the domain of text classification and model deployment.



