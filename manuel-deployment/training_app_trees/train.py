import os
import subprocess
import sys
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
from google.cloud import storage
import fire
import hypertune
import string
import re
from sklearn.metrics import roc_auc_score
import time
import keras
from keras import layers

def train_evaluate(training_file_path,validation_file_path,test_file_path,job_dir, hidden_dim, dropout, embedding_dim ,sequence_length ,max_features, hptune): 
    
    train0 = pd.read_csv(training_file_path)
    val0 = pd.read_csv(validation_file_path)
    test0 = pd.read_csv(test_file_path)
    dataset_tr = tf.data.Dataset.from_tensor_slices((train0.text.values,train0.label.values.astype("float32") ))
    dataset_tr = dataset_tr.shuffle(buffer_size=len(train0)).batch(batch_size=2)
    dataset_val = tf.data.Dataset.from_tensor_slices((val0.text.values,val0.label.values.astype("float32")  ))
    dataset_val = dataset_val.shuffle(buffer_size=len(val0)).batch(batch_size=2)
    dataset_test = tf.data.Dataset.from_tensor_slices((test0.text.values ))
    dataset_test = dataset_test.batch(batch_size=2)
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
        return tf.strings.regex_replace(
            stripped_html, f"[{re.escape(string.punctuation)}]", ""
        )
    max_features = int(max_features)
    embedding_dim = int(embedding_dim)
    sequence_length = int(sequence_length)
    vectorize_layer = keras.layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    text_ds = dataset_tr.map(lambda x, y: x).concatenate(dataset_val.map(lambda x, y: x)).concatenate(dataset_test)
    vectorize_layer.adapt(text_ds)
    
    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label
    def vectorize_text_test(text):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text)
    train_ds = dataset_tr.map(vectorize_text)
    train_ds = train_ds.cache().prefetch(buffer_size=10)
    
    val_ds = dataset_val.map(vectorize_text)
    val_ds = val_ds.cache().prefetch(buffer_size=10)
    
    test_ds = dataset_test.map(vectorize_text_test)
    test_ds = test_ds.cache().prefetch(buffer_size=10)
    
    inputs = keras.Input(shape=(None,), dtype="int64")
    x = layers.Embedding(max_features, embedding_dim)(inputs)
    x = layers.Dropout(float(dropout))(x)
    x = layers.Conv1D(int(hidden_dim), 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.Conv1D(int(hidden_dim), 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(int(hidden_dim), activation="relu")(x)
    x = layers.Dropout(float(dropout))(x)
    
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    model = keras.Model(inputs, predictions)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy",tf.keras.metrics.AUC()])
    
    num_cpus = os.cpu_count()
    
    epochs = 1
    tf.random.set_seed(1)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs,workers=num_cpus)
    
    
    preds = model.predict(test_ds)

    if hptune:
            roc_auc = roc_auc_score(test0.label,preds)
            print('Model roc_auc: {}'.format(roc_auc))
    
            hpt = hypertune.HyperTune()
            hpt.report_hyperparameter_tuning_metric(
              hyperparameter_metric_tag='roc_auc',
              metric_value=roc_auc
            )
    
    # Save the model
    if not hptune:
        model_filename = "end_to_end_detect_llm"
        inputs = keras.Input(shape=(1,), dtype="string")
        indices = vectorize_layer(inputs)
        outputs = model(indices)
        end_to_end_model = keras.Model(inputs, outputs)
        end_to_end_model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy",tf.keras.metrics.AUC()]
        )
        # end_to_end_model.save(model_filename,save_format="keras")
        tf.saved_model.save(
        obj=end_to_end_model, export_dir=model_filename)
    
        gcs_model_path = "{}/{}".format(job_dir, model_filename)
        subprocess.check_call(['gsutil', 'cp', '-r', model_filename, gcs_model_path], stderr=sys.stdout)
        print("Saved model in: {}".format(gcs_model_path)) 
if __name__ == "__main__":
        fire.Fire(train_evaluate)

