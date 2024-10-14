import os
import math
import evaluate
import torch
import rouge
import json
import numpy as np
from statistics import mean
from codecarbon import EmissionsTracker
from nltk.translate.bleu_score import corpus_bleu

def get_carburacy(score, emission_train, emission_test, alpha=10, beta_train=1, beta_test=100):
    carburacy_train = None
    if emission_train is not None:
        carburacy_train = math.exp(math.log(score/100, alpha)) / (1 + emission_train * beta_train)
        carburacy_train = round(100 * carburacy_train, 2)
    carburacy_test = None
    if emission_test is not None:
        carburacy_test = math.exp(math.log(score/100, alpha)) / (1 + emission_test * beta_test)
        carburacy_test = round(100 * carburacy_test, 2)
    carburacy = None
    if carburacy_train is not None and carburacy_test is not None:
        carburacy = (2 * carburacy_train * carburacy_test) / (carburacy_train + carburacy_test)
        carburacy = round(100 * carburacy, 2)
    return carburacy_train, carburacy_test, carburacy


def predict(trainer, predict_dataset, max_predict_samples, training_args, tokenizer, train_emissions, split):
    test_tracker = EmissionsTracker(measure_power_secs=100000, save_to_file=False)
    test_tracker.start()
    predict_results = trainer.predict(predict_dataset, metric_key_prefix=split)
    test_emissions = test_tracker.stop()

    metrics = predict_results.metrics

    metrics[f"{split}_samples"] = min(max_predict_samples, len(predict_dataset))
    metrics[f"{split}_emissions"] = test_emissions

    trainer.save_metrics(split, metrics)

    if trainer.is_world_process_zero():
        if training_args.predict_with_generate:
            predictions = predict_results.predictions
            predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
            predictions = tokenizer.batch_decode(
                predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            list_output_dict = []
            for i, pred in enumerate(predictions):
                output_dict = {"prediction": pred}
                list_output_dict.append(output_dict)

            if training_args.new_dir is not None:
                if not os.path.exists(training_args.new_dir):
                    os.makedirs(training_args.new_dir)
                output_prediction_file = os.path.join(training_args.new_dir, f"generated_{split}_set.json")
            else:
                output_prediction_file = os.path.join(training_args.output_dir, f"generated_{split}_set.json")
            

            with open(output_prediction_file, 'w') as json_file:
                json.dump(list_output_dict, json_file, indent=4) 


def preprocess_function(examples, data_args, tokenizer):
    
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    filtered_examples = [
        (input_val, target_val) 
        for input_val, target_val in zip(examples[data_args.input_column], examples[data_args.target_column]) 
        if input_val is not None and target_val is not None
    ]

    # Now, split the filtered results back into inputs and targets
    inputs, targets = zip(*filtered_examples)
    
    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

