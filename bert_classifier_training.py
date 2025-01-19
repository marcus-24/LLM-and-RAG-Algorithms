from dotenv import load_dotenv
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

# Reference for script: https://bhavikjikadara.medium.com/transfer-learning-step-by-step-implementation-using-hugging-face-824de8aa8afd
# transformers are the NN architecture where there's a combination of attention and multilayer perception layers.

# %% Load pretrained model
# load pre-trained tokenizer from BERT Uncased (upper/lower case doesnt matter)
model_name = "google-bert/bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load Bert Model with added sequnce classification layer at the end
# num_labels = Number of labels to use in the last layer added to the model, typically for a classification task.
model = BertForSequenceClassification.from_pretrained(
    model_name, num_labels=2, device_map="cuda"
)  # labels in data below are 0 (neg) and 1 (pos)


# %% Load and process data
train_dataset = load_dataset(
    "stanfordnlp/imdb", split="train[:1%]"
)  # load IMDB movie reviews

test_dataset = load_dataset(
    "stanfordnlp/imdb", split="test[:1%]"
)  # load IMDB movie reviews

# set to pad and truncate at max length of model
tokenized_train_datasets = train_dataset.map(
    lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True),
    batched=True,
)  # output batches of data rather than single samples

tokenized_test_datasets = test_dataset.map(
    lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True),
    batched=True,
)  # output batches of data rather than single samples

# %% Train the model

# The training classes below are used as pytorch wrappers to train the model on multiple GPUs/TPUs

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # where model preds and checkpoints are saved
    evaluation_strategy="epoch",  # model eval is done at end of each epoch
    per_device_train_batch_size=16,  # batch size for train data per GPU/TPU etc.
    per_device_eval_batch_size=16,  # batch size for eval data per GPU/TPU etc.
    num_train_epochs=1,  # num of training epochs
    weight_decay=0.01,  # apply weight decay as regularization to prevent over training due to large weights
    max_steps=2,  # the total number of training steps to perform.
    run_name="Marcus Run",
)
# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_test_datasets,
)
# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save the model and tokenizer
model.save_pretrained("./fine-tuned-bert")
# tokenizer.save_pretrained("./fine-tuned-bert")
