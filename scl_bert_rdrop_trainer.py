import json
import os
import sys
import logging
import datasets
import losses

import torch.nn as nn

import pandas as pd
import numpy as np

from transformers import BertTokenizerFast, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers import BertPreTrainedModel, BertModel,LongformerTokenizerFast,LongformerModel
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.model_selection import train_test_split

train=pd.read_csv("./trainingdata/translated_traindata.csv")
val=pd.read_csv("./trainingdata/devdata.csv")
test=pd.read_csv("./trainingdata/testdata.csv")


class BertScratch(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.alpha=0.2

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            scl_fct = losses.SupConLoss()
            scl_loss = scl_fct(pooled_output, labels)

            loss = ce_loss + self.alpha * scl_loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    #train, val = train_test_split(train, test_size=.2)

    train_dict = {'label': train['Label'], 'text1': train['Statement'],
                  'text2': train['Premise']}
    val_dict = {'label': val['Label'], 'text1': val['Statement'],
                  'text2': val['Premise']}
    test_dict = {'text1': test['Statement'],
                  'text2': test['Premise']}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    #tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('dmis-lab/biobert-v1.1')
    #tokenizer = BertTokenizerFast.from_pretrained('Charangan/MedBERT')
    #tokenizer = BertTokenizerFast.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    #tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')


    def preprocess_function(examples):
        return tokenizer(examples['text1'],examples['text2'], truncation=True)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #model = BertScratch.from_pretrained('bert-base-uncased')
    model = BertScratch.from_pretrained('dmis-lab/biobert-v1.1')
    #model = BertScratch.from_pretrained('Charangan/MedBERT')
    #model = BertScratch.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    #model = BertScratch.from_pretrained('bert-large-uncased')
    metric = datasets.load_metric("accuracy")



    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments(
        output_dir='./checkpoint',  # output directory
        num_train_epochs=9,  # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,
        save_strategy="no",
        evaluation_strategy="epoch",
        learning_rate=1.5e-5
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    Results={}
    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(len(test),len(test_pred))
    for i in range(len(test_pred)):
        if(test_pred[i]==0):
            Prediction="Contradiciton"
        else:
            Prediction="Entailment"
        Results[str(test['uuid'][i])]={"Prediction":Prediction}
    print(Results)
    with open("./trainingdata/biobertresults.json", 'w') as jsonFile:
        jsonFile.write(json.dumps(Results, indent=4))
