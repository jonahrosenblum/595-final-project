from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
from transformers import logging
import tensorflow as tf

import json

logging.set_verbosity_error()

# from transformers import AutoTokenizer, AutoModelForSequenceClassification

label_map = {False: 0, True: 1}

def read_jsonl(filename):
    jsonl = []
    with open(filename, 'r') as f:
        for line in f:
            jsonl.append(json.loads(line))
    return jsonl

def convert_data_to_examples(train, test, a, b, label): 
    train_input_examples = [InputExample(guid=None, text_a = x[a], text_b = x[b], label = label_map[x[label]]) for x in train]

    validation_input_examples = [InputExample(guid=None, text_a = x[a], text_b = x[b], label = label_map[x[label]]) for x in test]
  
    return train_input_examples, validation_input_examples
  
def examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            e.text_b,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            padding = 'max_length',
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict['input_ids'],
            input_dict['token_type_ids'], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    'input_ids': f.input_ids,
                    'attention_mask': f.attention_mask,
                    'token_type_ids': f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({'input_ids': tf.int32, 'attention_mask': tf.int32, 'token_type_ids': tf.int32}, tf.int64),
        (
            {
                'input_ids': tf.TensorShape([None]),
                'attention_mask': tf.TensorShape([None]),
                'token_type_ids': tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )



if __name__ == '__main__':
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # tokenizer = AutoTokenizer.from_pretrained("ishan/bert-base-uncased-mnli")
    # model = AutoModelForSequenceClassification.from_pretrained("ishan/bert-base-uncased-mnli")

    train = read_jsonl('small-train.json')
    test = read_jsonl('small-test.json')

    train_input_examples, validation_input_examples = convert_data_to_examples(train, test, 'question', 'passage', 'answer')

    train_data = examples_to_tf_dataset(train_input_examples, tokenizer)
    train_data = train_data.shuffle(100).batch(32).repeat(2)

    validation_data = examples_to_tf_dataset(validation_input_examples, tokenizer)
    validation_data = validation_data.batch(32)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

    model.fit(train_data, epochs=3, validation_data=validation_data)