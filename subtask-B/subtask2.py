from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
from transformers import logging
import tensorflow as tf
import tensorflow.keras.backend as K

import xmltodict, json

import random, argparse

logging.set_verbosity_error()

golden_label_questions = {"Q2619", "Q2623", "Q2647", "Q2685", "Q2688", "Q2690", "Q2700", "Q2708", "Q2716", "Q2717", "Q2719", "Q2725", "Q2726", "Q2733", "Q2741", "Q2744", "Q2747", "Q2750", "Q2754", "Q2758", "Q2765", "Q2767", "Q2797", "Q2799", "Q2819", "Q2831", "Q2833", "Q2842", "Q2843", "Q2851", "Q2882", "Q2891", "Q2898", "Q2900"}


mnli_map = {'contradiction': False, 'entailment': True}
semeval_map = {'No': False, 'Yes': True, 'Unsure': 2}
label_map = {False: 0, True: 1}

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--multinli", help="Should MultiNLI be used for training", action="store_true")
parser.add_argument("-b", "--boolq", help="Should BoolQ be used for training", action="store_true")
parser.add_argument("-n", "--modelname", type=str, help="What name to save the model as.")

def read_xml(filename):
    semeval_list = []
    with open(filename) as f:
        doc = xmltodict.parse(f.read())
        for q in doc['root']['Question']:
            if q['@QTYPE'] == "YES_NO":
                br = False
                for comment in q['Comment']:
                    # if there's only one comment, we don't have a list
                    if type(comment) == str:
                        comment = q['Comment']
                        bf = True

                    if comment['@CGOLD'] == 'Good' and comment['@CGOLD_YN'] in ['Yes', 'No']:
                        semeval_list.append({"question": q['QBody'], 
                                             "passage": comment['CBody'], #comment['CBody'] if comment['CSubject'] in comment['CBody'] else comment['CSubject'] + ' ' + comment['CBody'], 
                                             "answer": semeval_map[comment['@CGOLD_YN']]
                                            })
                    if br:
                        break
    return semeval_list

def read_test(filename):
    test_list = []

    with open(filename) as f:
        doc = xmltodict.parse(f.read())
        for q in doc['root']['Question']:
            if q['@QID'] in golden_label_questions:
                comments = []
                br = False
                for comment in q['Comment']:
                    if type(comment) == str:
                        comment = q['Comment']
                        bf = True
                    if comment['@CGOLD'] == 'Good':
                        comments.append(comment['CBody'])
                    if br:
                        break
                test_list.append({'question': q['QBody'], 'comments': comments, 'qanswer': q['@QGOLD_YN'], 'QID': q['@QID']})

    return test_list

def read_jsonl(filename):
    jsonl = []
    with open(filename, 'r') as f:
        for line in f:
            jsonl.append(json.loads(line))
    return jsonl

def read_jsonl_nli(filename):
    jsonl = []
    with open(filename, 'r') as f:
        for line in f:
            line = json.loads(line)
            if line['gold_label'] in ['contradiction', 'entailment']:
                jsonl.append({'question': line['sentence1'], 'passage': line['sentence2'], 'answer': mnli_map[line['gold_label']]})

    return jsonl

def convert_data_to_examples(train, test, a, b, label):
    train_input_examples = [InputExample(guid=None, text_a = x[a], text_b = x[b], label = label_map[x[label]]) for x in train]

    validation_input_examples = [InputExample(guid=None, text_a = x[a], text_b = x[b], label = label_map[x[label]]) for x in test]
  
    return train_input_examples, validation_input_examples
  
def examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = []

    for e in examples:
        input_dict = tokenizer.encode_plus(
            e.text_a,
            e.text_b,
            add_special_tokens=True,
            max_length=max_length,
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
        for feature in features:
            yield (
                {
                    'input_ids': feature.input_ids,
                    'attention_mask': feature.attention_mask,
                    'token_type_ids': feature.token_type_ids,
                },
                feature.label,
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

def train_model(model, tokenizer, train, test):
    train_input_examples, validation_input_examples = convert_data_to_examples(train, test, 'question', 'passage', 'answer')

    train_data = examples_to_tf_dataset(train_input_examples, tokenizer)
    train_data = train_data.shuffle(100).batch(32)

    validation_data = examples_to_tf_dataset(validation_input_examples, tokenizer)
    validation_data = validation_data.batch(32)

    model.fit(train_data, epochs=3, validation_data=validation_data)


def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

if __name__ == '__main__':
    args = vars(parser.parse_args())

    train_l = []
    test_l = []
    name = []
    if args['multinli']:
        train_l.append(random.sample(read_jsonl_nli('../multinli_1.0/multinli_1.0_train.jsonl'), 40000))
        test_l.append(read_jsonl_nli('../multinli_1.0/multinli_1.0_dev_mismatched.jsonl'))
        name.append('MULTINLI')

    if args['boolq']:
        train_l.append(read_jsonl('train.json'))
        test_l.append(read_jsonl('dev.json'))
        name.append('BOOLQ')

    train_l.append(read_xml('../English-data/datasets/CQA-QL-train.xml'))
    test_l.append(read_xml('../English-data/datasets/CQA-QL-devel.xml'))
    name.append('SEMEVAL')

    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy'),
                           f1_metric
                  ])

    
    i = 0
    for train, test in zip(train_l, test_l):
        print(f'TRAINING: {name[i]}')
        train_model(model, tokenizer, train, test)
        i += 1


    final_eval = read_test('../English-data/datasets/CQA-QL-devel.xml')
    
    for eval in final_eval:
        final_prediction = "Unsure"
        prediction_counts = {"No": 0, "Yes": 0}
        total_comments = 0
        for comment in eval['comments']:
            total_comments += 1
            input_dict = tokenizer.encode_plus(
                    eval['question'],
                    comment,
                    add_special_tokens=True,
                    max_length=128,
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    padding = 'max_length',
                    truncation=True,
                    return_tensors='tf'
                )

            tf_outputs = model(input_dict)
            tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
            # preds = tf_predictions[0].numpy()
            # print(preds)
            # prediction_counts['No'] += preds[0]
            # prediction_counts['Yes'] += preds[1]
            # exit(0)
            labels = ['No','Yes']
            label = tf.argmax(tf_predictions, axis=1)
            label = label.numpy()

            prediction_counts[labels[label[0]]] += 1
        
        if (prediction_counts['Yes'] / total_comments) >= 2/3:
            final_prediction = 'Yes'
        if (prediction_counts['No'] / total_comments) >= 2/3:
            final_prediction = 'No'
        print(eval['QID'] + '	' + final_prediction)
        # print(prediction_counts)
