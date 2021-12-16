from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
from transformers import logging
import tensorflow as tf
import tensorflow.keras.backend as K

import xmltodict, json
import numpy as np

import argparse

logging.set_verbosity_error()

mnli_map = {'contradiction': False, 'entailment': True}
semeval_map = {'No': False, 'Yes': True, 'Unsure': 2}
label_map = {False: 0, True: 1}

golden_label_questions = {"Q2619", "Q2623", "Q2647", "Q2685", "Q2688", "Q2690", "Q2700", "Q2708", "Q2716", "Q2717", "Q2719", "Q2725", "Q2726", "Q2733", "Q2741", "Q2744", "Q2747", "Q2750", "Q2754", "Q2758", "Q2765", "Q2767", "Q2797", "Q2799", "Q2819", "Q2831", "Q2833", "Q2842", "Q2843", "Q2851", "Q2882", "Q2891", "Q2898", "Q2900"}

parser = argparse.ArgumentParser()
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
    semeval_list = []

    with open(filename) as f:
        doc = xmltodict.parse(f.read())
        for q in doc['root']['Question']:
            if q['@QID'] in golden_label_questions:
                print(q)
                # br = False
                # for comment in q['Comment']:
                #     # if there's only one comment, we don't have a list
                #     if type(comment) == str:
                #         comment = q['Comment']
                #         bf = True

                #     if comment['@CGOLD'] == 'Good' and comment['@CGOLD_YN'] in ['Yes', 'No']:
                #         semeval_list.append({"question": q['QBody'], 
                #                              "passage": comment['CBody'], #comment['CBody'] if comment['CSubject'] in comment['CBody'] else comment['CSubject'] + ' ' + comment['CBody'], 
                #                              "answer": semeval_map[comment['@CGOLD_YN']]
                #                             })
                #     if br:
                #         break

    return semeval_list


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

    ret = [{'input_ids': feature.input_ids, 'attention_mask': feature.attention_mask, 'token_type_ids': feature.token_type_ids} for feature in features]
    return ret


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

    if not args['modelname']:
        print('No modelname given')
        exit(1)

    eval_model = TFBertForSequenceClassification.load(f"saved_models/{args['modelname']}")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input_dict = tokenizer.encode_plus(
            'Is the sky blue?',
            'Yes, the sky is in fact blue.',
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=True,
            return_attention_mask=True,
            padding = 'max_length',
            truncation=True
        )

    input_dict.batch(32)
    tf_outputs = eval_model(input_dict)
    print(tf_outputs)
    # tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)

    # a = [{'first': 'Is the sky blue?', 'second': 'Yes, the sky is in fact blue.', 'label': True}]

    # train_input_examples, _ = convert_data_to_examples(a, a, 'first', 'second', 'label')

    # eval_data = examples_to_tf_dataset(train_input_examples, tokenizer)

    # for d in eval_data:
    #     print(eval_model(input_ids=d['input_ids'], attention_mask=d['attention_mask'], token_type_ids=d['token_type_ids']))

    # res = eval_model(eval_data)
    # print(res)

    # model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    #               metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy'),
    #                        f1_metric
    #               ])
    
    # for train, test in zip(train_l, test_l):
    #     train_model(model, tokenizer, train, test)
    
    # if args['modelname']:
    #     model.save(f"saved_models/{args['modelname']}")
