# import json
# from transformers import BertTokenizer
# import numpy as np
# import tensorflow as tf

# def read_jsonl(filename):
#     jsonl = []
#     with open(filename, 'r') as f:
#         for line in f:
#             jsonl.append(json.loads(line))
#     return jsonl

# jsonl = read_jsonl("small.json")


# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# encodings = []
# for json in jsonl:
#     encoding = tokenizer(json['question'], json['passage'], padding="max_length", truncation=True)
#     encodings.append(encoding)

# bert = hub.load('https://tfhub.dev/google/experts/bert/wiki_books/qnli/2')


# # Feed the inputs to the model to get the pooled and sequence outputs
# bert_outputs = bert(encodings, training=False)
# pooled_output = bert_outputs['pooled_output']
# sequence_output = bert_outputs['sequence_output']

# print('\nSentences:')
# print(sentences)
# print('\nPooled output:')
# print(pooled_output)
# print('\nSequence output:')
# print(sequence_output)
