import numpy as np

from transformers import BertTokenizer


def tokenize(strings, tokenizer):
    output = []
    for string in strings:
        output.append(tokenizer.tokenize(string))
    return output


def clip_or_pad(titles, questions, answers, max_length):
    output = []

    for idx, (title, question, answer) in enumerate(zip(titles, questions,
                                                        answers)):
        extra = (len(title)
                 + len(question)
                 + len(answer)
                 + len(['CLS]', '[SEP]', '[SEP]'])
                 - max_length
                 )
        title = ['[CLS]'] + title
        if extra > 0:
            clip1 = int(extra * len(question) / (len(question) + len(answer)))
            clip2 = extra - clip1
            if clip1 > 0:
                question = question[:-clip1]
            if clip2 > 0:
                answer = answer[:-clip2]
            answer = ['[SEP]'] + answer + ['[SEP]']
        else:
            answer = ['[SEP]'] + answer + ['[SEP]'] + \
                     ['[PAD]' for _ in range(0, -extra)]

        concat = title + question + answer
        output.append(concat)

    return output


def one_hot_encode(category_list):
    output = []
    oh_dict = {
        'SCIENCE': [1.0, 0.0, 0.0, 0.0, 0.0],
        'CULTURE': [0.0, 1.0, 0.0, 0.0, 0.0],
        'LIFE_ARTS': [0.0, 0.0, 1.0, 0.0, 0.0],
        'STACKOVERFLOW': [0.0, 0.0, 0.0, 1.0, 0.0],
        'TECHNOLOGY': [0.0, 0.0, 0.0, 0.0, 1.0]
    }
    for row in category_list:
        output.append(oh_dict[row])

    return output


def encode(strings, tokenizer):
    output = []
    for string in strings:
        output.append(tokenizer.convert_tokens_to_ids(string))
    return output


def bert_tokens(titles, questions, answers):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)
    bert_token_size = 512
    title_tks = tokenize(titles, tokenizer)
    quest_tks = tokenize(questions, tokenizer)
    ans_tks = tokenize(answers, tokenizer)
    concat = clip_or_pad(title_tks, quest_tks, ans_tks, bert_token_size)
    output = encode(concat, tokenizer)
    return output


def target_vector_label(target_dict):
    print("Creating label vector.")
    for idx0, (_, data) in enumerate(target_dict.items()):
        if idx0 == 0:
            samples = len(data)
            features = len(target_dict)
            label = [[0.0 for _ in range(features)] for _ in range(samples)]
        for idx1, row in enumerate(data):
            label[idx1][idx0] = float(row)

    return label


def text_feature(input_dict):
    print("Creating text input.")
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']
    text = bert_tokens(titles, questions, answers)

    return text


def mask_feature(input_dict):
    print("Creating attention mask.")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)
    bert_token_size = 512

    masks = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    title_tks = tokenize(titles, tokenizer)
    quest_tks = tokenize(questions, tokenizer)
    ans_tks = tokenize(answers, tokenizer)

    for title, question, answer in zip(title_tks, quest_tks, ans_tks):
        length = len(title) + len(question) + len(answer)
        mask = [1.0 if (bert_token_size > i >= length) else 0.0 for i
                in range(0, bert_token_size)]
        masks.append(mask)

    return masks


def category_feature(input_dict):
    print("One hot encoding category.")
    category_list = input_dict['category']
    category = one_hot_encode(category_list)
    return category
