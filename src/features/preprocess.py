import math


def tokenize(strings, tokenizer):
    output = []
    for string in strings:
        output.append(tokenizer.tokenize(string))
    return output


def clip_or_pad(titles, questions, answers, max_length):
    output = []
    mask = []
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

        mask.append([1 if token != ['PAD'] else 0 for token in concat])
    return output, mask


def encode(strings, tokenizer):
    output = []
    for string in strings:
        output.append(tokenizer.convert_tokens_to_ids(string))
    return output


def preprocess(titles, questions, answers, max_length, tokenizer):
    title_tks = tokenize(titles, tokenizer)
    quest_tks = tokenize(questions, tokenizer)
    ans_tks = tokenize(answers, tokenizer)
    concat, mask = clip_or_pad(title_tks, quest_tks, ans_tks, max_length)
    output = encode(concat, tokenizer)
    return output, mask
