import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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


def input_ids_feature(input_dict):
    print("Creating text input.")
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']
    text = bert_tokens(titles, questions, answers)

    return text


def attention_mask_feature(input_dict):
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
        mask = [0.0 if (bert_token_size > i >= length) else 1.0 for i
                in range(0, bert_token_size)]
        masks.append(mask)

    return masks


def category_feature(input_dict):
    print("One hot encoding category.")
    category_list = input_dict['category']
    category = one_hot_encode(category_list)
    return category


def sentence_lengths_feature(input_dict):
    print('Calulating title, question, and answer lengths.')
    lengths = []

    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']
    for title, question, answer in zip(titles, questions, answers):
        title_length = len(title.split())
        question_length = len(question.split())
        answer_length = len(answer.split())

        lengths.append([title_length, question_length, answer_length])

    return lengths


def newlines_feature(input_dict):
    print('Counting the number of lines.')
    count = []
    questions = input_dict['question_body']
    answers = input_dict['answer']
    for question, answer in zip(questions, answers):
        count.append([question.count('\n'), answer.count('\n')])
    return count


def similarity_feature(input_dict):
    print('Calculating word similarity.')
    similarity = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        qandt = set(title.lower().split()) | set(question.lower().split())
        answer = set(answer.lower().split())
        similarity.append([len(qandt & answer)])

    return similarity


def hyperlinks_feature(input_dict):
    print('Counting the number of hyperlinks.')
    links = []
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for question, answer in zip(questions, answers):
        counts = [question.count('https://') + question.count('http://'),
                  answer.count('https://') + answer.count('http://')]
        links.append(counts)

    return links


def first_person_feature(input_dict):
    print('Counting first-person pronouns.')
    counts = []
    words = [' i ', ' me ', ' my ', ' mine ', ' we ', ' us ', ' ours ']

    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0]
        for word in words:
            count[0] += title.lower().count(word)
            count[0] += question.lower().count(word)
            count[1] += answer.lower().count(word)
        counts.append(count)

    return counts


def latex_feature(input_dict):
    print('Detecting the use of mathematical formulas.')
    maths = []
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for question, answer in zip(questions, answers):
        count = [question.count('$') // 2, answer.count('$') // 2]
        maths.append(count)

    return maths


def line_lengths_feature(input_dict):
    print('Measuring average line length.')
    lengths = []
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for question, answer in zip(questions, answers):
        qlines = question.splitlines()
        alines = question.splitlines()
        qmean = sum(len(line) for line in qlines) / len(qlines)
        amean = sum(len(line) for line in alines) / len(alines)

        lengths.append([qmean, amean])

    return lengths


def brackets_feature(input_dict):
    print('Counting bracket use.')
    code = []
    questions = input_dict['question_body']
    answers = input_dict['answer']
    brackets = ['{', '}', '[', ']', '(', ')']
    for question, answer in zip(questions, answers):
        count = [sum(question.count(symbol) for symbol in brackets) // 2,
                 sum(answer.count(symbol) for symbol in brackets) // 2]
        code.append(count)

    return code


def sentiment_feature(input_dict):
    print('Measuring sentiment.')
    sentiments = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    for title, question, answer in zip(titles, questions, answers):
        sentiment = [sid.polarity_scores(title)['compound'],
                     sid.polarity_scores(question)['compound'],
                     sid.polarity_scores(answer)['compounwd']
                     ]
        sentiments.append(sentiment)
    return sentiments


def spell_feature(input_dict):
    print('Counting words that relating to spelling.')
    mentions = []
    words = [' spell ', ' spelled ', ' spells ', ' spelling ', ' grammar ',
             ' spelt ', ' speller ']

    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]
        for word in words:
            count[0] += title.lower().count(word)
            count[1] += question.lower().count(word)
            count[2] += answer.lower().count(word)
        mentions.append(count)

    return mentions


def site_feature():
    pass


def hyperlink_to_self_feature():
    pass


def punction_count_feature():
    pass


def question_mark_feature():
    pass
