import csv
import pathlib
from collections import OrderedDict

import nltk
import numpy as np
import pandas as pd
import torch
import transformers
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from torch.utils import data
from transformers import RobertaTokenizer

from src.model import model

input_fields = [
    'qa_id',
    'question_title',
    'question_body',
    'question_user_name',
    'question_user_page',
    'answer',
    'answer_user_name',
    'answer_user_page',
    'url',
    'category',
    'host'
]

feature_fields = [
    'input_ids',
    'attention_mask',
    'category',
    'host',
    'sentence_lengths',
    'newlines',
    'similarity',
    'hyperlinks',
    'first_person',
    'latex',
    'brackets',
    'sentiment',
    'spell',
    'punctuation',
    'question_mark',
    'exclamation',
    'yes_no',
    'numeric_answer',
    'short_keyword',
    'consequence',
    'how',
    'choice',
    'comparison',
    'comma',
    'period',
    'instruction',
    'self_reference',
    'parts_of_speech',
    'language',
    'sonic',
    'reference',
    'word_part',
]

target_fields = [
    'question_asker_intent_understanding',
    'question_body_critical',
    'question_conversational',
    'question_expect_short_answer',
    'question_fact_seeking',
    'question_has_commonly_accepted_answer',
    'question_interestingness_others',
    'question_interestingness_self',
    'question_multi_intent',
    'question_not_really_a_question',
    'question_opinion_seeking',
    'question_type_choice',
    'question_type_compare',
    'question_type_consequence',
    'question_type_definition',
    'question_type_entity',
    'question_type_instructions',
    'question_type_procedure',
    'question_type_reason_explanation',
    'question_type_spelling',
    'question_well_written',
    'answer_helpful',
    'answer_level_of_information',
    'answer_plausible',
    'answer_relevance',
    'answer_satisfaction',
    'answer_type_instructions',
    'answer_type_procedure',
    'answer_type_reason_explanation',
    'answer_well_written'
]

id_field = 'qa_id'


def tokenize(strings, tokenizer):
    output = []
    for string in strings:
        output.append(tokenizer.tokenize(string))
    return output


def clip_or_pad(titles, questions, answers, max_length):
    texts = []
    masks = []
    token_ids = []

    for idx, (title, question, answer) in enumerate(zip(titles, questions,
                                                        answers)):
        extra = (len(title)
                 + len(question)
                 + len(answer)
                 + len(['CLS]', '[SEP]', '[SEP]'])
                 - max_length
                 )

        if extra > 0:
            clip1 = int(extra * len(question) / (len(question) + len(answer)))
            clip2 = extra - clip1
            if clip1 > 0:
                question = question[:-clip1]
            if clip2 > 0:
                answer = answer[:-clip2]
            pad = []
        else:
            pad = ['[PAD]' for _ in range(0, -extra)]

        title = ['[CLS]'] + title
        answer = ['[SEP]'] + answer + ['[SEP]']

        title_mask = [1] * len(title)
        title_token_ids = [0] * len(title)

        question_mask = [1] * len(question)
        question_token_ids = [0] * len(question)

        answer_mask = [1] * len(answer)
        answer_token_ids = [0] + [1] * (len(answer) - 1)

        pad_mask = [0] * len(pad)
        pad_token_ids = [1] * len(pad)

        text = title + question + answer + pad
        text_mask = title_mask + question_mask + answer_mask + pad_mask
        text_token_ids = title_token_ids + question_token_ids + \
                         answer_token_ids + pad_token_ids

        texts.append(text)
        masks.append(text_mask)
        token_ids.append(text_token_ids)

    return texts, masks, token_ids


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
    bert_token_size = 512

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base',
                                                 sep_token='[SEP]',
                                                 pad_token='[PAD]',
                                                 cls_token='[CLS]')

    title_tks = tokenize(titles, tokenizer)
    quest_tks = tokenize(questions, tokenizer)
    ans_tks = tokenize(answers, tokenizer)
    concat, _, _ = clip_or_pad(title_tks, quest_tks, ans_tks, bert_token_size)
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
    bert_tkn_size = 512

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base',
                                                 sep_token='[SEP]',
                                                 pad_token='[PAD]',
                                                 cls_token='[CLS]')

    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    title_tks = tokenize(titles, tokenizer)
    quest_tks = tokenize(questions, tokenizer)
    ans_tks = tokenize(answers, tokenizer)

    _, masks, _ = clip_or_pad(title_tks, quest_tks, ans_tks, bert_tkn_size)

    return masks


def token_type_ids_feature(input_dict):
    print("Creating token_ids mask.")
    bert_tkn_size = 512

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base',
                                                 sep_token='[SEP]',
                                                 pad_token='[PAD]',
                                                 cls_token='[CLS]')

    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    title_tks = tokenize(titles, tokenizer)
    quest_tks = tokenize(questions, tokenizer)
    ans_tks = tokenize(answers, tokenizer)

    _, _, token_ids = clip_or_pad(title_tks, quest_tks, ans_tks, bert_tkn_size)

    return token_ids


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
                     sid.polarity_scores(answer)['compound']
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


def host_feature(input_dict):
    sites = {'blender.stackexchange.com': 0, 'ell.stackexchange.com': 1,
             'gamedev.stackexchange.com': 2,
             'graphicdesign.stackexchange.com': 3,
             'mechanics.stackexchange.com': 4,
             'softwarerecs.stackexchange.com': 5,
             'meta.math.stackexchange.com': 6, 'gaming.stackexchange.com': 7,
             'serverfault.com': 8, 'music.stackexchange.com': 9,
             'biology.stackexchange.com': 10, 'bicycles.stackexchange.com': 11,
             'mathoverflow.net': 12, 'askubuntu.com': 13,
             'ux.stackexchange.com': 14,
             'cs.stackexchange.com': 15, 'meta.askubuntu.com': 16,
             'rpg.stackexchange.com': 17, 'raspberrypi.stackexchange.com': 18,
             'movies.stackexchange.com': 19,
             'meta.christianity.stackexchange.com': 20,
             'math.stackexchange.com': 21, 'diy.stackexchange.com': 22,
             'programmers.stackexchange.com': 23,
             'webapps.stackexchange.com': 24,
             'money.stackexchange.com': 25,
             'expressionengine.stackexchange.com': 26,
             'electronics.stackexchange.com': 27,
             'judaism.stackexchange.com': 28,
             'codereview.stackexchange.com': 29,
             'boardgames.stackexchange.com': 30,
             'sharepoint.stackexchange.com': 31,
             'crypto.stackexchange.com': 32,
             'gis.stackexchange.com': 33, 'physics.stackexchange.com': 34,
             'academia.stackexchange.com': 35, 'unix.stackexchange.com': 36,
             'superuser.com': 37, 'mathematica.stackexchange.com': 38,
             'travel.stackexchange.com': 39, 'dsp.stackexchange.com': 40,
             'cooking.stackexchange.com': 41,
             'salesforce.stackexchange.com': 42,
             'android.stackexchange.com': 43, 'stats.stackexchange.com': 44,
             'anime.stackexchange.com': 45, 'scifi.stackexchange.com': 46,
             'chemistry.stackexchange.com': 47,
             'webmasters.stackexchange.com': 48,
             'meta.codereview.stackexchange.com': 49,
             'drupal.stackexchange.com': 50,
             'security.stackexchange.com': 51, 'dba.stackexchange.com': 52,
             'magento.stackexchange.com': 53,
             'wordpress.stackexchange.com': 54,
             'stackoverflow.com': 55, 'tex.stackexchange.com': 56,
             'robotics.stackexchange.com': 57, 'photo.stackexchange.com': 58,
             'christianity.stackexchange.com': 59,
             'english.stackexchange.com': 60,
             'apple.stackexchange.com': 61, 'meta.stackexchange.com': 62}

    websites = []
    hosts = input_dict['host']
    for host in hosts:
        oh = [0.0] * len(sites)
        oh[sites[host]] = 1.0
        websites.append(oh)

    return websites


def self_reference_feature(input_dict):
    counter = []
    questions = input_dict['question_body']
    answers = input_dict['answer']
    hosts = input_dict['host']
    for host, question, answer in zip(hosts, questions, answers):
        count = [0, 0]
        count[0] += question.lower().count(host)
        count[1] += answer.lower().count(host)
        counter.append(count)
    return counter


def punctuation_feature(input_dict):
    characters = ['"', ':', ';', '?', '!', '-']
    punctuation = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]
        for char in characters:
            count[0] += title.count(char)
            count[1] += question.count(char)
            count[2] += answer.count(char)
        punctuation.append(count)

    return punctuation


def question_mark_feature(input_dict):
    counter = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]

        count[0] += title.count('?')
        count[1] += question.count('?')
        count[2] += answer.count('?')
        counter.append(count)

    return counter


def exclamation_feature(input_dict):
    exclaim = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]

        count[0] += title.count('!')
        count[1] += question.count('!')
        count[2] += answer.count('!')
        exclaim.append(count)

    return exclaim


def yes_no_feature(input_dict):
    words = ['does ', 'can ', 'has ', 'have ', 'had ', 'is this ',
             'is that ']
    counter = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]
        for word in words:
            count[0] += title.lower().count(word)
            count[1] += question.lower().count(word)
            count[2] += answer.lower().count(word)
        counter.append(count)

    return counter


def numeric_answer_feature(input_dict):
    words = ['how many', 'how much', 'when ', 'the number ']
    counter = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]
        for word in words:
            count[0] += title.lower().count(word)
            count[1] += question.lower().count(word)
            count[2] += answer.lower().count(word)
        counter.append(count)

    return counter


def short_keyword_feature(input_dict):
    words = ['where ', 'who ']
    counter = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]
        for word in words:
            count[0] += title.lower().count(word)
            count[1] += question.lower().count(word)
            count[2] += answer.lower().count(word)
        counter.append(count)

    return counter


def consequence_feature(input_dict):
    words = ['what happens ', 'what will happen ', 'what has happened ',
             'consequence ', 'result ', 'what happened ']
    counter = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]
        for word in words:
            count[0] += title.lower().count(word)
            count[1] += question.lower().count(word)
            count[2] += answer.lower().count(word)
        counter.append(count)

    return counter


def how_feature(input_dict):
    words = ['how']
    counter = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]
        for word in words:
            count[0] += title.lower().count(word)
            count[1] += question.lower().count(word)
            count[2] += answer.lower().count(word)
        counter.append(count)

    return counter


def thanks_feature(input_dict):
    words = ['thanks', 'thank you', 'i appreciate']
    counter = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]
        for word in words:
            count[0] += title.lower().count(word)
            count[1] += question.lower().count(word)
            count[2] += answer.lower().count(word)
        counter.append(count)

    return counter


def choice_feature(input_dict):
    words = ['which', 'or ', 'instead ', 'either', 'other']
    counter = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]
        for word in words:
            count[0] += title.lower().count(word)
            count[1] += question.lower().count(word)
            count[2] += answer.lower().count(word)
        counter.append(count)

    return counter


def comparison_feature(input_dict):
    words = ['same', 'different', 'equal', 'opposite', 'like', 'better'
                                                               'worse']
    counter = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]
        for word in words:
            count[0] += title.lower().count(word)
            count[1] += question.lower().count(word)
            count[2] += answer.lower().count(word)
        counter.append(count)

    return counter


def comma_feature(input_dict):
    counter = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]

        count[0] += title.count(',')
        count[1] += question.count(',')
        count[2] += answer.count(',')
        counter.append(count)

    return counter


def period_feature(input_dict):
    counter = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]

        count[0] += title.count('.')
        count[1] += question.count('.')
        count[2] += answer.count('.')
        counter.append(count)

    return counter


def instruction_feature(input_dict):
    words = ['first', 'second', 'third', 'last', 'next', 'before',
             'you should']
    counter = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]
        for word in words:
            count[0] += title.lower().count(word)
            count[1] += question.lower().count(word)
            count[2] += answer.lower().count(word)
        counter.append(count)

    return counter


def parts_of_speech_feature(input_dict):
    words = ['noun', 'verb', 'adjective', 'adverb', 'pronoun', 'preposition',
             'conjunction', 'interjection', 'article', 'parts of speech']
    counter = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]
        for word in words:
            count[0] += title.lower().count(word)
            count[1] += question.lower().count(word)
            count[2] += answer.lower().count(word)
        counter.append(count)

    return counter


def language_feature(input_dict):
    words = ['mandarin', 'chinese', 'spanish', 'english', 'hindi', 'hindustani'
                                                                   'bengali',
             'portuguese', 'russian', 'japanese', 'punjabi',
             'turkish', 'korean', 'french', 'german', 'italian', 'vietnamese'
                                                                 'arabic',
             'polish', 'dutch', 'romainian', 'greek', 'latin']
    counter = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]
        for word in words:
            count[0] += title.lower().count(word)
            count[1] += question.lower().count(word)
            count[2] += answer.lower().count(word)
        counter.append(count)

    return counter


def sonic_feature(input_dict):
    words = ['sound', 'pronounce', 'pronunciation', 'enunciate', 'enunciation',
             'hear', 'accent', 'tone', 'voice', 'silent', 'say', 'speak'
                                                                 'utter',
             'stress', 'voice', 'tone', 'intonation', 'phonetic',
             'sonic']
    counter = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]
        for word in words:
            count[0] += title.lower().count(word)
            count[1] += question.lower().count(word)
            count[2] += answer.lower().count(word)
        counter.append(count)

    return counter


def reference_feature(input_dict):
    words = ['book', 'dictionary', 'thesarus', 'dictionaries', 'corpus', 'text'
                                                                         'oxford',
             'literature', 'poem', 'essay', 'paragraph',
             'vocabulary', 'author', 'shakespeare', 'ngram', 'etymology']
    counter = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]
        for word in words:
            count[0] += title.lower().count(word)
            count[1] += question.lower().count(word)
            count[2] += answer.lower().count(word)
        counter.append(count)

    return counter


def word_part_feature(input_dict):
    words = ['suffix', 'vowel', 'consonant', 'root', 'prefix', 'syllable']
    counter = []
    titles = input_dict['question_title']
    questions = input_dict['question_body']
    answers = input_dict['answer']

    for title, question, answer in zip(titles, questions, answers):
        count = [0, 0, 0]
        for word in words:
            count[0] += title.lower().count(word)
            count[1] += question.lower().count(word)
            count[2] += answer.lower().count(word)
        counter.append(count)

    return counter


def preprocess(data_path):
    with data_path.open(mode='r') as f:
        csv_reader = csv.DictReader(f)
        ids = []

        input_dict = OrderedDict()
        target_dict = OrderedDict()
        feature_dict = OrderedDict()

        for field in input_fields:
            input_dict[field] = []

        for field in target_fields:
            target_dict[field] = []
        for row in csv_reader:

            ids.append(row[id_field])
            for field in input_fields:
                if field == 'question_title':
                    item = ' ' + row[field]
                else:
                    item = row[field]
                input_dict[field].append(item)

            for field in feature_fields:
                feature_function = getattr(preprocess, f"{field}_feature")
                feature_dict[field] = feature_function(input_dict)

        yield ids, input_dict, feature_dict


class KaggleQuestDataset(data.Dataset):
    def __init__(self, csv_path):
        ids, inputs, features = preprocess(csv_path)
        self.data = features
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        output = {}
        for field in self.data:
            output[field] = torch.tensor(self.data[field][index])
        return self.ids[index], output

class RobertaForQUEST(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super(RobertaForQUEST, self).__init__(config)
        model_name = 'roberta-base'
        self.bert = transformers.RobertaModel.from_pretrained(model_name,
                                                              config=config)
        self.embedding = torch.nn.Linear(5, 5)
        self.embedding2 = torch.nn.Linear(63, 10)
        self.linear_layer1 = torch.nn.Linear(3160, 2000)
        self.linear_layer2 = torch.nn.Linear(2000, 1000)
        self.linear_layer3 = torch.nn.Linear(1000, 500)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(500, config.num_labels)

        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self,
                input_ids=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                category=None,
                sentence_lengths=None,
                newlines=None,
                similarity=None,
                hyperlinks=None,
                first_person=None,
                latex=None,
                brackets=None,
                sentiment=None,
                spell=None,
                token_type_ids=None,
                punctuation=None,
                question_mark=None,
                exclamation=None,
                yes_no=None,
                numeric_answer=None,
                short_keyword=None,
                consequence=None,
                how=None,
                choice=None,
                comparison=None,
                comma=None,
                period=None,
                instruction=None,
                host=None,
                self_reference=None,
                parts_of_speech=None,
                language=None,
                sonic=None,
                reference=None,
                word_part=None
                ):
        outputs = self.bert(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            )

        hidden_states = outputs[2]

        h6 = torch.mean(hidden_states[-1], 1).reshape((-1, 1, 768))
        h5 = torch.mean(hidden_states[-2], 1).reshape((-1, 1, 768))
        h4 = torch.mean(hidden_states[-3], 1).reshape((-1, 1, 768))
        h3 = torch.mean(hidden_states[-4], 1).reshape((-1, 1, 768))

        embedding = self.embedding(category)
        embedding2 = self.embedding2(host)

        e = embedding.view(-1, 1, 5)
        e2 = embedding2.view(-1, 1, 10)
        sl = sentence_lengths.view(-1, 1, 3)
        n = newlines.view(-1, 1, 2)
        s = similarity.view(-1, 1, 1)
        h = hyperlinks.view(-1, 1, 2)
        fp = first_person.view(-1, 1, 2)
        l = latex.view(-1, 1, 2)
        b = brackets.view(-1, 1, 2)
        sp = spell.view(-1, 1, 3)

        p = punctuation.view(-1, 1, 3)
        q = question_mark.view(-1, 1, 3)
        ex = exclamation.view(-1, 1, 3)
        y = yes_no.view(-1, 1, 3)
        na = numeric_answer.view(-1, 1, 3)
        sk = short_keyword.view(-1, 1, 3)
        c = consequence.view(-1, 1, 3)

        hw = how.view(-1, 1, 3)
        ch = choice.view(-1, 1, 3)
        co = comparison.view(-1, 1, 3)
        cm = comma.view(-1, 1, 3)
        pd = period.view(-1, 1, 3)
        i = instruction.view(-1, 1, 3)

        sr = self_reference.view(-1, 1, 2)
        ps = parts_of_speech.view(-1, 1, 3)
        la = language.view(-1, 1, 3)
        sn = sonic.view(-1, 1, 3)
        rf = reference.view(-1, 1, 3)
        wp = word_part.view(-1, 1, 3)

        hcat = torch.cat([h3, h4, h5, h6], 2)

        fcat = torch.cat(
            [e, e2, sl, s, n, h, fp, l, b, sp, p, q, ex, y, na, sk,
             c, hw, ch, co, cm, pd, i, sr, ps, la, sn, rf, wp], 2)

        cat = torch.cat([hcat, fcat], 2)
        cat = torch.nn.ReLU()(cat)

        layer1 = self.linear_layer1(cat)
        layer1 = torch.nn.LeakyReLU()(layer1)
        layer2 = self.linear_layer2(layer1)
        layer2 = torch.nn.LeakyReLU()(layer2)
        layer3 = self.linear_layer3(layer2)
        layer3 = torch.nn.LeakyReLU()(layer3)
        logits = self.classifier(layer3)

        return torch.squeeze(logits)


if __name__ == '__main__':

    test_dir = pathlib.Path('/home/scott/Projects/kaggle__google_quest_q_and_a_labeling/data/test.csv')
    test_model = pathlib.Path('/home/scott/Projects/kaggle__google_quest_q_and_a_labeling/data/saved_models/load_model')
    output_path = pathlib.Path('/home/scott/Projects/kaggle__google_quest_q_and_a_labeling/data/results.csv')

    GPU_CAP_TEST = 50

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Using {device}.")

    model_name = 'roberta-base'
    model_config = transformers.RobertaConfig.from_pretrained(
        model_name,
        output_hidden_states=True,
        num_labels=30,
    )

    bnet = model.RobertaForQUEST(config=model_config)

    state_dict_path = None

    for filepath in test_dir.iterdir():
        state_dict_path = filepath.resolve().as_posix()
    if state_dict_path:
        print('Loading saved model.')
        state_dict = torch.load(state_dict_path)
        bnet.load_state_dict(state_dict)

    bnet.to(device)

    dataset = KaggleQuestDataset(test_dir)

    testloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=GPU_CAP_TEST,
                                             shuffle=True,
                                             num_workers=0
                                             )
    offset = 0
    predicted = np.zeros((len(dataset), len(target_fields)))
    ids = []
    for data in testloader:
        entry_id, features = data
        ids.append(entry_id)
        for field in features:
            features[field] = features[field].to(device)

        output = bnet(**features)

        predicted[offset: offset + output.size()[0], :] = \
            output.cpu().numpy()

    df_ids = pd.DataFrame(ids)
    df_data = pd.DataFrame(predicted)
    df_data.columns(target_fields)
    df_data.insert(0,'qa_id',df_ids)
    df_data.to_csv(ouput_dir)