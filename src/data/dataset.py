import json

import torch

from torch.utils import data
from sklearn.model_selection import KFold

class QuestTrainDataset(data.Dataset):
    def __init__(self, redis_db, indices=None, splits=None):
        self.redis_db = redis_db

        if not indices:
            self.ids = json.loads(redis_db.get('ids'))
        else:
            self.ids = indices

        if not splits:
            self.splits = [0]
        else:
            self.splits = splits

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        qa_id = self.ids[index]
        text = json.loads(self.redis_db.hget(qa_id, 'text'))
        features = json.loads(self.redis_db.hget(qa_id, 'features'))
        label = json.loads(self.redis_db.hget(qa_id, 'label'))
        mask = json.loads(self.redis_db.hget(qa_id, 'mask'))
        return torch.tensor(text),torch.tensor(features), torch.tensor(
            label), torch.tensor(mask)

class QuestTestDataset(data.dataset)

class KfoldQuestDataset:
    def __init__(self, folds, redis_db):
        self.redis_cnxn = redis_db
        self.ids = json.loads(redis_db.get('ids'))

        kf =  KFold(n_splits=folds, shuffle=True)

        train_indices = []
        test_indices = []
        segments = [0]

        for train, test in kf.split(self.ids):
            train_indices += train
            test_indices += test
            segments.append(len(train))



        # self.target = ['question_asker_intent_understanding',
        #                'question_body_critical',
        #                'question_conversational',
        #                'question_expect_short_answer',
        #                'question_fact_seeking',
        #                'question_has_commonly_accepted_answer',
        #                'question_interestingness_others',
        #                'question_interestingness_self',
        #                'question_multi_intent',
        #                'question_not_really_a_question',
        #                'question_opinion_seeking',
        #                'question_type_choice',
        #                'question_type_compare',
        #                'question_type_consequence',
        #                'question_type_definition',
        #                'question_type_entity',
        #                'question_type_instructions',
        #                'question_type_procedure',
        #                'question_type_reason_explanation',
        #                'question_type_spelling',
        #                'question_well_written',
        #                'answer_helpful',
        #                'answer_level_of_information',
        #                'answer_plausible',
        #                'answer_relevance',
        #                'answer_satisfaction',
        #                'answer_type_instructions',
        #                'answer_type_procedure',
        #                'answer_type_reason_explanation',
        #                'answer_well_written'
        #                ]