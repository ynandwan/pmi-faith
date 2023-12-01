
#Adapted from https://github.com/nlpxucan/ZRKGC
"""
    python data_preprocessing/process_topicalchat.py --conversations_dir=../../Topical-Chat/conversations --knowledge_dir=../../Topical-Chat/reading_sets/post-build --outdir=data/tc_nopersonal --split=train
    python data_preprocessing/process_topicalchat.py --conversations_dir=../../Topical-Chat/conversations --knowledge_dir=../../Topical-Chat/reading_sets/post-build --outdir=data/tc_nopersonal --split=valid_rare
    python data_preprocessing/process_topicalchat.py --conversations_dir=../../Topical-Chat/conversations --knowledge_dir=../../Topical-Chat/reading_sets/post-build --outdir=data/tc_nopersonal --split=test_rare
"""



import os
import sys
import argparse

import pandas as pd
from tqdm import tqdm
import numpy as np
import yaml
import json
import re
import string
import copy

def get_args(description='', parse_string=None):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--conversations_dir',type=str, required=True,
                        help='Conversations dir')

    parser.add_argument('--knowledge_dir', type=str, required=True,
                        help = 'knowledge dir')

    parser.add_argument('--split', type = str, default = 'valid_rare',
                        choices=['test_freq', 'test_rare', 'train', 'valid_freq', 'valid_rare'])

    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--skip_personal', type=int, default = 2,
                        help = '0 == do not skip anything,'
                        ' 1 == skip if PK is **the** only source'
                        ' 2 = skip even if PK is **a** source')

    if parse_string is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(parse_string)

    return args



def get_knowledge(every_content, knowledge_turn):
    know_ids = every_content["knowledge_source"]
    single_knowledges = knowledge_turn[every_content["agent"]]
    article_knowledge = knowledge_turn["article"]
    pk = False
    knowledge_list = []

    for know_id in know_ids:
        wiki_know = ''

        if know_id == "Personal Knowledge":
            pk = True
        elif know_id in single_knowledges:
            assert "shortened_wiki_lead_section" or "summarized_wiki_lead_section" in single_knowledges[
                know_id]

            if "shortened_wiki_lead_section" in single_knowledges[know_id]:
                wiki_know = single_knowledges[know_id]["shortened_wiki_lead_section"]
            else:
                wiki_know = single_knowledges[know_id]["summarized_wiki_lead_section"]
            knowledge_list.append(wiki_know.strip())

            for item in single_knowledges[know_id]["fun_facts"]:
                knowledge_list.append(item.strip())
        elif know_id in article_knowledge:
            knowledge_list.append(article_knowledge[know_id].strip())

    knowledge_list = list(filter(lambda x: len(x) > 0,knowledge_list))
    this_knowledge = ''

    if len(knowledge_list) > 0:
        knowledge_list = [x if x[-1] in string.punctuation else '{}.'.format(x) for x in knowledge_list]
        this_knowledge = ' '.join(knowledge_list).encode('unicode_escape').decode('utf-8')
    #

    return this_knowledge, pk

#Adapted from https://github.com/nlpxucan/ZRKGC

def extract_topical_chat_dataset(args):
    dialog_path = os.path.join(args.conversations_dir, args.split+'.json')
    know_path = os.path.join(args.knowledge_dir, args.split+'.json')

    with open(dialog_path, 'r', encoding='utf-8') as f:
        conversations_file = json.load(f)
    with open(know_path, 'r', encoding='utf-8') as f:
        know_file = json.load(f)

    all_dialogues = []
    skip_utterances = 0
    skip_dialogues = 0
    total_utterances = 0
    count_accented_response = 0
    #count_accented_knowledge_snippets = 0

    for dialog_id, dialog_turn in conversations_file.items():
        message_history = []
        dialog_content = dialog_turn["content"]
        knowledge_turn = know_file[dialog_id]
        this_dialogue = {}
        this_dialogue['dialog_idx'] = dialog_id
        this_dialogue['utterances'] = []
        prev_agent = ''

        for i,every_content in enumerate(dialog_content):
            response0 = every_content["message"].strip()
            response = response0.encode('unicode_escape').decode('utf-8')

            if response != response0:
                count_accented_response += 1
                #print('BEFORE: {}'.format(response0))
                #print('AFTER: {}'.format(response))
                #print("$$$$$$")


            agent = every_content['agent']
            assert agent != prev_agent
            prev_agent = agent

            if len(message_history) > 0:
                this_knowledge, has_personal = get_knowledge(every_content, knowledge_turn)

                if has_personal and (args.skip_personal == 2):
                    skip_utterances += 1
                elif (len(this_knowledge) > 0) or (args.skip_personal == 0):
                    #len(this_knowledge) > 0 indicates that personal is not the only source of knowledge.
                    utterance = {'history': copy.deepcopy(message_history), 'knowledge': this_knowledge, 'response': response}
                    this_dialogue['utterances'].append(utterance)
                    total_utterances += 1
                else:
                    skip_utterances +=1
            #
            message_history.append(response)

        if len(this_dialogue['utterances']) > 0:
            all_dialogues.append(this_dialogue)
        else:
            skip_dialogues +=1


    print("\n#Included utterances: ", total_utterances,
          "\n###Skipped utterances: ", skip_utterances,
          "\n#Skipped dialogues: ", skip_dialogues,
          "\n#Accented response: ", count_accented_response
    )

    os.makedirs(args.outdir,exist_ok =True)

    with open(os.path.join(args.outdir, 'tc_{}.json'.format(args.split)),"w", encoding="utf-8") as fh:
        json.dump(all_dialogues, fh, indent = 2)



def main():
    parse_string = ['--conversations_dir=Topical-Chat/conversations',
                    '--knowledge_dir=Topical-Chat/reading_sets/post-build',
                   '--outdir=FaithDial/data/tc_nopersonal',
                   '--skip_personal=2']


    split_string_list = [
                         '--split=test_rare',
                         '--split=valid_rare',
                         '--split=train'
    ]

    for this_split in split_string_list:
        this_parse_string = parse_string + [this_split]
        args = get_args(description='Topical chat preprocessing',parse_string=this_parse_string)
        print(args)
        extract_topical_chat_dataset(args)



if __name__ == '__main__':
    args = get_args()
    print(args)
    extract_topical_chat_dataset(args)
