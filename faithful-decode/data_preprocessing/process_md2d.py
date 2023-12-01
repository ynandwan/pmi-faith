import argparse
import copy
from enum import Enum
import json
import logging
import os
from pathlib import Path
import random

PREFIX = 'multidoc2dial_dial'
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



class GroundingErrorType(Enum):
    SAME_DOCUMENT = 0
    SAME_DOMAIN = 1
    DIFFERENT_DOMAIN = 2
    NO_ERROR = 3


class ContextResponsePair:
    def __init__(self, id, context, response, grounded_doc_ids, grounding_error_type=GroundingErrorType.NO_ERROR):
        self.id = id
        self.context = context
        self.response = response
        self.grounded_doc_ids = grounded_doc_ids
        self.grounding_error_type = grounding_error_type

    def get_context_as_string(self):
        returnString = ""
        for utt in self.context:
            returnString += utt["speaker"] + ": " + utt["text"].strip() + "\n"
        return returnString

    def get_context_as_list(self):
        history = []
        last_speaker = None
        for utt in self.context:
            if last_speaker and utt['speaker'] == last_speaker:
                history[-1] += f'. {utt["text"].strip()}'
            else:
                history.append(utt["speaker"] + ": " + utt["text"].strip())
            last_speaker = utt['speaker']
        return history

    def get_response(self):
        return self.response

    def get_grounded_doc_ids(self):
        return self.grounded_doc_ids

    def get_id(self):
        return self.id

    def get_grounding_error_type(self):
        return self.grounding_error_type

    def to_dict(self) -> dict:
        return {
            "id": self.get_id(),
            "context": self.get_context_as_string().strip(),
            "response": self.get_response().strip(),
            "grounded_doc_ids": self.get_grounded_doc_ids(),
            "grounding_error_type": self.get_grounding_error_type().value,
        }


class DataLoader:
    def __init__(self, data_folder, options={}):
        # self.context_response_pairs is a list of ContextResponsePair objects
        # self.grounded_docs is a dictionary of document ids and its content
        pass

    def __len__(self):
        return len(self.context_response_pairs)

    def __getitem__(self, index):
        return self.get_data_point_at(index)

    def get_data_point_at(self, i):
        if i < 0 or i > len(self.context_response_pairs):
            return None
        return self.context_response_pairs[i]

    def print_data_point_at(self, i):
        if i < 0 or i > len(self.context_response_pairs):
            return None
        crp = self.context_response_pairs[i]
        return (
            "GROUNDED DOC:\n"
            + self.get_doc_content(crp.get_grounded_doc_ids())
            + "\n\nCONTEXT:\n"
            + crp.get_context_as_string()
            + "\nGT RESPONSE:\n"
            + crp.get_response()
            + "\n\n"
        )

    def get_single_doc_content(self, grounded_doc_id):
        if grounded_doc_id not in self.grounded_docs:
            return ""

        return self.grounded_docs[grounded_doc_id]

    def get_doc_content(self, grounded_doc_ids):
        contents = ""
        added_contents = set([])
        for grounded_doc_id in grounded_doc_ids:
            single_doc_content = self.get_single_doc_content(grounded_doc_id)
            if single_doc_content not in added_contents:
                contents += " " + single_doc_content
                added_contents.add(single_doc_content)
                contents = contents.strip()
        return contents

    def get_random_cr_pair(self):
        random_id = random.randint(0, len(self.context_response_pairs) - 1)
        return self.context_response_pairs[random_id]


class MultiDoc2DialDataLoader(DataLoader):
    def __init__(
        self,
        data_folder: str,
        fname: str,
        documents_file: str,
        user_tag: str = "user",
        agent_tag: str = "agent",
        segmentation_type: str = "passage",
    ):
        self.user_tag = user_tag
        self.agent_tag = agent_tag

        self.segmentation_type = segmentation_type

        self.populate_context_response_pairs(os.path.join(data_folder, fname))

        self.populate_grounded_docs(documents_file)

    def print_cr_pair_id(self, cr_pair_id):
        for i, cr_pair in enumerate(self.populate_context_response_pairs):
            if cr_pair.get_id == cr_pair_id:
                self.print_data_point_at(i)

    def populate_context_response_pairs(self, dialog_json_filepath):
        self.context_response_pairs = []
        no_of_dialogs = 0
        no_of_cr_pairs = 0
        with open(dialog_json_filepath) as json_file:
            data = json.load(json_file)
            dial_data = data["dial_data"]
            for domain in dial_data.keys():
                dialogs = dial_data[domain]
                for dialog in dialogs:
                    no_of_dialogs += 1
                    dialog_id = dialog["dial_id"]
                    turns = dialog["turns"]
                    context = []
                    for turn in turns:

                        speaker = turn["role"]
                        utterance = turn["utterance"]
                        turn_id = turn["turn_id"]
                        if speaker == "agent":
                            no_of_cr_pairs += 1
                            doc_ids = []
                            for i in range(len(turn["references"])):
                                if self.segmentation_type == "document":
                                    doc_id = turn["references"][i]["doc_id"]
                                else:
                                    doc_id = turn["references"][i]["doc_id"] + "##" + turn["references"][i]["id_sp"]
                                if doc_id not in doc_ids:
                                    doc_ids.append(doc_id)

                            cr_pair_id = dialog_id + "#" + str(turn_id)
                            grounding_error_type = GroundingErrorType.NO_ERROR
                            if "grounding_error_type" in turn:
                                grounding_error_type = GroundingErrorType(turn["grounding_error_type"])

                            self.context_response_pairs.append(
                                ContextResponsePair(
                                    cr_pair_id,
                                    copy.deepcopy(context),
                                    utterance,
                                    doc_ids,
                                    grounding_error_type=grounding_error_type,
                                )
                            )
                            context.append({"speaker": self.agent_tag, "text": utterance})
                        else:
                            context.append({"speaker": self.user_tag, "text": utterance})
        print("\tFound", no_of_dialogs, "Dialogs with", no_of_cr_pairs, "Context-Response Pairs")

    def populate_grounded_docs(self, doc_json_filepath):
        self.grounded_docs = {}
        self.doc_to_domain_map = {}
        with open(doc_json_filepath) as json_file:
            data = json.load(json_file)
            doc_data = data["doc_data"]
            for domain in doc_data.keys():
                domain_docs = doc_data[domain]
                for doc_id in domain_docs.keys():

                    if self.segmentation_type == "document":
                        self.grounded_docs[doc_id] = domain_docs[doc_id]["doc_text"]
                        self.doc_to_domain_map[doc_id] = domain
                    else:
                        spans = domain_docs[doc_id]["spans"]
                        current_title = None
                        current_content = ""
                        current_id_sps = set([])

                        keys_as_int = [int(i) for i in spans.keys()]
                        for key in sorted(keys_as_int):
                            current_span = spans[str(key)]
                            if current_title != current_span["title"]:
                                if current_title != None:
                                    for id_sp in current_id_sps:
                                        self.grounded_docs[doc_id + "##" + id_sp] = current_content
                                        self.doc_to_domain_map[doc_id + "##" + id_sp] = domain
                                current_id_sps = set([])
                                current_content = ""
                                current_title = current_span["title"]

                            current_id_sps.add(current_span["id_sp"])
                            current_content += current_span["text_sp"]
                        if current_title != None:
                            for id_sp in current_id_sps:
                                self.grounded_docs[doc_id + "##" + id_sp] = current_content
                                self.doc_to_domain_map[doc_id + "##" + id_sp] = domain

        print("\tFound", len(self.grounded_docs), "Documents")

def generate_utterances(data_dir, split):
    utterances = []
    
    dl = MultiDoc2DialDataLoader(data_dir, f"multidoc2dial_dial_{split}.json", documents_file=Path(data_dir) / "multidoc2dial_doc.json")
    for i in range(len(dl)):
        cr_pair = dl.get_data_point_at(i)

        document_content = dl.get_doc_content(cr_pair.get_grounded_doc_ids())
        response = cr_pair.get_response()

        utterance = {
            'speaker': 'agent',
            'history': cr_pair.get_context_as_list(),
            'knowledge': document_content,
            'response': response
        }
        utterances.append(utterance)
    return utterances


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/multidoc2dial')
    return parser.parse_args()


def main():
    args = setup_args()
    logging.info(args)

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    for split in ['train', 'validation', 'test']:
        utterances = generate_utterances(args.data_dir, split)
        utterances_dict = { 
            'utterances': utterances
        }
        with open(Path(args.output_dir) / f'{split}.json', 'w') as fw:
            json.dump([utterances_dict], fw, indent=2)
        
        logging.info(f'{split} utterances: {len(utterances)}')

if __name__ == '__main__':
    main()
