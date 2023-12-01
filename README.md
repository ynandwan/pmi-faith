# Pointwise Mutual Information Based Metric and Decoding Strategy for Faithful Generation in Document Grounded Dialogs


This repo contains the code for our paper [Pointwise Mutual Information Based Metric and Decoding Strategy for Faithful Generation in Document Grounded Dialogs](https://arxiv.org/abs/2305.12191)

In our paper, we propose:
1. `pmi-faith`: a metric for quantifying faithfulness of a response to a given document
2. `pmi-decode`: a decoding strategy that generates more faithful responses than simple likelihood based decoding. 

Accordingly, our repo is divided into two parts:

1. `faithfulness-metrics`: This folder contains code for measuring pmi-faith
2. `faithful-decode`: This folder contains finetuning and inference code. It has been copied from a public repo [FaithDial](https://github.com/McGill-NLP/FaithDial/tree/main). To implement `pmi-decode`, we only modify their inference code while keeping the training part as it is. 


## faithfulness-metrics

### Requirements

Create a conda env using `env.yaml`. 
```
conda env create -f env.yaml
conda activate pmifaith
python -m spacy download en_core_web_sm
```


### Computing PMI-Faith

Given a grounding document, *D*, dialogue history, *h*, and a response, *r*, we define *PMI-Faith* as:
```
PMI-Faith(D, h, r) = log Pr ( r | D,h) - log Pr (r | h)
```

To compute the log probabilities, we can use any LLM. In our experiments, we use `Bloom-560m` to compute PMI.

As suggested by one of the reviewers, we also compute unconditional PMI as one of the metrics to quantify faithfulness. In unconditional PMI-Faith, the probabilities are not conditioned on dialogue history *h*. It is defined as:
```
Uncond-PMI-Faith(D, h, r) = log Pr ( r | D) - log Pr (r)
```

#### Usage

Checkout `notebooks/compute_faithfulness_api.ipynb' for an example.

```
import src.compute_faithfulness_api as faithfulness
model_name = 'bigscience/bloom-560m'
model,tokenizer = faithfulness.get_huggingface_pretrained_model(model_name, device_map = 'auto')

api = faithfulness.ComputeFaithfulness((model, tokenizer))
document = ' '
response = ' ' 
history = ' '
result = api.compute_faithfulness(document, history, response = response)
print(result)
```
## faithful-decode

This folder contains finetuning and inference code. Finetuning code has been copied from a public repo [FaithDial](https://github.com/McGill-NLP/FaithDial/tree/main). To implement `pmi-decode`, we only modify their inference code while keeping the training part as it is. We also provide scripts to preprocess Topical-Chat and Multidoc2dial datasets into the format required by FaithDial.

### Requirements

Please follow the instructions in the README of FaithDial to setup the training environment. We have
```
cd faithful-decode
conda env create -f faithdial_conda_env.yaml
conda activate faithdial
pip install -r requirements.txt
```

### Preprocess Topical-Chat and Multidoc2Dial

FaithDial requires data to be in a specific format for training and inference. Accordingly, we provide scripts to process Topical-Chat and MultiDoc2Dial in the required format

#### Topical-chat
Please ensure that you download the Topical-Chat dataset by following the instructions in official repo: https://github.com/alexa/Topical-Chat.

Run the following commands to process the data:

```
    cd faithful-decode
    python data_preprocessing/process_topicalchat.py --conversations_dir=../../Topical-Chat/conversations --knowledge_dir=../../Topical-Chat/reading_sets/post-build --outdir=data/tc_nopersonal --split=train
    python data_preprocessing/process_topicalchat.py --conversations_dir=../../Topical-Chat/conversations --knowledge_dir=../../Topical-Chat/reading_sets/post-build --outdir=data/tc_nopersonal --split=valid_rare
    python data_preprocessing/process_topicalchat.py --conversations_dir=../../Topical-Chat/conversations --knowledge_dir=../../Topical-Chat/reading_sets/post-build --outdir=data/tc_nopersonal --split=test_rare

```

#### Multidoc2Dial
Please ensure that you download the multidoc2dial dataset by following the instructions in its official repo: https://github.com/IBM/multidoc2dial
Once you follow the download process, it will download the document and dialogue data into folder `<path_to_md2d_repo>/data/multidoc2dial`

Run the following commands to process the data:

```
    cd faithful-decode
    python data_preprocessing/process_md2d.py --data_dir <path_to_md2d_repo>/data/multidoc2dial

```
The above command will create `data/multidoc2dial` folder with `train.json`, `test.json` and `validation.json` files.


### Training

Please see `faithful-decode/scripts/train.sh` script for the exact train commands. The command below will finetune `bart-large` model on FaithDial dataset. It will create two folders in the `output_dir (trained-models/fd/bart-large/lr-5)`: one for best model checkpoint (`best_model`)  and the other for the last model checkpoint (`last_model`)

```
cd faithful-decode
python models/dialog.py --model_name_or_path facebook/bart-large --do_train --output_dir trained-models/fd/bart-large/lr-5  --warmup_ratio 0.04 --max_seq_length 512 --max_history 2 --gradient_accumulation_steps 4 --num_train_epochs 10 --train_batch_size 32 --loss_truncation 0 --filter 0 --learning_rate 6.25e-5
```

### Generation
Please see generate scripts within `faithful-decode/scripts` folder. The following command will generate responses on FaithDial test set with `alpha=0.25` and masking `top_p=0.6`:

```
python models/pmi_generate.py --model_name_or_path trained-models/fd/bart-large/lr-5/best_model --output trained-models/fd/bart-large/lr-5/generated/best_model --batch_size 32 --top_p 0.6 --pmi_weight 0.25 --pmi_decode
```

#### Computing the faithfulness-adjusted score during decoding

At each step *t*, PMI decoding needs to adjust the log likelihood with `pmi-faith` of the sequence decoded so far. 
We do so by defining a custom [`LogitProcessor`](https://huggingface.co/docs/transformers/v4.35.2/en/internal/generation_utils#transformers.LogitsProcessor): [`PMILogitsProcessor`](faithful-decode/models/logits_processors.py) and pass it to the generate function in [`pmi_generate.py`](faithful-decode/models/pmi_generate.py#L310).

It assumes that the first half of the batch is being decoded with `document` and `dialogue history` in the input context, and the second half of the batch is decoding the same examples as in the first half, but with only corresponding `dialogue history` in the input context. Logits are converted into log probabilities and `pmi-faith` adjusted score is computed as:
 
 ```
    final_score = (
                pmi_weight * (log_prob[:(batch_size//2)] - log_prob[(batch_size//2):])
                + (1.0 - self.pmi_weight) * log_prob[:(batch_size//2)]
            )
 ```

To ensure that the same token gets selected when decoding with (1st half of the batch) and without (2nd half of the batch) document, the `PMILogitsProcessor` returns the same scores for the corresponding inputs (with and without document in the context) so that any non-sampling based selection (greedy or beam search) would pick the same token:

```
    return final_score.repeat(2,1)
```

#### Hack for sampling based method (nucleus sampling and beam sampling)

Even though the `PMILogitsProcessor` returns the same scores for decoding with and without document, there is no gaurantee that the same token will be selected due to randomness in the sampling based decoding strategies. To do so, we need to first sample, ignore the tokens sampled by the lower half of the batch (decoding without document), and repeat the tokens selected by the 1st half of the batch (decoding with document):

```
    probs = nn.functional.softmax(next_token_scores, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

    next_tokens[next_tokens.shape[0] // 2 :] = next_tokens[
       : next_tokens.shape[0] // 2
    ]
            
```

The above code needs to go in [`generate` function in Transformers](https://github.com/huggingface/transformers/blob/94b3f544a1f5e04b78d87a2ae32a7ac252e22e31/src/transformers/generation_utils.py#L2518). 
We provide a modified [`generation_utils.py`](/faithful-decode/models/generation_utils_pmi.py). For sampling based decoding, we need to replace the `generation_utils.py` in Transformers with the modified one provided in this repo.



