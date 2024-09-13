import os

from torchmetrics.text import EditDistance, BLEUScore, ROUGEScore

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"]= "true"
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pickle

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score, balanced_accuracy_score
from transformers import RobertaForMaskedLM, RobertaConfig
from transformers import PreTrainedTokenizerFast
import numpy as np
import random
import evaluate
import gc
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import logging
from torch import nn
from transformers import Trainer
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate import meteor
from collections import Counter


#torch.cuda.set_device(1)
#print(device)
##free GPU cache
def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()
    gc.collect()
    torch.cuda.empty_cache()
    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)
    print("GPU Usage after emptying the cache")
    gpu_usage()

#ree_gpu_cache()

##set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(123)

tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer.json")
tokenizer.add_special_tokens({'bos_token': '<s>'})
tokenizer.add_special_tokens({'eos_token': '</s>'})
tokenizer.add_special_tokens({'unk_token': '<unk>'})
tokenizer.add_special_tokens({'mask_token': '<mask>'})
tokenizer.add_special_tokens({'pad_token': '<pad>'})

def testing(path):
    with open('results_%s.pkl'%path,'rb') as f:
        test_dataset = pickle.load(f)

    print(len(test_dataset))
    em = 0
    em_10 = 0
    masks = 0
    mrrs = 0
    number = len(final_valid_data)
    for data in final_valid_data:
        #no = no +1
        #print(no)
        input = {'input_ids': data['input_ids']}
        labels = data['labels'][0]
        with torch.no_grad():
            output = model(**input).logits
        mask_token_index = (input['input_ids'] == 4)[0].nonzero(as_tuple=True)[0]

        masks = masks + len(mask_token_index)
        predicted_token_id = output[0, mask_token_index].argmax(axis=-1)
        #print(predicted_token_id)

        for j in range(0, len(mask_token_index)):
            if predicted_token_id[j] == labels[mask_token_index[j]]:
                em = em + 1

        a, idx1 = torch.sort(output[0, mask_token_index], descending=True)
        predicted_token_id_10 = idx1[:,:10]
        for x in range(0, len(mask_token_index)):
            for y in range(0, 10):
                if labels[mask_token_index[x]] == predicted_token_id_10[x,y]:
                    em_10 = em_10 + 1
                    mrrs = mrrs + float(1/(y+1))
                    break
 

    metrics = {
        'em': em,
        'masks': masks,
        'Acc-1': float(em/masks),
        'Acc-10': float(em_10/masks),
        'MRR': float(mrrs/masks),
    }
    print(metrics)

print("No Pretraining:")
testing("no-pretraining")

