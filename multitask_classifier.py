import time, random, numpy as np, argparse, sys, re, os
from datetime import datetime
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import *

from bert import BertModel
from optimizer import AdamW, AdamaxW
from tqdm import tqdm
from itertools import zip_longest, cycle
from pcgrad import PCGrad

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_multitask, test_model_multitask

# libraries for plotting
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl


TQDM_DISABLE=False  # set False to enable iteration timer on console

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5
MAX_SIMILARITY = 5  # actual scale for similarity


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        # raise NotImplementedError
        self.sst_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.para_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sts_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.sst_linear = nn.Linear(config.hidden_size, len(config.num_labels))
        # self.para_linear = nn.Linear(config.hidden_size * 3, 1)  #  original is two sen embeddings and their difference
        self.para_linear = nn.Linear(config.hidden_size, 1)
        self.sts_linear = nn.Linear(config.hidden_size, 1)


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        # raise NotImplementedError

        embeddings = self.bert(input_ids, attention_mask)['pooler_output']
        return embeddings


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs probility distribution for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        The output contains 5 normalized probilities for each sentence.
        '''
        ### TODO
        # raise NotImplementedError

        embeddings = self.forward(input_ids, attention_mask)
        embeddings = self.sst_dropout(embeddings)
        logits = self.sst_linear(embeddings) # unnormalized
        probs = F.softmax(logits, dim=-1)  # normalized prob distribution
        return probs


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        The output is normalized by sigmoid function, which is probility of paraphrase.
        '''
        ### TODO
        # raise NotImplementedError

        # embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        # embeddings_2 = self.forward(input_ids_2, attention_mask_2)
        # embeddings_diff = torch.abs(embeddings_1 - embeddings_2)
        # embeddings = torch.cat((embeddings_1, embeddings_2, embeddings_diff), dim=-1)  # simply concat two embeddings

        input_ids = torch.cat((input_ids_1, input_ids_2), dim=-1)
        attention_mask = torch.cat((attention_mask_1, attention_mask_2), dim=-1)
        embeddings = self.forward(input_ids, attention_mask)

        embeddings = self.para_dropout(embeddings)
        logits = self.para_linear(embeddings)  # unnormalized
        probs = torch.sigmoid(logits)
        return probs.squeeze()


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        The output logit
        '''
        ### TODO
        # raise NotImplementedError

        # embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        # embeddings_2 = self.forward(input_ids_2, attention_mask_2)
        # embeddings = torch.cat((embeddings_1, embeddings_2), dim=-1)  # simply concat two embeddings
        # embeddings = self.sts_dropout(embeddings)
        # embeddings_1, embeddings_2 = torch.split(embeddings, embeddings_1.size()[1], dim=1)  # split back to two sentence embeddings
        # logits = torch.cosine_similarity(embeddings_1, embeddings_2)  # unnormalized range of (-1, 1)
        # scores = F.relu(logits * 5)  # map (-1, 1) to (0, 5)

        input_ids = torch.cat((input_ids_1, input_ids_2), dim=-1)
        attention_mask = torch.cat((attention_mask_1, attention_mask_2), dim=-1)
        embedddings = self.forward(input_ids, attention_mask)
        embedddings = self.sts_dropout(embedddings)
        logits = self.sts_linear(embedddings)
        scores = torch.sigmoid(logits) * 5

        return scores.squeeze()




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    # sst datasets and dataloaders
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.sst_batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.sst_batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    # para datasets and dataloaders
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.para_batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.para_batch_size,
                                    collate_fn=para_dev_data.collate_fn)
    
    # sts datasets and dataloaders
    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.sts_batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.sts_batch_size,
                                    collate_fn=sts_dev_data.collate_fn)
    
    # repeat a small chunck of cycled dataset to make training batch nums exactly equal
    equal_batch_size = max(len(sst_train_dataloader), len(para_train_dataloader), len(sts_train_dataloader))
    cycled_sst_train_dataloader = sst_train_dataloader
    cycled_para_train_dataloader = para_train_dataloader
    cycled_sts_train_dataloader = sts_train_dataloader
    if len(sst_train_dataloader) < equal_batch_size:
        cycled_sst_train_dataloader = cycle(sst_train_dataloader)
    if len(para_train_dataloader) < equal_batch_size:
        cycled_para_train_dataloader = cycle(para_train_dataloader)
    if len(sts_train_dataloader) < equal_batch_size:
        cycled_sts_train_dataloader = cycle(sts_train_dataloader)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    # optimizer = PCGrad(AdamW(model.parameters(), lr=lr))
    optimizer = PCGrad(AdamaxW(model.parameters(), lr=lr))  # switch to AdamaxW algo
    # optimizer = AdamaxW(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer.optimizer, gamma=args.gamma)  # parameterize later to provide options via CLI
    # scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    best_dev_metric = 0

    train_loss_lst = []
    valid_loss_lst = []

    para_train_acc_lst = []
    para_dev_acc_lst = []
    sst_train_acc_lst = []
    sst_dev_acc_lst = []
    sts_train_corr_lst = []
    sts_dev_corr_lst = []

    # Save training results
    prompt_output = []

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        sst_train_loss = 0
        para_train_loss = 0
        sts_train_loss = 0
        sst_num_batches = 0
        para_num_batches = 0
        sts_num_batches = 0

        # round robin over 3 tasks cyclically by zip
        # task run out of data will be cycled
        for sst_b, para_b, sts_b in tqdm(zip(cycled_sst_train_dataloader, cycled_para_train_dataloader, cycled_sts_train_dataloader), desc=f'epoch-{epoch}', disable=TQDM_DISABLE):

            # loss = torch.zeros([])
            # loss = loss.to(device)

            # sst task
            if sst_b:  # skip none batches
                sst_b_ids, sst_b_mask, sst_b_labels = (sst_b['token_ids'],
                                        sst_b['attention_mask'], sst_b['labels'])

                sst_b_ids = sst_b_ids.to(device)
                sst_b_mask = sst_b_mask.to(device)
                sst_b_labels = sst_b_labels.to(device)

                # optimizer.zero_grad()
                sst_probs = model.predict_sentiment(sst_b_ids, sst_b_mask)
                sst_loss = F.cross_entropy(sst_probs, sst_b_labels.view(-1))  # cross entropy as loss funtion

                # sst_loss.backward()
                # optimizer.step()

                sst_train_loss += sst_loss.item()
                sst_num_batches += 1


            # para task
            if para_b:
                para_b_ids_1, para_b_ids_2, para_b_mask_1, para_b_mask_2, para_b_labels = (
                    para_b['token_ids_1'], para_b['token_ids_2'],
                    para_b['attention_mask_1'], para_b['attention_mask_2'], para_b['labels'])

                para_b_ids_1 = para_b_ids_1.to(device)
                para_b_ids_2 = para_b_ids_2.to(device)
                para_b_mask_1 = para_b_mask_1.to(device)
                para_b_mask_2 = para_b_mask_2.to(device)
                para_b_labels = para_b_labels.to(device)

                # optimizer.zero_grad()
                para_probs = model.predict_paraphrase(para_b_ids_1, para_b_mask_1, para_b_ids_2, para_b_mask_2)
                para_loss = F.binary_cross_entropy(para_probs, para_b_labels.view(-1).float())  # binary cross entropy as loss function

                # para_loss.backward()
                # optimizer.step()

                para_train_loss += para_loss.item()
                para_num_batches += 1


            # sts task
            if sts_b:
                sts_b_ids_1, sts_b_ids_2, sts_b_mask_1, sts_b_mask_2, sts_b_scores = (
                    sts_b['token_ids_1'], sts_b['token_ids_2'],
                    sts_b['attention_mask_1'], sts_b['attention_mask_2'], sts_b['labels'])

                sts_b_ids_1 = sts_b_ids_1.to(device)
                sts_b_ids_2 = sts_b_ids_2.to(device)
                sts_b_mask_1 = sts_b_mask_1.to(device)
                sts_b_mask_2 = sts_b_mask_2.to(device)
                sts_b_scores = sts_b_scores.to(device)

                # optimizer.zero_grad()
                sts_scores = model.predict_similarity(sts_b_ids_1, sts_b_mask_1, sts_b_ids_2, sts_b_mask_2)
                sts_loss = F.mse_loss(sts_scores, sts_b_scores.view(-1))  # mean squared error as loss function

                # sts_loss.backward()
                # optimizer.step()

                sts_train_loss += sts_loss.item()
                sts_num_batches += 1

            optimizer.zero_grad()
            losses = [sst_loss, para_loss, sts_loss]
            optimizer.pc_backward(losses)
            optimizer.step()

        num_batches = sst_num_batches + para_num_batches + sts_num_batches
        train_loss = (sst_train_loss + para_train_loss + sts_train_loss) / num_batches
        sst_train_loss = sst_train_loss / sst_num_batches
        para_train_loss = para_train_loss / para_num_batches
        sts_train_loss = sts_train_loss / sts_num_batches
        train_loss_lst.append(train_loss)

        # no need to eval train set which is time-consuming
        para_train_acc, _, _, sst_train_acc, _, _, sts_train_corr, *_ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        average_train_metric = (para_train_acc + sst_train_acc + sts_train_corr) / 3
        para_train_acc_lst.append(para_train_acc)
        sst_train_acc_lst.append(sst_train_acc)
        sts_train_corr_lst.append(sts_train_corr)

        para_dev_acc, _, _, sst_dev_acc, _, _, sts_dev_corr, *_ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)
        average_dev_metric = (para_dev_acc + sst_dev_acc + sts_dev_corr) / 3
        para_dev_acc_lst.append(para_dev_acc)
        sst_dev_acc_lst.append(sst_dev_acc)
        sts_dev_corr_lst.append(sts_dev_corr)

        sst_dev_loss = 0
        para_dev_loss = 0
        sts_dev_loss = 0
        dev_sst_num_batches = 0
        dev_para_num_batches = 0
        dev_sts_num_batches = 0

        # calculate dev loss
        model.eval()
        for dev_sst_b, dev_para_b, dev_sts_b in tqdm(zip_longest(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader), desc=f'epoch-{epoch}', disable=TQDM_DISABLE):

            # sst task
            if dev_sst_b:  # skip none batches
                dev_sst_b_ids, dev_sst_b_mask, dev_sst_b_labels = (dev_sst_b['token_ids'],
                                        dev_sst_b['attention_mask'], dev_sst_b['labels'])

                dev_sst_b_ids = dev_sst_b_ids.to(device)
                dev_sst_b_mask = dev_sst_b_mask.to(device)
                dev_sst_b_labels = dev_sst_b_labels.to(device)

                dev_sst_probs = model.predict_sentiment(dev_sst_b_ids, dev_sst_b_mask)
                dev_sst_loss = F.cross_entropy(dev_sst_probs, dev_sst_b_labels.view(-1))  # cross entropy as loss funtion

                sst_dev_loss += dev_sst_loss.item()
                dev_sst_num_batches += 1


            # para task
            if dev_para_b:
                dev_para_b_ids_1, dev_para_b_ids_2, dev_para_b_mask_1, dev_para_b_mask_2, dev_para_b_labels = (
                    dev_para_b['token_ids_1'], dev_para_b['token_ids_2'],
                    dev_para_b['attention_mask_1'], dev_para_b['attention_mask_2'], dev_para_b['labels'])

                dev_para_b_ids_1 = dev_para_b_ids_1.to(device)
                dev_para_b_ids_2 = dev_para_b_ids_2.to(device)
                dev_para_b_mask_1 = dev_para_b_mask_1.to(device)
                dev_para_b_mask_2 = dev_para_b_mask_2.to(device)
                dev_para_b_labels = dev_para_b_labels.to(device)

                dev_para_probs = model.predict_paraphrase(dev_para_b_ids_1, dev_para_b_mask_1, dev_para_b_ids_2, dev_para_b_mask_2)
                dev_para_loss = F.binary_cross_entropy(dev_para_probs, dev_para_b_labels.view(-1).float())  # binary cross entropy as loss function

                para_dev_loss += dev_para_loss.item()
                dev_para_num_batches += 1


            # sts task
            if dev_sts_b:
                dev_sts_b_ids_1, dev_sts_b_ids_2, dev_sts_b_mask_1, dev_sts_b_mask_2, dev_sts_b_scores = (
                    sts_b['token_ids_1'], sts_b['token_ids_2'],
                    sts_b['attention_mask_1'], sts_b['attention_mask_2'], sts_b['labels'])

                dev_sts_b_ids_1 = dev_sts_b_ids_1.to(device)
                dev_sts_b_ids_2 = dev_sts_b_ids_2.to(device)
                dev_sts_b_mask_1 = dev_sts_b_mask_1.to(device)
                dev_sts_b_mask_2 = dev_sts_b_mask_2.to(device)
                dev_sts_b_scores = dev_sts_b_scores.to(device)

                dev_sts_scores = model.predict_similarity(dev_sts_b_ids_1, dev_sts_b_mask_1, dev_sts_b_ids_2, dev_sts_b_mask_2)
                dev_sts_loss = F.mse_loss(dev_sts_scores, dev_sts_b_scores.view(-1))  # mean squared error as loss function

                sts_dev_loss += dev_sts_loss.item()
                dev_sts_num_batches += 1

        dev_num_batches = dev_sst_num_batches + dev_para_num_batches + dev_sts_num_batches
        dev_loss = (sst_dev_loss + para_dev_loss + sts_dev_loss) / dev_num_batches
        sst_dev_loss = sst_dev_loss / dev_sst_num_batches
        para_dev_loss = para_dev_loss / dev_para_num_batches
        sts_dev_loss = sts_dev_loss / dev_sts_num_batches
        valid_loss_lst.append(dev_loss)

        # use average_train_metric as update threshold
        if average_dev_metric > best_dev_metric:
            best_dev_metric = average_dev_metric
            # save_model(model, optimizer.optimizer, args, config, args.filepath)
            save_model(model, optimizer.optimizer, args, config, args.filepath)

        scheduler.step()

        print(f"Epoch {epoch}: sst loss :: {sst_train_loss :.3f}, para loss :: {para_train_loss :.3f}, sts loss :: {sts_train_loss :.3f}")
        print(f"Epoch {epoch}: sst dev loss :: {sst_dev_loss :.3f}, para dev loss :: {para_dev_loss :.3f}, sts dev loss :: {sts_dev_loss :.3f}")
        print(f"Epoch {epoch}: avg train loss :: {train_loss :.3f}, avg dev loss :: {dev_loss :.3f}")
        print(f"Epoch {epoch}: avg train metric :: {average_train_metric :.3f}, avg dev metric :: {average_dev_metric :.3f}")
        # print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, avg dev metric :: {average_dev_metric :.3f}")

        prompt_output.append(f"Epoch {epoch}: sst train loss :: {sst_train_loss :.3f}, para train loss :: {para_train_loss :.3f}, sts train loss :: {sts_train_loss :.3f}\n")
        prompt_output.append(f"Epoch {epoch}: sst dev loss :: {sst_dev_loss :.3f}, para dev loss :: {para_dev_loss :.3f}, sts dev loss :: {sts_dev_loss :.3f}\n")
        prompt_output.append(f"Epoch {epoch}: avg train loss :: {train_loss :.3f}, avg dev loss :: {dev_loss :.3f}\n")
        prompt_output.append(f"Epoch {epoch}: sst train acc :: {sst_train_acc :.3f}, para train acc :: {para_train_acc :.3f}, sts train corr :: {sts_train_corr :.3f}\n")
        prompt_output.append(f"Epoch {epoch}: sst dev acc :: {sst_dev_acc :.3f}, para dev acc :: {para_dev_acc :.3f}, sts dev corr :: {sts_dev_corr :.3f}\n")
        prompt_output.append(f"Epoch {epoch}: avg train metric :: {average_train_metric :.3f}, avg dev metric :: {average_dev_metric :.3f}\n")


    def save_scores(prompt_output, filename):
        # use time as file name
        if filename is None:
            filename = "training result " + datetime.now().strftime("%H:%M:%S")
        with open(filename + '.txt', 'w') as f:
            f.writelines(prompt_output)

    # save output to filename
    save_scores(prompt_output, args.filename)

    # generate the loss figure
    # Edit the font, font size, and axes width
    mpl.rcParams['font.family'] = 'DejaVu Sans'  # font
    plt.rcParams['font.size'] = 18  # font size
    plt.rcParams['axes.linewidth'] = 2  # axes width
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_axes([0, 0, 1, 1])  # Add axes object to our figure that takes up entire figure
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
    ax.plot(train_loss_lst, linewidth=2, color='b', label="Train Loss", alpha=1)
    ax.plot(valid_loss_lst, linewidth=2, color='r', label="Dev Loss", alpha=1)
    ax.set_ylabel('Loss', labelpad=10, fontsize=20)
    ax.set_xlabel('Epochs', labelpad=10, fontsize=20)
    ax.grid(color='g', ls='-.', lw=0.5)
    plt.legend(loc="upper right", fontsize=20)
    plt.title("Multitask Classifier")
    plt.savefig('Loss_10epoch.png', dpi=300, transparent=False, bbox_inches='tight')

    # generate accuracy figure
    def plot_score(train_lst, dev_lst, legend_label, y_label, plt_title, fig_name):
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_axes([0, 0, 1, 1])  # Add axes object to our figure that takes up entire figure
        ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
        ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
        ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
        ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
        ax.plot(train_lst, linewidth=2, color='b', label="Train " + legend_label, alpha=1)
        ax.plot(dev_lst, linewidth=2, color='r', label="Dev " + legend_label, alpha=1)
        ax.set_ylabel(y_label, labelpad=10, fontsize=20)
        ax.set_xlabel('Epochs', labelpad=10, fontsize=20)
        ax.grid(color='g', ls='-.', lw=0.5)
        plt.legend(loc="lower right", fontsize=20)
        plt.title(plt_title)
        plt.savefig(fig_name + '.png', dpi=300, transparent=False, bbox_inches='tight')

    plot_score(para_train_acc_lst, para_dev_acc_lst, 'ACC', 'Accuracy', 'Multitask Classifier for Paraphrase Detection',
               'Acc_para_10epoch')
    plot_score(sst_train_acc_lst, sst_dev_acc_lst, 'ACC', 'Accuracy', 'Multitask Classifier for Sentiment Analysis',
               'Acc_sst_10epoch')
    plot_score(sts_train_corr_lst, sts_dev_corr_lst, 'CORR', 'Correlation', 'Multitask Classifier for Semantic Textual Similarity',
               'Corr_sts_10epoch')

def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="finetune")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    # use seperate batch sizes for different tasks since training sets have variant size
    # ideal batch sizes to make similar batch nums for 3 tasks should be 4 : 64 : 3
    # larger batch sizes are better but limited by GPU memory capability
    # current default sizes can fit in 24G GPU
    parser.add_argument("--sst_batch_size", help='fit with para batch size', type=int, default=2)  # Jerry edited
    parser.add_argument("--para_batch_size", help='48 can fit in 24G GPU', type=int, default=36)  # Jerry edited
    parser.add_argument("--sts_batch_size", help='fit with para batch size', type=int, default=2)  # Jerry edited
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-4)
    parser.add_argument("--gamma", type=float, help='learning rate decay factor, default gamma is 0.9', default=0.9)

    parser.add_argument("--filename", type=str, default=None)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
