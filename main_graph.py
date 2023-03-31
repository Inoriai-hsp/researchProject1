import logging
from tqdm import tqdm
import numpy as np
import time

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl.dataloading import GraphDataLoader

import torch
from torch.utils.data import random_split

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from graphormer.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphormer.data.data_utils import getCallGraphsForFinetune, getGraphs, getCallGraphs
from graphormer.models import build_model, build_classify_model

def pretrain(model, train_loader, optimizer, max_epoch, device, scheduler, logger=None):
    epoch_iter = tqdm(iterable = range(max_epoch), position=0)
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for batch in train_loader:
            batch_g = batch
            batch_g = batch_g.to(device)

            feat = batch_g.ndata["tac_op"]
            model.train()
            loss, loss_dict = model(batch_g, feat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            if logger is not None:
                loss_dict["lr"] = get_current_lr(optimizer)
                logger.note(loss_dict, step=epoch)
        if scheduler is not None:
            scheduler.step()
        # epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")
        print(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")

    return model

            
def collate_fn(batch):
    batch_g = dgl.batch(batch)
    return batch_g

def callGraphsFn(batch):
    graphs = [x[0].add_self_loop() for x in batch]
    labels = [x[1] for x in batch]
    graphs = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0)
    return graphs, None, labels

def callGraphsFnForFinetune(batch):
    call_graphs = []
    labels = []
    batch_pdgs = []
    for (address, call_graph, pdgs, label) in batch:
        call_graphs.append(call_graph.add_self_loop())
        batch_pdgs.extend(pdgs)
        labels.append(label)
    call_graphs = dgl.batch(call_graphs)
    batch_pdgs = dgl.batch(batch_pdgs)
    labels = torch.cat(labels, dim=0)
    return call_graphs, batch_pdgs, labels

def classify(model, dataloader, optimizer, loss_fn, device):
    (train_loader, test_loader) = dataloader
    for epoch in range(200):
        model.train()
        loss_list = []
        for batch, batch_pdgs, labels in train_loader:
            batch_g = batch.to(device)
            if batch_pdgs is not None:
                batch_pdgs = batch_pdgs.to(device)
            y = labels.to(device)

            # feat = batch_g.ndata['x']
            model.train()
            out = model(batch_g, batch_pdgs)
            loss = loss_fn(out, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
        print(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")
        train_acc, train_f1, train_pre, train_rec = test(model, train_loader, device)
        print(f"train_acc: {train_acc:.4f}, train_f1: {train_f1:.4f}, train_pre: {train_pre:.4f}, train_rec: {train_rec:.4f}")
        test_acc, test_f1, test_pre, test_rec = test(model, test_loader, device)
        print(f"test_acc: {test_acc:.4f}, test_f1: {test_f1:.4f}, test_pre: {test_pre:.4f}, test_rec: {test_rec:.4f}")

def test(model, dataloader, device):
    y_pred = torch.tensor([], dtype=torch.int64)
    y_true = torch.tensor([], dtype=torch.int64)
    model.eval()
    for batch, batch_pdgs, labels in dataloader:
        batch_g = batch.to(device)
        if batch_pdgs is not None:
            batch_pdgs = batch_pdgs.to(device)
        out = model(batch_g, batch_pdgs)
        out = torch.softmax(out, dim=1)
        pred = out.argmax(dim=1)
        y_pred = torch.cat([y_pred, pred.cpu()])
        y_true = torch.cat([y_true, labels.cpu()])
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro')
    return accuracy, f1, precision, recall

def main(args):
    # device = args.device if args.device >= 0 else "cpu"
    device = 1
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler
    pooling = args.pooling
    deg4feat = args.deg4feat
    batch_size = args.batch_size
    args.num_features = num_hidden
    num_classes = 2

    pretrain_graphs, graphs = getGraphs(sample_number=None)
    print(len(pretrain_graphs))
    train_loader = GraphDataLoader(pretrain_graphs, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)

    if pooling == "mean":
        pooler = AvgPooling()
    elif pooling == "max":
        pooler = MaxPooling()
    elif pooling == "sum":
        pooler = SumPooling()
    else:
        raise NotImplementedError


    set_random_seed(1234)

    if logs:
        logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
    else:
        logger = None

    model = build_model(args)
        
    if not load_model:
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
        model = pretrain(model, train_loader, optimizer, max_epoch, device, scheduler, logger)
        model = model.cpu()

    if load_model:
        logging.info("Loading Model ... ")
        model.load_state_dict(torch.load("checkpoint.pt"))
    if save_model:
        logging.info("Saveing Model ...")
        torch.save(model.state_dict(), "checkpoint.pt")
    
    if args.finetune:
        model.cpu()
        call_graphs = getCallGraphsForFinetune(graphs)
        # call_graphs = getCallGraphs(model, pooler, graphs, device)
        train_split = int(len(call_graphs) * 0.8)
        test_split = len(call_graphs) - train_split
        train_graphs, test_graphs = random_split(call_graphs, [train_split, test_split], generator=torch.Generator().manual_seed(42))
        train_loader = GraphDataLoader(train_graphs, collate_fn=callGraphsFnForFinetune, batch_size=batch_size, shuffle=True)
        test_loader = GraphDataLoader(test_graphs, collate_fn=callGraphsFnForFinetune, batch_size=batch_size)
        classify_model = build_classify_model(256, 256, 256, 2, model.embedding, model.encoder, pooler, args.finetune)
        classify_model.to(device)
        classify_optimizer = create_optimizer(optim_type, classify_model, 0.0001, weight_decay)
        classify_loss_fn = torch.nn.CrossEntropyLoss()
        classify_loss_fn.to(device)
        classify(classify_model, (train_loader, test_loader), classify_optimizer, classify_loss_fn, device)
    else:
        model.to(device)
        call_graphs = getCallGraphs(model, pooler, graphs, device)
        torch.cuda.empty_cache()
        train_split = int(len(call_graphs) * 0.8)
        test_split = len(call_graphs) - train_split
        train_graphs, test_graphs = random_split(call_graphs, [train_split, test_split], generator=torch.Generator().manual_seed(42))
        train_loader = GraphDataLoader(train_graphs, collate_fn=callGraphsFn, batch_size=batch_size, shuffle=True)
        test_loader = GraphDataLoader(test_graphs, collate_fn=callGraphsFn, batch_size=batch_size)
        classify_model = build_classify_model(256, 256, 256, 2, None, None, pooler, args.finetune)
        classify_model.to(device)
        classify_optimizer = create_optimizer(optim_type, classify_model, 0.0001, weight_decay)
        classify_loss_fn = torch.nn.CrossEntropyLoss()
        classify_loss_fn.to(device)
        classify(classify_model, (train_loader, test_loader), classify_optimizer, classify_loss_fn, device)


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    args.encoder = 'gin'
    args.decoder = 'mlp'
    args.device = 1
    args.batch_size = 128
    args.lr = 0.001
    args.save_model = False
    args.load_model = True
    args.pooling = "sum"
    args.finetune = False
    print(args)
    main(args)
