import os
import time
import json
import shutil
import torch
import torch.nn.functional as F

from . import transforms

def train_one_epoch(model, optimizer, ds, res):
    """
    Trains any model which takes (image, rois) and
    outputs class_logits for one epoch.
    Expects a dataset.pdosp.PDOSP dataset.
    Uses cross-entropy loss.
    """
    model.train();
    device = next(model.parameters()).device
    loss_list = []
    label_match_list = []

    for image_batch, rois_batch, labels_batch in ds:
        # compute batch loss
        optimizer.zero_grad()
        for image, rois, labels in zip(image_batch, rois_batch, labels_batch):
            # load data
            image = image.to(device)
            rois = rois.to(device)
            labels = labels.to(device)

            # augment data
            image, rois = transforms.augment(image, rois)

            # preprocess image
            image = transforms.preprocess(image, res=res)
            
            # predict occupancy
            class_logits = model(image, rois)

            # compute loss
            loss = F.cross_entropy(class_logits, labels)
            loss.backward()
            loss_list += [loss.tolist()]
            
            # compute accuracy
            pred_lab = torch.argmax(class_logits, 1)
            label_match_list += (pred_lab == labels).tolist()
            
        # update weights
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
    
    # compute mean metrics
    mean_loss = float(torch.mean(torch.tensor(loss_list)))
    mean_accuracy = float(torch.mean(torch.tensor(label_match_list, dtype=torch.float32)))
    
    return mean_loss, mean_accuracy


@torch.no_grad()
def eval_one_epoch(model, ds, res):
    """
    Evaluates any model which takes (image, rois) and outputs class_logits.
    Expects a dataset.pdosp.PDOSP dataset.
    Outputs accuracy and cross-entropy loss.
    """
    model.eval();
    device = next(model.parameters()).device
    loss_list = []
    label_match_list = []
    
    for image_batch, rois_batch, labels_batch in ds:
        for image, rois, labels in zip(image_batch, rois_batch, labels_batch):
            # load data
            image = image.to(device)
            rois = rois.to(device)
            labels = labels.to(device)

            # preprocess image
            image = transforms.preprocess(image, res=res)

            # predict occupancy
            class_logits = model(image, rois)

            # compute loss
            loss = F.cross_entropy(class_logits, labels)
            loss_list += [float(loss)]
            
            # compute accuracy
            pred_lab = torch.argmax(class_logits, 1)
            label_match_list += (pred_lab == labels)
    
    # compute mean metrics
    mean_loss = float(torch.mean(torch.tensor(loss_list)))
    mean_accuracy = float(torch.mean(torch.tensor(label_match_list, dtype=torch.float32)))
    
    return mean_loss, mean_accuracy


def train_model(model, train_ds, valid_ds, test_ds, model_dir, device, lr=1e-4, epochs=100, lr_decay=50, res=None, verbose=False):
    """
    Trains any model which takes (image, rois) and outputs class_logits.
    Expects dataset.pdosp.PDOSP datasets.
    Uses cross-entropy loss.
    """
    # transfer model to device
    model = model.to(device)
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay, gamma=0.1)
    
    # train
    for epoch in range(1, epochs+1):
        # train for one epoch
        t0 = time.time()
        train_loss, train_accuracy = train_one_epoch(model, optimizer, train_ds, res)
        scheduler.step()
        
        # evaluate on the valid dataset
        valid_loss, valid_accuracy = eval_one_epoch(model, valid_ds, res)
        
        # print progess
        if verbose:
            print(f'epoch {epoch:3} -- train acc: {train_accuracy:.4f} -- valid acc.: {valid_accuracy:.4f} -- {time.time()-t0:.0f} sec')
        
        # if this is the first epoch
        if epoch == 1:
            # ensure (an empty) model dir exists
            shutil.rmtree(model_dir, ignore_errors=True)
            os.makedirs(model_dir, exist_ok=False)

            # create log header
            with open(f'{model_dir}/train_log.csv', 'w', newline='\n', encoding='utf-8') as f:
                f.write('train_loss,train_accuracy,valid_loss,valid_accuracy\n')

        # save epoch logs
        with open(f'{model_dir}/train_log.csv', 'a', newline='\n', encoding='utf-8') as f:
            f.write(f'{train_loss:.4f},{train_accuracy:.4f},{valid_loss:.4f},{valid_accuracy:.4f}\n')

        # save weights
        torch.save(model.state_dict(), f'{model_dir}/weights_last_epoch.pt')
        
    # test model on test dataset
    test_loss, test_accuracy = eval_one_epoch(model, test_ds, res)
    with open(f'{model_dir}/test_logs.json', 'w') as f:
        json.dump({'loss': test_loss, 'accuracy': test_accuracy}, f)

    # delete model from memory
    del model

