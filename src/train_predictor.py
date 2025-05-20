import os

import pandas as pd
import numpy as np
import pickle

import torch
import torch.nn as nn

from model import *
from utils import set_seed, Logger, EarlyStopper


def get_fold_dataloader(total_dataset, indices, batch_size=512, seed=42):
    train_idx, valid_idx, test_idx = indices

    train_dataset = torch.utils.data.Subset(total_dataset, train_idx)
    valid_dataset = torch.utils.data.Subset(total_dataset, valid_idx)
    test_dataset = torch.utils.data.Subset(total_dataset, test_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, valid_dataloader, test_dataloader


def train_predictor(args):

    # parameters
    fold = 10
    batch_zise = 512
    hidden_dim1 = 256
    hidden_dim2 = 128
    output_dim = 1
    lr = 1e-3
    set_seed(args.seed)

    model_names = [f'THERAPI_predictor_CV{i}' for i in range(fold)]

    # load data
    fold_indices = []
    for cv in range(fold):
        cv_dir = args.data_dir + f'GDSC_split/fold_{cv}_indices.pkl'
        with open(cv_dir, 'rb') as f:
            train_idx, valid_idx, test_idx = pickle.load(f)
        fold_indices.append((train_idx, valid_idx, test_idx))

    gdsc_resp_dir = os.path.join(args.data_dir, 'GDSC_resp.csv')
    gdsc_rank_dir = os.path.join(args.data_dir, 'GDSC_rankrepresentation.csv')
    gdsc_pert_dir = os.path.join(args.data_dir, 'GDSC_perturbation.npy')
    gdsc_comp_dir = os.path.join(args.data_dir, 'GDSC_perturbation_compound.npy')

    gdsc_resp_df = pd.read_csv(gdsc_resp_dir)
    gdsc_rank_df = pd.read_csv(gdsc_rank_dir, index_col=0).loc[gdsc_resp_df['Cell_ID']].values
    gdsc_pert_np = np.load(gdsc_pert_dir)
    gdsc_comp_np = np.load(gdsc_comp_dir)

    gdsc_label_col = 'Label'
    gdsc_dataset = ExpDrugDataset(gdsc_pert_np, gdsc_rank_df, gdsc_comp_np, gdsc_resp_df[gdsc_label_col].values)
    gdsc_fold_dataloder = [get_fold_dataloader(gdsc_dataset, indices, batch_size=batch_zise, seed=args.seed) for indices in fold_indices]

    if not os.path.exists('ckpts'):
        os.makedirs('ckpts', exist_ok=True)

    # training
    for i, cv in enumerate(range(fold)):
        model_name = model_names[i]
        logger = Logger(model_name)
        logger('Start training {} model'.format(model_name))
        
        train_dataloader, valid_dataloader, _ = gdsc_fold_dataloder[i]

        model = Response_predictor(gdsc_dataset.emb_dim, gdsc_dataset.genef_dim, gdsc_dataset.chemical_dim, hidden_dim1, hidden_dim2, output_dim).to(args.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        earlystopper = EarlyStopper(patience=10, path=f'ckpts/{model_name}.pt', verbose=True)

        epoch = 0
        while True:
            epoch += 1
            model.train()
            train_loss = 0
            for emb, genef, chemical, resp in train_dataloader:
                
                emb, genef, chemical, resp = emb.to(args.device), genef.to(args.device), chemical.to(args.device), resp.to(args.device)

                optimizer.zero_grad()
                output = model(emb, genef, chemical)
                loss = criterion(output.squeeze(), resp)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_dataloader)

            model.eval()
            with torch.no_grad():
                valid_preds = []
                valid_targets = []
                for emb, genef, chemical, resp in valid_dataloader:
                    emb, genef, chemical = emb.to(args.device), genef.to(args.device), chemical.to(args.device)
                    output = model(emb, genef, chemical)
                    valid_preds.append(output.cpu())
                    valid_targets.append(resp)
                valid_preds = torch.cat(valid_preds)
                valid_targets = torch.cat(valid_targets)

            valid_loss = criterion(valid_preds.squeeze(), valid_targets).item()

            logger(f'Epoch {epoch}, train loss: {train_loss:.4f}, valid loss: {valid_loss:.4f}')

            if earlystopper(valid_loss, model):
                break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train the predictor')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--data_dir', type=str, default='./data/')
    args = parser.parse_args()

    train_predictor(args)