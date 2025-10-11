import os

import pandas as pd

import torch
import torch.nn as nn

from model import *
from utils import set_seed, Logger
from center_loss import CenterLoss


def train_aligner(args):

    model_name = 'THERAPI_aligner'
    logger = Logger(model_name)
    logger('Start training {} model'.format(model_name))
    set_seed(args.seed, logger)

    # parameters
    batch_size = 128
    dim_latent = 128
    lr = 1e-3
    loss_a = 0.2
    loss_b = 0.8
    loss_c = 0.4

    # load data
    source_data_dir = os.path.join(args.data_dir, f'{args.source}/{args.source}_gex.csv')
    source_info_dir = os.path.join(args.data_dir, f'{args.source}/{args.source}_info.csv')

    target_unlabeled_data_dir = os.path.join(args.data_dir, f'{args.target}/{args.target}_unlabeled_gex.csv')
    target_unlabeled_info_dir = os.path.join(args.data_dir, f'{args.target}/{args.target}_unlabeled_info.csv')

    source_data_df = pd.read_csv(source_data_dir, index_col=0)
    source_info_df = pd.read_csv(source_info_dir)
    num_tissue = len(source_info_df['tissue_label'].unique())

    target_unlabeled_data_df = pd.read_csv(target_unlabeled_data_dir, index_col=0)
    target_unlabeled_info_df = pd.read_csv(target_unlabeled_info_dir)

    source_dataset = AlignerDataset(source_data_df, args.source, source_info_df['tissue_label'])
    target_unlabeled_dataset = AlignerDataset(target_unlabeled_data_df, args.target, target_unlabeled_info_df['tissue_label'])
    target_unlabeled_dataloader = DataLoader(target_unlabeled_dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator = torch.Generator().manual_seed(args.seed))

    # model
    source_AE = SOURCE_AE(n_genes=source_dataset.n_genes, n_classes=num_tissue, n_latent=dim_latent)
    target_weightencoder = TARGET_weightencoder(n_genes=target_unlabeled_dataset.n_genes, n_latent=dim_latent, n_celines=source_data_df.shape[0])
    emb_dis_classifier = Emb_Dis_classifier(n_latent=dim_latent, n_classes=num_tissue)
    exp_dis_classifier = Exp_Dis_classifier(n_genes=target_unlabeled_dataset.n_genes, n_latent=dim_latent, n_classes=num_tissue)
    source_AE.to(args.device)
    target_weightencoder.to(args.device)
    emb_dis_classifier.to(args.device)
    exp_dis_classifier.to(args.device)

    autoencoder_criterion = nn.MSELoss()
    center_criterion = CenterLoss(num_classes=num_tissue, feat_dim=dim_latent, device=args.device)
    classifier_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(source_AE.parameters())+list(target_weightencoder.parameters())+list(emb_dis_classifier.parameters())+list(exp_dis_classifier.parameters()), lr=lr)

    if not os.path.exists('ckpts'):
        os.makedirs('ckpts', exist_ok=True)

    # training
    for epoch in range(199):
        source_AE.train()
        target_weightencoder.train()
        emb_dis_classifier.train()
        exp_dis_classifier.train()

        train_losses = 0
        g_losses = 0
        t_losses = 0
        for target_gex, _, target_dis_label in target_unlabeled_dataloader:
            target_gex = target_gex.to(args.device)
            target_dis_label = target_dis_label.to(args.device)

            source_gex = source_dataset.data.to(args.device)
            source_dis_label = source_dataset.dis_label.to(args.device)
            source_z, source_recon = source_AE(source_gex)

            source_emb_dis_pred = emb_dis_classifier(source_z)
            source_exp_dis_pred = exp_dis_classifier(source_recon)
            
            # source loss
            Grecon_loss = autoencoder_criterion(source_recon, source_gex)
            Gcenter_loss = center_criterion(source_z, source_dis_label)
            Gclass_loss_emb = classifier_criterion(source_emb_dis_pred, source_dis_label)
            Gclass_loss_exp = classifier_criterion(source_exp_dis_pred, source_dis_label)
            G_losses = loss_a*Grecon_loss + loss_b*Gcenter_loss + loss_c*(Gclass_loss_emb + Gclass_loss_exp) 

            # target loss
            target_weights, target_latent, target_wgex, target_recon = target_weightencoder(target_gex, source_z, source_gex)
            target_emb_dis_pred = emb_dis_classifier(target_latent)
            target_exp_dis_pred = exp_dis_classifier(target_wgex)

            Trecon_loss = autoencoder_criterion(target_recon, target_gex)        
            Tcenter_loss = center_criterion(target_latent, target_dis_label)
            Tclass_loss_emb = classifier_criterion(target_emb_dis_pred, target_dis_label)
            Tclass_loss_exp = classifier_criterion(target_exp_dis_pred, target_dis_label)
            T_losses = loss_a*Trecon_loss + loss_b*Tcenter_loss + loss_c*(Tclass_loss_emb + Tclass_loss_exp)

            # update
            optimizer.zero_grad()
            total_losses = G_losses + T_losses
            total_losses.backward()
            optimizer.step()

            train_losses += total_losses.item()
            g_losses += G_losses.item()
            t_losses += T_losses.item()

        train_losses /= len(target_unlabeled_dataloader)
        g_losses /= len(target_unlabeled_dataloader)
        t_losses /= len(target_unlabeled_dataloader)
        logger(f'Epoch {epoch+1}, Train loss {train_losses:.4f}, G_losses {G_losses:.4f}, T_losses {T_losses:.4f}')

    # save model
    torch.save({'epoch': epoch,
                'source_AE': source_AE.state_dict(),
                'target_weightencoder': target_weightencoder.state_dict(),
                'emb_dis_classifier': emb_dis_classifier.state_dict(),
                'exp_dis_classifier': exp_dis_classifier.state_dict(),
                'optimizer': optimizer.state_dict()
                }, f'ckpts/{model_name}.pt')
   
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--source', type=str, default='GDSC')
    parser.add_argument('--target', type=str, default='TCGA')
    

    args = parser.parse_args()

    train_aligner(args)
