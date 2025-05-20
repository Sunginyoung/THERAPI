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
    gdsc_data_dir = os.path.join(args.data_dir, 'GDSC_gex.csv')
    gdsc_info_dir = os.path.join(args.data_dir, 'GDSC_info.csv')

    tcga_unlabeled_data_dir = os.path.join(args.data_dir, 'TCGA_unlabeled_gex.csv')
    tcga_unlabeled_info_dir = os.path.join(args.data_dir, 'TCGA_unlabeled_info.csv')

    gdsc_data_df = pd.read_csv(gdsc_data_dir, index_col=0)
    gdsc_info_df = pd.read_csv(gdsc_info_dir)
    num_tissue = len(gdsc_info_df['tissue_label'].unique())

    tcga_unlabeled_data_df = pd.read_csv(tcga_unlabeled_data_dir, index_col=0)
    tcga_unlabeled_info_df = pd.read_csv(tcga_unlabeled_info_dir)

    gdsc_dataset = AlignerDataset(gdsc_data_df, 'gdsc', gdsc_info_df['tissue_label'])
    tcga_unlabeled_dataset = AlignerDataset(tcga_unlabeled_data_df, 'tcga', tcga_unlabeled_info_df['tissue_label'])
    tcga_unlabeled_dataloader = DataLoader(tcga_unlabeled_dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator = torch.Generator().manual_seed(args.seed))

    # model
    gdsc_AE = GDSC_AE(n_genes=gdsc_dataset.n_genes, n_classes=num_tissue, n_latent=dim_latent)
    tcga_weightencoder = TCGA_weightencoder(n_genes=tcga_unlabeled_dataset.n_genes, n_latent=dim_latent, n_celines=gdsc_data_df.shape[0])
    emb_dis_classifier = Emb_Dis_classifier(n_latent=dim_latent, n_classes=num_tissue)
    exp_dis_classifier = Exp_Dis_classifier(n_genes=tcga_unlabeled_dataset.n_genes, n_latent=dim_latent, n_classes=num_tissue)
    gdsc_AE.to(args.device)
    tcga_weightencoder.to(args.device)
    emb_dis_classifier.to(args.device)
    exp_dis_classifier.to(args.device)

    autoencoder_criterion = nn.MSELoss()
    center_criterion = CenterLoss(num_classes=num_tissue, feat_dim=dim_latent, device=args.device)
    classifier_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(gdsc_AE.parameters())+list(tcga_weightencoder.parameters())+list(emb_dis_classifier.parameters())+list(exp_dis_classifier.parameters()), lr=lr)

    if not os.path.exists('ckpts'):
        os.makedirs('ckpts', exist_ok=True)

    # training
    for epoch in range(199):
        gdsc_AE.train()
        tcga_weightencoder.train()
        emb_dis_classifier.train()
        exp_dis_classifier.train()

        train_losses = 0
        g_losses = 0
        t_losses = 0
        for tcga_gex, _, tcga_dis_label in tcga_unlabeled_dataloader:
            tcga_gex = tcga_gex.to(args.device)
            tcga_dis_label = tcga_dis_label.to(args.device)

            gdsc_gex = gdsc_dataset.data.to(args.device)
            gdsc_dis_label = gdsc_dataset.dis_label.to(args.device)
            gdsc_z, gdsc_recon = gdsc_AE(gdsc_gex)

            gdsc_emb_dis_pred = emb_dis_classifier(gdsc_z)
            gdsc_exp_dis_pred = exp_dis_classifier(gdsc_recon)
            
            # GDSC loss
            Grecon_loss = autoencoder_criterion(gdsc_recon, gdsc_gex)
            Gcenter_loss = center_criterion(gdsc_z, gdsc_dis_label)
            Gclass_loss_emb = classifier_criterion(gdsc_emb_dis_pred, gdsc_dis_label)
            Gclass_loss_exp = classifier_criterion(gdsc_exp_dis_pred, gdsc_dis_label)
            G_losses = loss_a*Grecon_loss + loss_b*Gcenter_loss + loss_c*(Gclass_loss_emb + Gclass_loss_exp) 

            # TCGA loss
            tcga_weights, tcga_latent, tcga_wgex, tcga_recon = tcga_weightencoder(tcga_gex, gdsc_z, gdsc_gex)
            tcga_emb_dis_pred = emb_dis_classifier(tcga_latent)
            tcga_exp_dis_pred = exp_dis_classifier(tcga_wgex)

            Trecon_loss = autoencoder_criterion(tcga_recon, tcga_gex)        
            Tcenter_loss = center_criterion(tcga_latent, tcga_dis_label)
            Tclass_loss_emb = classifier_criterion(tcga_emb_dis_pred, tcga_dis_label)
            Tclass_loss_exp = classifier_criterion(tcga_exp_dis_pred, tcga_dis_label)
            T_losses = loss_a*Trecon_loss + loss_b*Tcenter_loss + loss_c*(Tclass_loss_emb + Tclass_loss_exp)

            # update
            optimizer.zero_grad()
            total_losses = G_losses + T_losses
            total_losses.backward()
            optimizer.step()

            train_losses += total_losses.item()
            g_losses += G_losses.item()
            t_losses += T_losses.item()

        train_losses /= len(tcga_unlabeled_dataloader)
        g_losses /= len(tcga_unlabeled_dataloader)
        t_losses /= len(tcga_unlabeled_dataloader)
        logger(f'Epoch {epoch+1}, Train loss {train_losses:.4f}, G_losses {G_losses:.4f}, T_losses {T_losses:.4f}')

    # save model
    torch.save({'epoch': epoch,
                'gdsc_AE': gdsc_AE.state_dict(),
                'tcga_weightencoder': tcga_weightencoder.state_dict(),
                'emb_dis_classifier': emb_dis_classifier.state_dict(),
                'exp_dis_classifier': exp_dis_classifier.state_dict(),
                'optimizer': optimizer.state_dict()
                }, f'ckpts/{model_name}.pt')
   
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--data_dir', type=str, default='./data/')
    args = parser.parse_args()

    train_aligner(args)
