import os
import shutil
import sys
import torch
import yaml
import numpy as np
from datetime import datetime

import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.nt_xent import NTXentLoss

import wandb

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./configs/config_pretrain_clip.yaml', os.path.join(model_checkpoints_folder, 'config_pretrain_clip.yaml'))


class Trainer(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

    
        
        dir_name = "llmclip_[{}]_[{}]_[{}]_[{}]_wo_rdkit".format(self.config['dataset_name'],self.config['model_type'],self.config['load_model'], self.config['llm'])
        self.log_dir = os.path.join('ckpt', dir_name)
        # self.writer = SummaryWriter(log_dir=log_dir)
        self.use_wandb = False
        self.dataset = dataset

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    # def _step(self, model, graph, text):
    #     graph_features, text_features = model(graph, text)

    #     # logits.shape=[Batch,Batch]
    #     logits = (text_features @ graph_features.T)
    #     graphs_similarity = graph_features @ graph_features.T
    #     texts_similarity = text_features @ text_features.T
        
    #     logits = F.softmax(logits, dim=-1)
    #     targets = F.softmax(
    #         (graphs_similarity + texts_similarity) / 2, dim=-1
    #     )
    #     loss = (-targets * logits).sum(1)  # shape: (batch_size)

    #     return loss.mean()

    def _step(self, model, graph, text):
        graph_features, text_features, logit_scale = model(graph, text)

        graph_features = F.normalize(graph_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logits_per_graph = logit_scale * graph_features @ text_features.T   # [B,B] dim=1维度里同一个graph
        logits_per_text = logit_scale * text_features @ graph_features.T    # [B,B] dim=1维度里同一个text

        device = graph_features.device

        labels = torch.arange(logits_per_graph.shape[0], device=device, dtype=torch.long)
        total_loss = (
            F.cross_entropy(logits_per_graph, labels) + #cross entropy中softmax是对dim=1做的，因此infonce的分子是graph
            F.cross_entropy(logits_per_text, labels)    #infonce的分子是text
        ) / 2

        # soft_out = F.softmax(logits_per_graph,dim=1)#给每个样本的pred向量做指数归一化---softmax
        # log_soft_out = torch.log(soft_out)#将上面得到的归一化的向量再point-wise取对数
        # loss = F.nll_loss(log_soft_out, labels)#将归一化且取对数后的张量根据标签求和，实际就是计算loss的过程

        return total_loss

    def train(self):
        if self.use_wandb:
            wandb.init(project="llmclip", config=self.config, name="pretrain_llmclip_[{}]_[{}]_[{}]_[{}]".format(self.config['dataset_name'],self.config['model_type'],self.config['load_model'], self.config['llm']))


        train_loader, valid_loader = self.dataset.get_data_loaders()


        # if self.config['model_type'] == 'gin':
        #     from models.ginet_molclr import GINet
        #     model = GINet(**self.config["model"]).to(self.device)
        #     model = self._load_pre_trained_weights(model)
        # elif self.config['model_type'] == 'gcn':
        #     from models.gcn_molclr import GCN
        #     model = GCN(**self.config["model"]).to(self.device)
        #     model = self._load_pre_trained_weights(model)
        # else:
        #     raise ValueError('Undefined GNN model.')
        from models.CLIP import GraphCLIP
        
        model = GraphCLIP().to(self.device)

        model = self._load_pre_trained_weights(model)

        # layer_list = []
        # for name, param in model.named_parameters():
        #     if 'mlp.fc' in name:
        #         print(name, param.requires_grad)
        #         layer_list.append(name)

        
        # text_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        # graph_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        # optimizer = torch.optim.Adam(
        #     [{'params': text_params, 'lr': self.config['text_init_lr']}, {'params': graph_params}],
        #     self.config['graph_init_lr'], weight_decay=eval(self.config['weight_decay'])
        # )
        
        optimizer = torch.optim.Adam(
            model.parameters(), self.config['init_lr'], 
            weight_decay=eval(self.config['weight_decay'])
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.config['epochs']-self.config['warm_up'], 
            eta_min=0, last_epoch=-1
        )

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.log_dir, 'checkpoints')


        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        

        for epoch_counter in range(self.config['epochs']):

            total_loss = 0
            for bn, (graph, text) in enumerate(train_loader):
                optimizer.zero_grad()

                graph = graph.to(self.device)
                text = text.to(self.device)

                #print(graph.shape)
                # print(text.shape)

                loss = self._step(model, graph, text)

                total_loss += loss.item()

                # if n_iter % self.config['log_every_n_steps'] == 0:
                #     self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                #     self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    #print(epoch_counter, bn, loss.item())

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            total_loss = total_loss/len(train_loader)

            
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                print('Epochs:{} | Valid Loss:{:.4f}'.format(epoch_counter, valid_loss))

                if self.use_wandb:
                    wandb.log({"train_loss": total_loss, "valid_loss":valid_loss, "lr": scheduler.get_last_lr()[0]})

                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
            
                # self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                # valid_n_iter += 1
            
            if (epoch_counter+1) % self.config['save_every_n_epochs'] == 0:
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(epoch_counter))))

            # warmup for the first few epochs
            if epoch_counter >= self.config['warm_up']:
                scheduler.step()
                print()
        
        if self.use_wandb:
            wandb.finish()

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['load_model'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict,strict=False)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (graph, text) in valid_loader:
                graph = graph.to(self.device)
                text = text.to(self.device)

                loss = self._step(model, graph, text)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        
        model.train()
        return valid_loss


def main():
    config = yaml.load(open("./configs/config_pretrain_clip.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    from dataset.dataset_clip import MoleculeDatasetWrapper


    dataset = MoleculeDatasetWrapper(config['batch_size'], config['dataset_name'], config['llm'], **config['dataset'])
    trainer = Trainer(dataset, config)
    trainer.train()


if __name__ == "__main__":
    main()
