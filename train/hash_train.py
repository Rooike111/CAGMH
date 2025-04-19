from torch.nn.modules import loss
from model.hash_model import CAGMH as CAGMH
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.io as scio
from prefetch_generator import BackgroundGenerator
from .base import TrainBase
from model.optimization import BertAdam
from utils import get_args, calc_neighbor, cosine_similarity, euclidean_similarity
from utils.calc_utils import calc_map_k_matrix as calc_map_k
from utils.utils import multimodal_proxy_loss
from dataset.dataloader import dataloader
from utils.Cross_modal_class_balance_loss import Cross_modal_class_balance_loss
import time


import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import scipy.io as scio
from PIL import Image
from torchvision import transforms


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Trainer(TrainBase):
    def __init__(self,rank=1):
        args = get_args()
        super(Trainer, self).__init__(args, args.device)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()


    def _init_model(self):
        self.logger.info("init model.")
        self.logger.info("ViT+GPT!")
        HashModel = CAGMH
        self.model = HashModel(outputDim=self.args.output_dim, clipPath=self.args.clip_path,
                            writer=self.writer, logger=self.logger, is_train=self.args.is_train,device=self.args.device).to(self.rank)

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))
        self.model.float()
        self.optimizer = BertAdam([
                    {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
                    {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
                    {'params': self.model.text_hash.parameters(), 'lr': self.args.lr},

                    ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine', 
                    b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
                    weight_decay=self.args.weight_decay, max_grad_norm=1.0)
        self.mpl = multimodal_proxy_loss().to(self.rank)
        self.cmcbloss = Cross_modal_class_balance_loss(self.args,self.args.output_dim).to(self.rank)
        self.optimizer_loss = torch.optim.SGD(params=self.mpl.parameters(), lr=0.02, momentum=0.9, weight_decay=0.0005)
        self.csq_loss = torch.optim.SGD(params=self.cmcbloss.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)#lr 0.00005
        self.total_time = 0


    def _init_dataset(self):
        self.logger.info("init dataset.")
        self.logger.info(f"Using {self.args.dataset} dataset.")
        self.args.index_file = os.path.join("./dataset", self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join("./dataset", self.args.dataset, self.args.caption_file)
        self.args.label_file = os.path.join("./dataset", self.args.dataset, self.args.label_file)
        train_data, query_data, retrieval_data = dataloader(captionFile=self.args.caption_file, 
                                        indexFile=self.args.index_file, 
                                        labelFile=self.args.label_file, 
                                        maxWords=self.args.max_words,
                                        imageResolution=self.args.resolution,
                                        query_num=self.args.query_num,
                                        train_num=self.args.train_num,
                                        seed=self.args.seed)
        self.train_labels = train_data.get_all_label().to(self.args.device)
        
        self.query_labels = query_data.get_all_label()
        self.retrieval_labels = retrieval_data.get_all_label()
        self.args.retrieval_num = len(self.retrieval_labels)
        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")
        self.train_loader = DataLoaderX(
                dataset=train_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True
            )
        self.query_loader = DataLoaderX(
                dataset=query_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True
            )
        self.retrieval_loader = DataLoaderX(
                dataset=retrieval_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True
            )


    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d"%(epoch, self.args.epochs))
        all_loss = 0

        for image, text, label, index in self.train_loader:

            start_time = time.time()
            self.global_step += 1
            image.float()

            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True)
            index = index.numpy()
            hash_img, hash_text = self.model(image, text)
            loss = self.mpl(hash_img, hash_text, label)
            auxiliaryLoss_img = self.cmcbloss(hash_img,label,self.args)
            auxiliaryLoss_txt = self.cmcbloss(hash_text,label,self.args)
            loss = loss+auxiliaryLoss_img[0]+auxiliaryLoss_txt[0]
            all_loss += loss 
            self.optimizer.zero_grad()
            self.csq_loss.zero_grad()
            self.optimizer_loss.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_loss.step()
            self.csq_loss.step()

            t = time.time()
            self.total_time += time.time() - start_time


        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}, time: {self.total_time}")

    def train(self):
        self.logger.info("Start train.")
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            self.valid(epoch)
        self.logger.info(f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")


    def get_code(self, data_loader, length: int):
        img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        encoder_time = 0
        for image, text, label, index in tqdm(data_loader):
            start_encoder_time = time.time()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            index = index.numpy()
            image_hash ,text_hash = self.model.encoding(image,text)
            image_hash = torch.sign(image_hash)
            text_hash = torch.sign(text_hash)
            encoder_time = time.time() - start_encoder_time
            img_buffer[index, :] = image_hash.data
            text_buffer[index, :] = text_hash.data
        
        return img_buffer, text_buffer, encoder_time


    def test(self, mode_name="i2t"):
        if self.args.pretrained == "":
            raise RuntimeError("test step must load a model! please set the --pretrained argument.")
        self.change_state(mode="valid")
        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)
        query_img, query_txt, q_encoder_time = self.get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt, r_encoder_time = self.get_code(self.retrieval_loader, self.args.retrieval_num)
        mAPi2t = calc_map_k(query_img.to(self.args.device), retrieval_txt.to(self.args.device), self.query_labels.to(self.args.device), self.retrieval_labels.to(self.args.device), None, self.rank)
        mAPt2i = calc_map_k(query_txt.to(self.args.device), retrieval_img.to(self.args.device), self.query_labels.to(self.args.device), self.retrieval_labels.to(self.args.device), None, self.rank)
        mAPi2i = calc_map_k(query_img.to(self.args.device), retrieval_img.to(self.args.device), self.query_labels.to(self.args.device), self.retrieval_labels.to(self.args.device), None, self.rank)
        mAPt2t = calc_map_k(query_txt.to(self.args.device), retrieval_txt.to(self.args.device), self.query_labels.to(self.args.device), self.retrieval_labels.to(self.args.device), None, self.rank)
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(f">>>>>> MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}")
        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_dir, str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + ".mat"), result_dict)
        self.logger.info(">>>>>> save all data!")

    
    def valid(self, epoch):
        self.logger.info("Valid.")
        self.change_state(mode="valid")
        query_img, query_txt, q_encoder_time = self.get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt, r_encoder_time = self.get_code(self.retrieval_loader, self.args.retrieval_num)
        #mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        #mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2t = calc_map_k(query_img.to(self.args.device), retrieval_txt.to(self.args.device), self.query_labels.to(self.args.device), self.retrieval_labels.to(self.args.device), None, self.rank)
        mAPt2i = calc_map_k(query_txt.to(self.args.device), retrieval_img.to(self.args.device), self.query_labels.to(self.args.device), self.retrieval_labels.to(self.args.device), None, self.rank)
        #mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        #mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t")
        self.max_mapi2t = max(self.max_mapi2t, mAPi2t)
        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="t2i")
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, \
                    MAX MAP(i->t): {self.max_mapi2t}, MAX MAP(t->i): {self.max_mapt2i}, query_encoder_time: {q_encoder_time}, retrieval_encoder_time: {r_encoder_time}")
        

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t"):
        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)
        #save_path = os.path.join(save_dir, str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + ".mat")
        #torch.save({ 'q_img': query_img, 'q_txt': query_txt, 'r_img': retrieval_img, 'r_txt': retrieval_txt, 'q_l': self.query_labels, 'r_l': self.retrieval_labels }, save_path)
        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()
        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_dir, str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + ".mat"), result_dict)
        self.logger.info(f">>>>>> save best {mode_name} data!")


