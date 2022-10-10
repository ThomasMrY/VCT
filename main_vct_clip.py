import os
import math
import numpy as np
import torch.nn as nn
import torch.functional as F

import torch
from torch import optim
import logger
import argparse
import datetime
import gin
import itertools
import dataset
from torchvision import datasets, transforms

from imageio import imwrite
from torchvision.utils import make_grid
from models.visual_concept_tokenizor import VCT_Decoder, VCT_Encoder, MLP_layers
import random
from models.auto_encoder import *

import clip
from einops import repeat

import math

models = {
    'custom': {'vqvae': VQ_CVAE,
               'vqvae2': VQ_CVAE2},
    'imagenet': {'vqvae': VQ_CVAE,
                 'vqvae2': VQ_CVAE2},
    'cifar10': {'vae': CVAE,
                'vqvae': VQ_CVAE,
                'vqvae2': VQ_CVAE2},
    'mnist': {'vae': VAE,
              'vqvae': VQ_CVAE},
    'shapes3d':{
        'vqvae':VQ_CVAE
    },
    'dsprites': {'vae': VAE,
              'vqvae': VQ_CVAE},
}
datasets_classes = {
    'custom': datasets.ImageFolder,
    'imagenet': datasets.ImageFolder,
    'cifar10': datasets.CIFAR10,
    'mnist': datasets.MNIST,
    'shapes3d': dataset.custum_dataset_clip,
    'dsprites': dataset.custum_dataset_clip
}
dataset_train_args = {
    'custom': {},
    'imagenet': {},
    'cifar10': {'train': True, 'download': True},
    'mnist': {'train': True, 'download': True},
    'shapes3d':{},
    'dsprites':{'name':'dsprites'}
}
dataset_test_args = {
    'custom': {},
    'imagenet': {},
    'cifar10': {'train': False, 'download': True},
    'mnist': {'train': False, 'download': True},
    'shapes3d':{},
    'dsprites':{'name':'dsprites'}
}
dataset_n_channels = {
    'custom': 3,
    'imagenet': 3,
    'cifar10': 3,
    'shapes3d':3,
    'dsprites': 1,
    'mnist': 1,
}

dataset_transforms = {
    'custom': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'imagenet': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'cifar10': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'shapes3d': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'mnist': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5), (0.5))]),
    'dsprites': transforms.Compose([transforms.ToTensor(),
                                    lambda x: 255.*x,
                                    transforms.Normalize((0.5), (0.5))])
}


def return_shape(tensor,shape):
    return tensor.permute(0,2,1).view(-1,*shape[1:])

def gen_image(latents, energy_model, im_neg, flags, selected_idx = False):

    num_steps = flags.num_steps
    create_graph = flags.create_graph
    step_lr = flags.step_lr


    # latents list of Tensors shape (B, 1, D)
    im_negs = []
    im_neg.requires_grad_(requires_grad=True)
    for i in range(num_steps):
        energy = 0
        if selected_idx:
            energy = energy_model[selected_idx](im_neg, latents[0])
        else:
            for j in range(len(latents)):
                energy = energy_model[j % flags.components](im_neg, latents[j]) + energy
        im_grad, = torch.autograd.grad([energy.sum()],[im_neg],create_graph = create_graph)
        im_neg = im_neg - step_lr * im_grad

        im_negs.append(im_neg)
        im_neg = im_neg.detach()
        im_neg.requires_grad_()

    return im_neg, im_negs, im_grad

def decode_image(model,im_code, old_shape):
    pred = im_code
    _, cls_pred = pred.max(dim=-1)
    z_q = model.emb.weight[:,cls_pred].permute(1,2,0)
    z_q = return_shape(z_q,old_shape)
    img = model.decode(z_q)
    return img

class Flags():
    def __init__(self):
        pass

@gin.configurable
def get_train_flags(
    resume_iter, 
    num_epochs, 
    num_steps, 
    step_lr, 
    log_interval,
    save_interval,
    create_graph=True,
    without_ml = False,
    emb_loss = False,
    dis_detach = False,
    clip_loss = False,
    **kwargs):
    train_flags = Flags()
    train_flags.resume_iter = resume_iter
    train_flags.num_epochs = num_epochs
    train_flags.num_steps = num_steps
    train_flags.step_lr = step_lr
    train_flags.without_ml = without_ml
    train_flags.dis_detach = dis_detach
    train_flags.clip_loss = clip_loss

    train_flags.create_graph = create_graph
    train_flags.log_interval = log_interval
    train_flags.save_interval = save_interval
    train_flags.emb_loss = emb_loss

    for k,v in kwargs.items():
        train_flags.__setattr__(k,v)
    return train_flags

@gin.configurable
def get_test_flags(
    num_visuals,
    num_steps, 
    step_lr,
    num_additional,
    create_graph=False,
    **kwargs
    ):
    test_flags = Flags()
    test_flags.num_visuals = num_visuals
    test_flags.num_steps = num_steps
    test_flags.step_lr = step_lr
    test_flags.num_additional = num_additional
    test_flags.create_graph = create_graph
    for k,v in kwargs.items():
        test_flags.__setattr__(k,v)

    return test_flags

@gin.configurable
def get_args(
    dataset_dir_name = "",
    image_energy = False,
    joint_train = False,
    load_path = False,
    **kwargs
):
    args = Flags()
    for k,v in kwargs.items():
        args.__setattr__(k,v)
    args.dataset_dir_name = dataset_dir_name
    args.image_energy = image_energy
    args.joint_train = joint_train
    args.load_path = load_path
    return args

@gin.configurable
def get_model_args(
    model,
    hidden,
    k,
    num_channels,
    lr,
    lr_sche,
    **kwargs
    ):
    model_args = Flags()
    model_args.model = model
    model_args.hidden = hidden
    model_args.k = k
    model_args.num_channels = num_channels
    model_args.lr = lr
    model_args.lr_sche = lr_sche
    for k,v in kwargs.items():
        model_args.__setattr__(k,v)
    return model_args

class ImageNormalizer(nn.Module):
    def __init__(self, mean, std):
        super(ImageNormalizer, self).__init__()

        self.mean = torch.as_tensor(mean).view(1, 3, 1, 1)
        self.std = torch.as_tensor(std).view(1, 3, 1, 1)

    def forward(self, input):
        device = input.device
        return (input - self.mean.to(device)) / self.std.to(device)


def encode_recImg(x, model, clip_model):
    im_code = model.encode(x)
    im_emb, _ = model.emb(im_code.detach())
    im_emb = im_emb.view(im_code.shape[0],im_code.shape[1],-1).permute(0,2,1)

    x = F.interpolate(x, mode = 'bicubic', size = 224)
    if not hasattr(clip_model,"norm_preprocess"):
        mu = (0.485, 0.456, 0.406)
        sigma = (0.229, 0.224, 0.225)
        clip_model.norm_preprocess = ImageNormalizer(mu, sigma)
    x = clip_model.norm_preprocess((x + 1.)/2.)
    clip_features = clip_model.encode_image(x)

    return im_emb, clip_features



def train(train_loader, model, clip_model, decoder_model, latent_encoder_clip, mlp_layers, optimizer, batch_size):
    train_flags = get_train_flags()
    train_flags.components = latent_encoder_clip.components
    it = train_flags.resume_iter
    logdir = os.path.join(logger.get_dir(),"checkpoints")
    ce_loss = nn.CrossEntropyLoss()
    os.makedirs(os.path.expanduser(logdir), exist_ok=True)
    for epoch in range(train_flags.num_epochs):
        for batch_idx, (data, data_clip) in enumerate(train_loader):
            data = data.cuda()
            data_clip = data_clip.cuda()
            sample_mat = np.array([[i for i in range(data.shape[0]) if i != j] for j in range(data.shape[0])])
            indexes = [i for i in range(data.shape[0])]
            optimizer.zero_grad()
            im_code = model.encode(data)
            
            old_shape = im_code.shape
            _, labels = model.emb(im_code.detach())

            clip_features = clip_model.encode_image(data_clip)
            clip_features = mlp_layers(clip_features.to(dtype=torch.float32))
            my_latents_clip = latent_encoder_clip(clip_features)

            sampled_concept = np.random.randint(my_latents_clip.shape[1], size = batch_size)
            sampled_index = np.random.randint(batch_size-1, size = batch_size)
            swappings = sample_mat[indexes,sampled_index]

            swapped_latent = my_latents_clip.clone()
            swapped_latent[indexes,sampled_concept] = my_latents_clip[swappings, sampled_concept]

            keeped_indexes = torch.norm(swapped_latent[indexes,sampled_concept]- my_latents_clip[indexes,sampled_concept], dim=-1) > 0.001

            original_latents = my_latents_clip[keeped_indexes]
            original_pred = decoder_model(original_latents)
            original_pred = F.gumbel_softmax(original_pred, tau=1, dim=-1, hard=True)
            z_q_org = torch.einsum("bik,jk->bij", original_pred, model.emb.weight)
            z_q_org = return_shape(z_q_org, old_shape)
            img_org = model.decode(z_q_org)

            _, clip_emb_org = encode_recImg(img_org, model, clip_model)
            if train_flags.dis_detach:
                clip_emb_org = mlp_layers(clip_emb_org.to(dtype=torch.float32).detach())
            else:
                clip_emb_org = mlp_layers(clip_emb_org.to(dtype=torch.float32))
            my_latents_org_clip = latent_encoder_clip(clip_emb_org)

            
            swapped_latents = swapped_latent[keeped_indexes]
            swapped_pred = decoder_model(swapped_latents)
            swapped_pred = F.gumbel_softmax(swapped_pred, tau=1, dim=-1, hard=True)
            z_q_swap = torch.einsum("bik,jk->bij", swapped_pred, model.emb.weight)
            z_q_swap = return_shape(z_q_swap, old_shape)
            img_swap = model.decode(z_q_swap)

            _, clip_emb_swap = encode_recImg(img_swap, model, clip_model)
            if train_flags.dis_detach:
                clip_emb_swap = mlp_layers(clip_emb_swap.to(dtype=torch.float32).detach())
            else:
                clip_emb_swap = mlp_layers(clip_emb_swap.to(dtype=torch.float32))
            my_latents_swap_clip = latent_encoder_clip(clip_emb_swap)


            norm_diff = F.normalize(torch.norm(my_latents_org_clip - my_latents_swap_clip, dim=-1), dim=-1)
            dis_loss = ce_loss(norm_diff, torch.from_numpy(sampled_concept)[keeped_indexes].cuda())




            opred = decoder_model(my_latents_clip)
            pred = opred.reshape(-1,opred.shape[-1])
            labels = labels.reshape(labels.shape[0],-1).reshape(-1)
            im_loss = ce_loss(pred, labels)

            loss = im_loss + dis_loss

            loss.backward()
            optimizer.step()

            if it % train_flags.log_interval == 0:
                loss = loss.item()

                kvs = {}
                kvs['loss'] = loss
                kvs['im_loss'] = im_loss.item()
                kvs['dis_loss'] = dis_loss.item()


                string = "Iteration {} ".format(it)

                for k, v in kvs.items():
                    string += "%s: %.6f  " % (k,v)

                # logger string
                logger.log(string)

            if it % train_flags.save_interval == 0 and it != 0:
                model_path = os.path.join(logdir, "model_{}.pth".format(it))


                ckpt = {}

                ckpt['decoder_model_state_dict'] = decoder_model.state_dict()
                ckpt['mlp_layers_dict'] = mlp_layers.state_dict()
                ckpt['encoder_model_clip_state_dict'] = latent_encoder_clip.state_dict()
                ckpt['optimizer_state_dict'] = optimizer.state_dict()

                torch.save(ckpt, model_path)
                logger.log("Saving model in directory....")
                logger.log('run test')
                with torch.no_grad():
                    image_folder = os.path.join(logger.get_dir(),"images")
                    os.makedirs(os.path.expanduser(image_folder), exist_ok=True)
                    data_rec = decode_image(model, opred[keeped_indexes], old_shape)
                    imgs_record = torch.cat([data[keeped_indexes].detach().cpu(),data_rec.detach().cpu(), img_swap.detach().cpu(), img_org.detach().cpu()], dim=0)
                    imgs_record = torch.clip((imgs_record+1.)/2, 0.0, 1.0)
                    if imgs_record.shape[1] != 3:
                        imgs_record = imgs_record.repeat(1,3,1,1)
                    imgs_record = make_grid(imgs_record, nrow=keeped_indexes.sum().item()).permute(1, 2, 0)
                    imgs_record = imgs_record.numpy()*255
                    imwrite("%s/s%08d_gen.png" % (image_folder,it), imgs_record)

                    print('test at step %d done!' % it)
            it += 1

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--debug", type=bool, default=False,
                        help="debug mode or not")
    parser.add_argument("--config", type=str, default="configs/cifar10.gin",
                        help="config file path")
    meta_args = parser.parse_args()

    seed = np.random.randint(1e6)
    random_seed(seed)
    time = datetime.datetime.now().strftime("exp_clip" + "-%Y-%m-%d-%H-%M-%S-%f")
    gin.parse_config_file(meta_args.config)
    args = get_args()
    gin.constant('num_steps', args.num_steps)
    gin.constant('step_lr', args.step_lr)
    gin.constant('image_energy', args.image_energy)
    gin.parse_config_file(f"configs/{args.name}_shared.gin")

    gin.bind_parameter("get_train_flags.dis_detach", True)
    gin.bind_parameter("get_train_flags.clip_loss", True)


    logger.configure(out_dir="%s_dis_exp"%args.name,debug=meta_args.debug,time=time)
    logger.log(meta_args.config)
    logger.log(f"seed:{seed}")
    model_args = get_model_args()
    model = models[args.name][model_args.model](model_args.hidden, k=model_args.k, num_channels=model_args.num_channels)
    model.cuda()
    # load
    model.load_state_dict(torch.load(model_args.path))

    mlp_layers = MLP_layers(z_dim=512, latent_dim=model_args.hidden, num_latents=36)
    mlp_layers.cuda()

    vct_enc = VCT_Encoder(z_index_dim = model_args.concepts_num_clip, dim = 256)
    vct_enc.cuda()

    perceiver_dec = VCT_Decoder(depth = 4, index_num = model_args.k, z_index_dim=model_args.shape_num, ce_loss=True)
    perceiver_dec.cuda()

    clip_model, preprocess = clip.load('ViT-B/32', "cuda")
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    model.joint_train = args.joint_train
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    optimizer = optim.Adam(itertools.chain(vct_enc.parameters(), mlp_layers.parameters(), perceiver_dec.parameters()), lr=model_args.lr)

    if args.load_path:
        vct_enc.load_state_dict(torch.load(args.load_path)['encoder_model_clip_state_dict'])
        mlp_layers.load_state_dict(torch.load(args.load_path)['mlp_layers_dict'])
        perceiver_dec.load_state_dict(torch.load(args.load_path)['decoder_model_state_dict'])



    kwargs = {'num_workers': 8, 'pin_memory': True}
    dataset_train_dir = os.path.join(args.data_dir, args.dataset_dir_name)
    dataset_test_dir = os.path.join(args.data_dir, args.dataset_dir_name)
    if args.name in ['imagenet', 'custom']:
        dataset_train_dir = os.path.join(dataset_train_dir, 'train')
        dataset_test_dir = os.path.join(dataset_test_dir, 'val')
    train_loader = torch.utils.data.DataLoader(
        datasets_classes[args.name](dataset_train_dir,
                                        transform=dataset_transforms[args.name],
                                        clip_transform=preprocess,
                                        **dataset_train_args[args.name]),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    train(train_loader, model, clip_model, perceiver_dec, vct_enc, mlp_layers, optimizer, args.batch_size)