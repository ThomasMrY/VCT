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
import evaluate

from imageio import imwrite
from torchvision.utils import make_grid
# from 
from models.auto_encoder import *
import math
import random
import data.ground_truth.shapes3d as dshapes3d
import data.ground_truth.mpi3d as dmpi3d
import data.ground_truth.dsprites as ddsprit
import data.ground_truth.cars3d as dcars3d


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
    'mpi_toy':{
        'vqvae':VQ_CVAE
    },
    'mpi_mid':{
        'vqvae':VQ_CVAE
    },
    'mpi_real':{
        'vqvae':VQ_CVAE
    },
    'clevr':{
        'vqvae':VQ_CVAE
    },
    'celebanpz':{
        'vqvae':VQ_CVAE
    },
    'celeba':{
        'vqvae':VQ_CVAE
    },
    'cars3d':{
        'vqvae' : VQ_CVAE
    }
}
datasets_classes = {
    'custom': datasets.ImageFolder,
    'imagenet': datasets.ImageFolder,
    'cifar10': datasets.CIFAR10,
    'mnist': datasets.MNIST,
    # 'mnist' : dataset.custum_dataset,
    'shapes3d': dataset.custum_dataset,
    'dsprites': dataset.custum_dataset,
    'mpi_toy': dataset.custum_dataset,
    'mpi_mid': dataset.custum_dataset,
    'mpi_real': dataset.custum_dataset,
    'clevr': dataset.custum_dataset,
    'celebanpz': dataset.custum_dataset,
    'celeba': dataset.custum_dataset,
    'cars3d': dataset.custum_dataset,
    
}
dataset_train_args = {
    'custom': {},
    'imagenet': {},
    'cifar10': {'train': True, 'download': True},
    'mnist': {'train': True, 'download': True},
    # 'mnist' : {'name':'mnist'},
    'shapes3d':{},
    'dsprites':{'name':'dsprites'},
    'mpi_toy':{'name':'mpi_toy'},
    'mpi_mid':{'name':'mpi_mid'},
    'mpi_real':{'name':'mpi_real'},
    'clevr':{'name':'clevr'},
    'celebanpz': {'name':"celebanpz"},
    'celeba':{'name':"celeba"},
    'cars3d':{'name':'cars3d'}

}
dataset_test_args = {
    'custom': {},
    'imagenet': {},
    'cifar10': {'train': False, 'download': True},
    # 'mnist': {'train': False, 'download': True},
    'mnist' : {'name':'mnist'},
    'shapes3d':{},
    'dsprites':{'name':'dsprites'},
    'mpi_toy':{'name':'mpi_toy'},
    'mpi_mid':{'name':'mpi_mid'},
    'mpi_real':{'name':'mpi_real'},
    'clevr':{'name':'clevr'},
    'celebanpz': {'name':"celebanpz"},
    'celeba':{'name':"celeba"},
    'cars3d':{'name':'cars3d'}
}
dataset_n_channels = {
    'custom': 3,
    'imagenet': 3,
    'cifar10': 3,
    'shapes3d':3,
    'dsprites': 1,
    'mnist': 1,
    'mpi_toy': 3,
    'mpi_mid': 3,
    'mpi_real':3,
    'clevr':3,
    'celebanpz':3,
    'celeba': 3,
    'cars3d': 3
}
eval_datasets = {
    "shapes3d":dshapes3d.Dataset,
    "mpi_toy": dmpi3d.Dataset,
    "dsprites": ddsprit.Dataset,
    "cars3d": dcars3d.Dataset
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
                                    transforms.Normalize((0.5), (0.5))]),
    'mpi_toy': transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'mpi_mid': transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'mpi_real': transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'clevr': transforms.Compose([transforms.Resize(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),    
    'celebanpz': transforms.Compose([transforms.Resize(64),
                                transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'celeba': transforms.Compose([transforms.Resize(64),
                                transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'cars3d': transforms.Compose([transforms.Resize(64),
                                transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
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
    **kwargs):
    train_flags = Flags()
    train_flags.resume_iter = resume_iter
    train_flags.num_epochs = num_epochs
    train_flags.num_steps = num_steps
    train_flags.step_lr = step_lr
    train_flags.without_ml = without_ml

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

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def train(train_loader, model, decoder_model, latent_encoder, optimizer, eval_dataset, test_loader):
    train_flags = get_train_flags()
    train_flags.components = latent_encoder.components
    it = train_flags.resume_iter
    logdir = os.path.join(logger.get_dir(),"checkpoints")
    ce_loss = nn.CrossEntropyLoss()
    os.makedirs(os.path.expanduser(logdir), exist_ok=True)
    for epoch in range(train_flags.num_epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            bs = data.shape[0]
            sample_mat = np.array([[i for i in range(bs) if i != j] for j in range(bs)])
            indexes = np.array([i for i in range(bs)])
            data = data.cuda()
            optimizer.zero_grad()
            with torch.no_grad():
                im_code = model.encode(data)
                old_shape = im_code.shape
            im_emb, labels = model.emb(im_code.detach())
            im_code_new = im_emb.view(im_code.shape[0],im_code.shape[1],-1).permute(0,2,1)

            my_latents = latent_encoder(im_code_new)

            if train_flags.dis_loss:
                with torch.no_grad():
                    sampled_concept = np.random.randint(my_latents.shape[1], size = bs)
                    sampled_index = np.random.randint(bs-1, size = bs)
                    swappings = sample_mat[indexes,sampled_index]

                    swapped_latent = my_latents.clone()
                    swapped_latent[indexes,sampled_concept] = my_latents[swappings, sampled_concept]

                    keeped_indexes = torch.norm(swapped_latent[indexes,sampled_concept]- my_latents[indexes,sampled_concept], dim=-1) > 0.001

                    original_latents = my_latents[keeped_indexes]
                    original_pred = decoder_model(original_latents)
                    original_pred = F.gumbel_softmax(original_pred, tau=1, dim=-1, hard=True)
                    z_q_org = torch.einsum("bik,jk->bij", original_pred, model.emb.weight)
                    z_q_org = return_shape(z_q_org, old_shape)
                    img_org = model.decode(z_q_org)
                    im_code_org = model.encode(img_org)
                im_emb_org, _ = model.emb(im_code_org.detach())
                im_emb_org = im_emb_org.view(im_code_org.shape[0],im_code_org.shape[1],-1).permute(0,2,1)
                my_latents_org = latent_encoder(im_emb_org)


                with torch.no_grad():
                    swapped_latents = swapped_latent[keeped_indexes]
                    swapped_pred = decoder_model(swapped_latents)

                    swapped_pred = F.gumbel_softmax(swapped_pred, tau=1, dim=-1, hard=True)
                    z_q_swap = torch.einsum("bik,jk->bij", swapped_pred, model.emb.weight)
                    z_q_swap = return_shape(z_q_swap, old_shape)
                    img_swap = model.decode(z_q_swap)
                    im_code_swap = model.encode(img_swap)

                im_emb_swap, _ = model.emb(im_code_swap.detach())
                im_emb_swap = im_emb_swap.view(im_code_swap.shape[0],im_code_swap.shape[1],-1).permute(0,2,1)
                my_latents_swap = latent_encoder(im_emb_swap)


                norm_diff = F.normalize(torch.norm(my_latents_org - my_latents_swap, dim=-1), dim=-1)
                dis_loss = ce_loss(norm_diff, torch.from_numpy(sampled_concept)[keeped_indexes].cuda())




            opred = decoder_model(my_latents)
            pred = opred.reshape(-1,opred.shape[-1])
            labels = labels.reshape(labels.shape[0],-1).reshape(-1)
            im_loss = ce_loss(pred, labels)
            if train_flags.dis_loss:
                loss = im_loss + dis_loss
            else:
                loss = im_loss
            loss.backward()

            optimizer.step()

            if it % train_flags.log_interval == 0 and it != 0:
                loss = loss.item()

                kvs = {}
                kvs['loss'] = loss
                kvs['im_loss'] = im_loss.item()
                if train_flags.dis_loss:
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
                ckpt['encoder_model_state_dict'] = latent_encoder.state_dict()
                ckpt['optimizer_state_dict'] = optimizer.state_dict()

                torch.save(ckpt, model_path)
                logger.log("Saving model in directory....")
                logger.log('run test')
                if bs > 16:
                    with torch.no_grad():
                        image_folder = os.path.join(logger.get_dir(),"images")
                        os.makedirs(os.path.expanduser(image_folder), exist_ok=True)
                        sample_num = 16
                        latent1 = my_latents[:sample_num]
                        latent2 = my_latents[sample_num]
                        imgs_list = []
                        for i in range(latent_encoder.latents.shape[0]):
                            swapped_latent = latent1.clone()
                            swapped_latent[:,i] = latent2[None, i].repeat(sample_num,1)
                            opred = vct_dec(swapped_latent)
                            swap_imgs = decode_image(model, opred, old_shape)
                            imgs_list.append(swap_imgs)
                        imgs_record = torch.cat([data[:sample_num].detach().cpu(), data[sample_num][None,:,:,:].repeat(sample_num,1,1,1).detach().cpu()] + [rec.detach().cpu() for rec in imgs_list], dim=0)
                        imgs_record = torch.clip((imgs_record+1.)/2, 0.0, 1.0)
                        if imgs_record.shape[1] != 3:
                            imgs_record = imgs_record.repeat(1,3,1,1)
                        imgs_record = make_grid(imgs_record, nrow=sample_num).permute(1, 2, 0)
                        imgs_record = imgs_record.numpy()*255
                        imwrite("%s/s%08d_split.png" % (image_folder,it), imgs_record)

                    print('test at step %d done!' % it)
                if train_loader.dataset.eval_flag:
                    metric_folder = os.path.join(logger.get_dir(),"metrics")
                    os.makedirs(os.path.expanduser(metric_folder), exist_ok=True)
                    eval_latents = evaluate.enc_func(test_loader, model, latent_encoder)
                    evaluate.eval_func(eval_dataset, eval_latents, metric_folder, it)
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
    parser.add_argument("--enc_model", type=str, default="sc_decoder",
                        help="config file path")
    parser.add_argument("--wo_dis_loss", type=bool, default=False,
                    help="config settings")
    parser.add_argument("--batch_size", type=int, default=-1,
                help="config settings")
    parser.add_argument("--concepts_num", type=int, default=20,
            help="config settings")
    parser.add_argument("--seed", type=int, default=-1,
    help="config settings")
    meta_args = parser.parse_args()
    time = datetime.datetime.now().strftime("exp" + (f"_cn{meta_args.concepts_num}_") + (f"_b{meta_args.batch_size}_" if not meta_args.batch_size == -1 else "") + ("wo_dis_" if meta_args.wo_dis_loss else "") + meta_args.enc_model + "-%Y-%m-%d-%H-%M-%S-%f")
    if meta_args.enc_model == "sc_decoder":
        from models.visual_concept_tokenizor import VCT_Decoder, VCT_Encoder
    else:
        assert NotImplemented
    gin.parse_config_file(meta_args.config)
    if meta_args.seed == -1:
        seed = np.random.randint(1e6)
    else:
        seed = meta_args.seed
    random_seed(seed)
    args = get_args()
    gin.constant('num_steps', args.num_steps)
    gin.constant('step_lr', args.step_lr)
    gin.constant('image_energy', args.image_energy)
    gin.parse_config_file(f"configs/{args.name}_shared.gin")
    
    if not meta_args.wo_dis_loss:
        gin.bind_parameter("get_train_flags.dis_loss", True)
    else:
        gin.bind_parameter("get_train_flags.dis_loss", False)

    logger.configure(out_dir="%s_dis_exp"%args.name,debug=meta_args.debug,time=time)
    logger.log(f"seed:{seed}")
    logger.log(meta_args.config)
    model_args = get_model_args()
    model = models[args.name][model_args.model](model_args.hidden, k=model_args.k, num_channels=model_args.num_channels)
    model.cuda()
    # load
    model.load_state_dict(torch.load(model_args.path, map_location='cpu'))

    vct_enc = VCT_Encoder(z_index_dim = meta_args.concepts_num)
    vct_dec = VCT_Decoder(index_num = model_args.k, z_index_dim=model_args.shape_num, ce_loss=True)

    vct_enc.cuda()
    vct_dec.cuda()

    model.joint_train = args.joint_train
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    optimizer = optim.Adam(itertools.chain(vct_enc.parameters(),vct_dec.parameters()), lr=model_args.lr)

    if args.load_path:
        vct_enc.load_state_dict(torch.load(args.load_path)['encoder_model_state_dict'])
        vct_dec.load_state_dict(torch.load(args.load_path)['decoder_model_state_dict'])



    kwargs = {'num_workers': 8, 'pin_memory': True}
    dataset_train_dir = os.path.join(args.data_dir, args.name)
    dataset_test_dir = os.path.join(args.data_dir, args.name)
    if args.name in ['imagenet', 'custom']:
        dataset_train_dir = os.path.join(dataset_train_dir, 'train')
        dataset_test_dir = os.path.join(dataset_test_dir, 'val')
    train_dataset = datasets_classes[args.name](dataset_train_dir,
                                        transform=dataset_transforms[args.name],
                                        **dataset_train_args[args.name])
    if args.name in ['mnist']:
        train_dataset.eval_flag = False
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=(args.batch_size if meta_args.batch_size == -1 else meta_args.batch_size), shuffle=True, **kwargs)
    if train_loader.dataset.eval_flag:
        test_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64, shuffle=False, **kwargs)
        eval_dataset = eval_datasets[args.name](np.arange(0,len(train_loader.dataset)))

        train(train_loader, model, vct_dec, vct_enc, optimizer, eval_dataset, test_loader)
    else:
        train(train_loader, model, vct_dec, vct_enc, optimizer, None, None)
