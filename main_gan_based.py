import os
import math
import numpy as np
import torch.nn as nn
import torch.functional as F
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
# torch.autograd.set_detect_anomaly(True)
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
from models.visual_concept_tokenizor import VCT_Decoder, VCT_Encoder
import random
from models.auto_encoder import *
from einops import repeat

from timm.models import create_model
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from stylegan2.VAE_deep import GANbaseline2 as GANbaseline
from LD.latent_deformator import LatentDeformator
from LD.latent_deformator import DeformatorType
from stylegan2.models import Generator, get_generator
import math


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
class ImageNormalizer(nn.Module):
    def __init__(self, mean, std):
        super(ImageNormalizer, self).__init__()

        self.mean = torch.as_tensor(mean).view(1, 3, 1, 1)
        self.std = torch.as_tensor(std).view(1, 3, 1, 1)

    def forward(self, input):
        device = input.device
        return (input - self.mean.to(device)) / self.std.to(device)

def normlize_beit(beit_model, x):
    x = F.interpolate(x, mode = 'bicubic', size = 224)
    if not hasattr(beit_model,"norm_preprocess"):
        mu = IMAGENET_INCEPTION_MEAN
        sigma = IMAGENET_INCEPTION_STD
        beit_model.norm_preprocess = ImageNormalizer(mu, sigma)
    return beit_model.norm_preprocess(x)

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

def generate(generator,rep):
    imgs, _ = generator(styles = [rep],input_is_latent=True)
    return imgs


def train(encoder, mlp, generator, latent_encoder, optimizer, batch_size, **kwargs):
    train_flags = get_train_flags()
    it = train_flags.resume_iter
    ce_loss = nn.CrossEntropyLoss()
    logdir = os.path.join(logger.get_dir(),"checkpoints")
    os.makedirs(os.path.expanduser(logdir), exist_ok=True)
    for epoch in range(train_flags.num_epochs):
        for ids in range(500):
            encoder.train()
            mlp.train()
            latent_encoder.train()
            generator.zero_grad()
            optimizer.zero_grad()
            
            noise = torch.randn(batch_size, 512).cuda()
            z = generator.style(noise)
            imgs = generate(generator,z)
            conv_feature_org = encoder(imgs)
            conv_feature_org = conv_feature_org.reshape(conv_feature_org.shape[0],conv_feature_org.shape[1],-1)
            my_latents_clip_org = latent_encoder(conv_feature_org.permute(0,2,1))

            
            target_indice = torch.randint(0,latent_encoder.num_latents, [batch_size], device='cuda') 
            shifts_1 = make_specific_shift(target_indice, batch_size, 512)
            shifts_1 = mlp(shifts_1)
            imgs_shifted_1 = generate(generator, z + shifts_1)
            conv_feature_swap = encoder(imgs_shifted_1)
            conv_feature_swap = conv_feature_swap.reshape(conv_feature_swap.shape[0],conv_feature_swap.shape[1],-1)
            my_latents_clip_swap = latent_encoder(conv_feature_swap.permute(0,2,1))


            norm_diff = F.normalize(torch.norm(my_latents_clip_org - my_latents_clip_swap, dim=-1), dim=-1)
            dis_loss = ce_loss(norm_diff, target_indice.cuda())


            loss = dis_loss

            loss.backward()
            optimizer.step()

            if it % train_flags.log_interval == 0:
                loss = loss.item()

                kvs = {}
                kvs['loss'] = loss
                string = "Iteration {} ".format(it)

                for k, v in kvs.items():
                    string += "%s: %.6f  " % (k,v)

                # logger string
                logger.log(string)

            if it % train_flags.save_interval == 0:
                model_path = os.path.join(logdir, "model_{}.pth".format(it))


                ckpt = {}
                ckpt['encoder_state_dict'] = encoder.state_dict()
                ckpt['mlp_state_dict'] = mlp.state_dict()
                ckpt['encoder_model_clip_state_dict'] = latent_encoder.state_dict()
                ckpt['optimizer_state_dict'] = optimizer.state_dict()

                torch.save(ckpt, model_path)
                logger.log("Saving model in directory....")

                with torch.no_grad():
                    image_folder = os.path.join(logger.get_dir(),"images")
                    os.makedirs(os.path.expanduser(image_folder), exist_ok=True)
                    mlp.eval()
                    generator.eval()
                    
                    with torch.no_grad():
                        noise = torch.randn(1,512).cuda()
                        style = generator.style(noise)

                        # first do W
                        samples = []
                        for k in range(latent_encoder.num_latents):
                            interpolation = torch.arange(-16, 16, 3)
                            for val in interpolation:
                                z = torch.zeros(512).cuda()
                                # print("z", z.shape)
                                z[k] = val
                                shift = mlp(z)
                                sample = generate(generator, style + shift)
                                sample = ((sample+1) / 2).clamp(0,1).cpu()
                                samples.append(sample)

                        samples = torch.cat(samples, dim = 0)
                        output = make_grid(samples, nrow= 11, padding = 0)

                    imgs_record = output.permute(1, 2, 0).cpu().numpy()*255
                    imwrite("%s/s%08d_split.png" % (image_folder,it), imgs_record)  
                        
            it += 1

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def make_specific_shift(target_indices, batch_size, latent_dim):
    shifts =  torch.randn(target_indices.shape, device='cuda')

    shifts = shift_scale * shifts
    shifts[(shifts < min_shift) & (shifts > 0)] = min_shift
    shifts[(shifts > min_shift) & (shifts < 0)] = -min_shift
    
    try:
        latent_dim[0]
        latent_dim = list(latent_dim)
    except Exception:
        latent_dim = [latent_dim]
    
    z_shift = torch.zeros([batch_size] + latent_dim, device='cuda')
    for i, (index, val) in enumerate(zip(target_indices, shifts)):
        z_shift[i][index] += val
    
    return z_shift
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--debug", type=bool, default=False,
                        help="debug mode or not")
    parser.add_argument("--schdle", type=bool, default=False,
                        help="debug mode or not")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size for training")
    parser.add_argument("--rand_seed", type=int, default=-1,
                        help="load file path")
    parser.add_argument("--config", type=str, default="configs/cifar10.gin",
                        help="config file path")
    parser.add_argument("--load_path", type=str, default="None",
                        help="load file path")      
    meta_args = parser.parse_args()

    if meta_args.rand_seed == -1:
        seed = np.random.randint(1e6) #913213
    else:
        seed = meta_args.rand_seed
    random_seed(seed)
    time = datetime.datetime.now().strftime(f"gan_base" + "-%Y-%m-%d-%H-%M-%S-%f")
    gin.parse_config_file(meta_args.config)
    args = get_args()
    gin.constant('num_steps', args.num_steps)
    gin.constant('step_lr', args.step_lr)
    gin.constant('image_energy', args.image_energy)
    gin.parse_config_file(f"configs/{args.name}_shared.gin")

    gin.bind_parameter("get_train_flags.dis_detach", True)
    gin.bind_parameter("get_train_flags.clip_loss", True)


    logger.configure(out_dir="%s_bert_exp"%args.name,debug=meta_args.debug,time=time)
    logger.log(meta_args.config)
    logger.log(f"seed:{seed}")
    model_args = get_model_args()

    used_dim = 30

    vct_enc = VCT_Encoder(z_index_dim = used_dim, ce_loss=True, dim = 256, latent_dim = 256, depth = 4)
    vct_enc.cuda()

    GAN_W_dim = 512
    VAE_dim = 512

    batch_size = 32
    N = batch_size

    shift_scale = 6.0
    min_shift = 0.5

    DEFORMATOR_TYPE_DICT = {
    'fc': DeformatorType.FC,
    'linear': DeformatorType.LINEAR,
    'id': DeformatorType.ID,
    'ortho': DeformatorType.ORTHO,
    'proj': DeformatorType.PROJECTIVE,
    'random': DeformatorType.RANDOM,
    'deeper': DeformatorType.DEEPER_FC
    }

    # now create model and init a name
    # get generators
    generator = get_generator(args.name,args)
    generator.eval().cuda()
    for p in generator.parameters():
        p.requires_grad_(False)
        
    mlp = LatentDeformator( shift_dim= VAE_dim,
                    input_dim= VAE_dim,
                    out_dim= GAN_W_dim,
                    type=DEFORMATOR_TYPE_DICT["ortho"],
                    random_init= True).cuda()
    
    encoder = Encoder4GAN(model_args.hidden, num_channels=model_args.num_channels)
    encoder.cuda()

    if meta_args.schdle:
        import sched
        optimizer = sched.ScheduledOptim(optim.Adam(itertools.chain(mlp.parameters(), vct_enc.parameters(), encoder.parameters()),betas=(0.9, 0.98), eps=1e-09),lr_mul=2.0,d_model=256, n_warmup_steps=5000)
    else:
        optimizer = optim.Adam(itertools.chain(mlp.parameters(), vct_enc.parameters(),encoder.parameters()), lr=model_args.lr)

    if meta_args.load_path != "None":
        ckpt = torch.load(meta_args.load_path)
        vct_enc.load_state_dict(ckpt['encoder_model_clip_state_dict'])
        encoder.load_state_dict(ckpt['encoder_state_dict'])

    train(encoder, mlp, generator, vct_enc, optimizer, batch_size)