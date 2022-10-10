
import sys
from main_vct_mask import models, get_args, get_model_args, datasets_classes, dataset_train_args, dataset_n_channels, dataset_transforms, decode_image
from torchvision.utils import make_grid
from models.visual_concept_tokenizor import VCT_Decoder, VCT_Encoder, MLP_layers
from models.auto_encoder import *
import os
import gin
import objects_metrics
import json
import argparse
import cv2 as cv
from imageio import imwrite

def encode_data(data, model, latent_encoder):
    data = data.cuda()
    im_code = model.encode(data)
    old_shape = im_code.shape
    im_emb, labels = model.emb(im_code.detach())
    im_code_new = im_emb.view(im_code.shape[0],im_code.shape[1],-1).permute(0,2,1)

    my_latents = latent_encoder(im_code_new)
    return my_latents, old_shape

def mask_iou(mask1, mask2):
    """
    mask1: [B,m1,n] m1 means number of predicted objects 
    mask2: [B,m2,n] m2 means number of gt objects
    Note: n means image_w x image_h
    """
    bs = mask1.shape[0]
    assert mask1.shape[0] == mask2.shape[0]
    mask2 = mask2.float()
    intersection = torch.einsum("bij,bkj->bik", mask1, mask2)
    area1 = torch.sum(mask1, dim=2).view(bs, 1, -1)
    area2 = torch.sum(mask2, dim=2).view(bs, 1, -1)
    union = (area1.permute(0,2,1) + area2) - intersection
    iou = intersection / union
    return iou
def write_text(result_dict,file):
    with open(file,'w+') as f:
        json.dump(result_dict,f)


gin.parse_config_file("configs/clevr_ce.gin")
args = get_args()
gin.parse_config_file(f"configs/{args.name}_shared.gin")
model_args = get_model_args()
model = models[args.name][model_args.model](model_args.hidden, k=model_args.k, num_channels=model_args.num_channels)
model.cuda()
# load
empty_model = nn.Module()
state_dicts = torch.load(model_args.path,map_location=torch.device('cpu'))
dict_enc = dict([(k.replace("encoder.",""),v) for k,v in state_dicts.items() if 'encoder' in k])
dict_emb = dict([(k.replace("emb.",""),v) for k,v in state_dicts.items() if 'emb' in k])

empty_dict = empty_model.state_dict()
empty_dict.update(dict_enc)
model.encoder.load_state_dict(empty_dict)

empty_dict = empty_model.state_dict()
empty_dict.update(dict_emb)
model.emb.load_state_dict(empty_dict)

perceiver_enc = VCT_Encoder(z_index_dim = model_args.concepts_num)
perceiver_enc.cuda()
perceiver_dec = VCT_Decoder(index_num = 256, z_index_dim=4, ce_loss=True)
perceiver_dec.cuda()

kwargs = {'num_workers': 6, 'pin_memory': True}
dataset_train_dir = os.path.join(args.data_dir, args.name)
dataset_test_dir = os.path.join(args.data_dir, args.name)
if args.name in ['imagenet', 'custom']:
    dataset_train_dir = os.path.join(dataset_train_dir, 'train')
    dataset_test_dir = os.path.join(dataset_test_dir, 'val')
train_loader = torch.utils.data.DataLoader(
    datasets_classes[args.name](dataset_train_dir,
                                    transform=dataset_transforms[args.name],
                                    **dataset_train_args[args.name]),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets_classes[args.name](dataset_train_dir,
                                    transform=dataset_transforms[args.name],
                                    **dataset_train_args[args.name], eval_flag=True),
    batch_size=args.batch_size, shuffle=True, **kwargs)

def get_mask(data, file_name):
    with torch.no_grad():
        bs = data.shape[0]
        my_latents1, old_shape = encode_data(data, model, perceiver_enc)
        my_latents_new = my_latents1.reshape(-1,1,my_latents1.shape[-1])
        opred = perceiver_dec(my_latents_new)
        opred = opred.reshape(opred.shape[0], -1, 4,4)
        im_rec, recons, masks = model.decode_full(opred,bs)
        sample_mask_list = []
        num_masks = []
        for sample_num in range(bs):
            masks_list = []
            for k in range(30):
                imgs_record = ((recons*masks)[sample_num] + 1)/2
                img = imgs_record[k].permute(1,2,0).detach().cpu().numpy()
                mask = np.where(np.any(np.where(np.logical_or(img<0.4,img>0.6), 1., 0.)>0.5, axis=-1),1.,0.)
                output = cv.connectedComponents(np.int8(mask), connectivity=8, ltype=cv.CV_32S)#计算连同域
                for i in range(1,output[0]):
                    template = np.zeros_like(output[1])
                    template[output[1] == i] = 1.
                    masks_list.append(template[None,:])
            masks_all = np.concatenate(masks_list,axis = 0)
            # masks_all_torch = torch.from_numpy(masks_all).float().cuda()
            # imgs_record = 1-make_grid(1-(0.4*((data[sample_num]+1)/2)[None,:].repeat(masks_all_torch.shape[0],1,1,1) + 0.6*masks_all_torch[:,None].repeat(1,3,1,1)), nrow=masks_all_torch.shape[0]).permute(1, 2, 0)
            # imgs_record = torch.clip(imgs_record, 0.0, 1.0)
            # imgs_record = imgs_record.detach().cpu().numpy()
            # imwrite(file_name.replace(".png",f"_{sample_num}.png"), imgs_record)
            sample_mask_list.append(masks_all[None,:])
            num_masks.append(masks_all.shape[1])
        
        sample_mask_list_new = []
        for masks_all in sample_mask_list:
            masks_all_new = np.zeros((1,max(num_masks),64,64))
            masks_all_new[:,:masks_all.shape[1],:,:] = masks_all
            sample_mask_list_new.append(masks_all_new)
    return torch.from_numpy(np.concatenate(sample_mask_list_new, axis=0)).cuda()

    
from tqdm import tqdm

parser = argparse.ArgumentParser(description="training codes")
parser.add_argument("--folder", type=str, default="clevr_dis_exp/abla_mask2_sc_decoder-2022-05-08-09-29-18-064002/",
                        help="debug mode or not")
args = parser.parse_args()
load_path = args.folder
import os
folder = os.path.join(load_path,"checkpoints")
for ckpt in os.listdir(folder):
    if int(ckpt.replace("model_","").replace(".pth","")) % 10000 == 0 and int(ckpt.replace("model_","").replace(".pth","")) > 40000:
    # if int(ckpt.replace("model_","").replace(".pth","")) == 176000:
        ckpt_file = os.path.join(folder, ckpt)
        model.load_state_dict(torch.load(ckpt_file)['model_state_dict'])
        perceiver_enc.load_state_dict(torch.load(ckpt_file)['encoder_model_state_dict'])
        perceiver_dec.load_state_dict(torch.load(ckpt_file)['decoder_model_state_dict'])

        # model.eval()
        # perceiver_enc.eval()
        # perceiver_dec.eval()

        os.makedirs(os.path.expanduser(folder.replace("checkpoints","metrics")), exist_ok=True)
        os.makedirs(os.path.expanduser(folder.replace("checkpoints","masks")), exist_ok=True)
        ari_list_all = []
        sc_list_all = []
        for data, masks in tqdm(test_loader):
            with torch.no_grad():
                data = data.cuda()
                masks = masks.cuda()
                bs = data.shape[0]
                masks = torch.where(masks, 1., 0.)
                pred_masks = get_mask(data, ckpt_file.replace("checkpoints","masks").replace(".pth",".png"))
                iou = mask_iou(masks.reshape(*masks.shape[:2],-1),pred_masks.reshape(*pred_masks.shape[:2],-1))
                range_vector = torch.tensor([[i] for i in range(bs)])
                matched_pred = pred_masks[range_vector, torch.argmax(iou,axis=-1)]
                for i in range(bs):
                    ari_list_all.append(objects_metrics.compute_mask_ari(masks[i,:].detach().cpu(),matched_pred[i,:].detach().cpu()))
                sc_list_all.append(objects_metrics.average_segcover(masks.reshape(-1,1,*masks.shape[2:]), matched_pred.reshape(-1,1,*matched_pred.shape[2:]))[0].item())

        print(ckpt)
        print(f"ari: {np.mean(ari_list_all)}")
        print(f"sc: {np.mean(sc_list_all)}")
        write_text({'ari':np.mean(ari_list_all), "sc":np.mean(sc_list_all)}, ckpt_file.replace("checkpoints","metrics").replace(".pth","_2.json"))
