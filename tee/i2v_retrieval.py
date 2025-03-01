#coding:utf-8
# @Time : 2021/6/19
# @Author : Han Fang
# @File: infer_retrieval.py
# @Version: version 1.0
import time 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import io
import torch
import numpy as np
import random
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.modeling import CLIP2Video
from evaluation.eval import eval_epoch, my_eval_epoch_image
from torch.profiler import profile, record_function, ProfilerActivity

from utils.config import get_args
from utils.utils import get_logger
from utils.dataloader import dataloader_msrvtt_test_video, dataloader_msrvtt_test_image

import csv
from PIL import Image
import csv
from tqdm import tqdm
import pickle
import argparse
# from torch.utils.bottleneck import profile

# define the dataloader
# new dataset can be added from import and inserted according to the following code
DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"test_video":dataloader_msrvtt_test_video, "test_image":dataloader_msrvtt_test_image}



def set_seed_logger(args):
    """Initialize the seed and environment variable

    Args:
        args: the hyper-parameters.

    Returns:
        args: the hyper-parameters modified by the random seed.

    """

    global logger

    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # get logger
    logger = get_logger()

def init_device(args, local_rank):
    """Initialize device to determine CPU or GPU

     Args:
         args: the hyper-parameters
         local_rank: GPU id

     Returns:
         devices: cuda
         n_gpu: number of gpu

     """
    global logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
    n_gpu = 1
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu


def init_model(args, device):
    """Initialize model.

    if location of args.init_model exists, model will be initialized from the pretrained model.
    if no model exists, the training will be initialized from CLIP's parameters.

    Args:
        args: the hyper-parameters
        devices: cuda

    Returns:
        model: the initialized model

    """

    # resume model if pre-trained model exist.
    model_file = os.path.join(args.checkpoint, "pytorch_model.bin.{}".format(args.model_num))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
    else:
        model_state_dict = None
        if args.local_rank == 0:
            logger.info("Model loaded fail %s", model_file)

    # Prepare model
    model = CLIP2Video.from_pretrained(args.cross_model, cache_dir=None, state_dict=model_state_dict,
                                       task_config=args)
    model.to(device)

    return model

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
        
def main():

    global logger

    # obtain the hyper-parameter
    parser = argparse.ArgumentParser('CLIP2Video on Visual-Video Retrieval Task')
    parser.add_argument("--do_eval", type=bool, default=True, help="Whether to run eval on the dev set.")
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--video_val_csv', type=str, default='./data/video_ids.csv', help='')
    parser.add_argument('--text_val_csv', type=str, default='./data/sentences.csv', help='')
    parser.add_argument('--data_path', type=str, default='./data/MSRVTT_data.json', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='no need, we have embeddings', help='feature path')
    parser.add_argument('--num_thread_reader', type=int, default=2, help='')
    parser.add_argument('--batch_size_val', type=int, default=64, help='batch size eval')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=32, help='')
    parser.add_argument('--max_frames', type=int, default=12, help='')
    parser.add_argument('--feature_framerate', type=int, default=2, help='frame rate for uniformly sampling the video')
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")


    # important extra argument for training and testing CLIP2Video
    parser.add_argument('--sim_type', type=str, default="seqTransf", choices=["meanP", "seqTransf"],
                        help="choice a similarity header.")

    # argument for testing
    parser.add_argument('--checkpoint', type=str, default='./pretrained_weights', help="checkpoint dir")
    parser.add_argument('--model_num', type=str, default='2', help="model id")
    parser.add_argument('--local_rank', default=0, type=int, help='shard_id: node rank for distributed training')
    parser.add_argument("--datatype", default="msrvtt", type=str, help="msvd | msrvtt | vatexEnglish | msrvttfull")

    # for different vocab size
    parser.add_argument('--vocab_size', type=int, default=49408, help="the number of vocab size")

    # for TDB block
    parser.add_argument('--temporal_type', type=str, default='TDB', help="TDB type") # 
    parser.add_argument('--temporal_proj', type=str, default='sigmoid_selfA', help="sigmoid_mlp | sigmoid_selfA")

    # for TAB block
    parser.add_argument('--center_type', type=str, default='TAB', help="TAB")
    parser.add_argument('--centerK', type=int, default=5, help='center number for clustering.')
    parser.add_argument('--center_weight', type=float, default=0.5, help='the weight to adopt the main simiarility')
    parser.add_argument('--center_proj', type=str, default='TAB_TDB', help='TAB | TAB_TDB')

    # model path of clip
    parser.add_argument('--clip_path', type=str,
                        default='./pretrained_weights/ViT-B-32.pt',
                        help="model path of CLIP")

    # for permutation 
    parser.add_argument('--TR', action='store_true', help="Text Row Permutation")
    parser.add_argument('--TC', action='store_true', help="Text Column Permutation")
    parser.add_argument('--VR', action='store_true', help="Visual Row Permutation")
    parser.add_argument('--VC', action='store_true', help="Visual Column Permutation")

    # for topk
    parser.add_argument('--topk', type=int, default=3, help='top-k for retrieval')

    args = parser.parse_args()

    # set the seed
    set_seed_logger(args)

    # setting the testing device
    device, n_gpu = init_device(args, args.local_rank)

    # setting tokenizer
    tokenizer = ClipTokenizer()
    # init model
    model = init_model(args, device)

    # get data from console input
    img_names = []
    while True:
        name = input("Input an iamge id (input exit to exit):")
        if name == "exit":
            break
        img_names.append(f'image{name}.jpg')

    img_folder = '../image_data'
    img_paths = []
    for name in img_names:
        img_paths.append(os.path.join(img_folder, name))

    # save user input images in order
    if os.path.exists(f'./retrieval_images'):
        import shutil
        shutil.rmtree(f'./retrieval_images')

    if not os.path.exists(f'retrieval_images'):
        os.mkdir(f'retrieval_images')
    for idx, img_path in enumerate(img_paths):
        img = Image.open(img_path).convert('RGB')
        name = str(img_path.split('/')[-1].split('.')[0])

        if not os.path.exists(f'retrieval_images/retrieval_{idx+1}'):
            os.mkdir(f'retrieval_images/retrieval_{idx+1}')
        img.save(f'retrieval_images/retrieval_{idx+1}/image.jpg')

    # init test dataloader
    assert args.datatype in DATALOADER_DICT
    test_dataloader_image, test_length_image = DATALOADER_DICT[args.datatype]["test_image"](args, img_paths)
    # no need, we have embeddings
    # test_dataloader_video, test_length_video = DATALOADER_DICT[args.datatype]["test_video"](args)  

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # if multi_sentence_ == True: compute the similarity with multi-sentences retrieval
    multi_sentence_ = False

    cut_off_points_, sentence_num_, video_num_ = [], -1, -1

    model.eval()

    with torch.no_grad():
        batch_list_i = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        batch_visual_TAB_list = []
        total_video_num = 0

        # extract the image embedding
        logger.info("Calculating the image embedding...")
        for bid, batch in tqdm(enumerate(test_dataloader_image)):
            image_ids, images = batch
            images = images.to(device)
            # image_embeddings = model.clip.encode_image(images, return_hidden=True)

            # F1 part
            embedding_output = model.clip.visual.F1_forward(images)

            # Backbone part
            encoder_outputs = model.clip.visual.Backbone_forward(embedding_output)

            # F2 part
            image_embeddings = model.clip.visual.F2_forward(encoder_outputs)

            # image_embeddings should be [1, 1, 512] or [1, 512]
            batch_sequence_output_list.append(image_embeddings)
            batch_list_i.append((image_ids,))

        # video_cache_file
        video_cache_file = os.path.join('./data','video_embeddings.pkl')

        # extract the video embedding
        try:
            logger.info("Loading video embedding...")
            with open(video_cache_file, 'rb') as f:
                batch_list_v, batch_visual_output_list, batch_visual_TAB_list, batch_visual_TDB_list = CPU_Unpickler(f).load()

        except FileNotFoundError:
            # logger.info("Fail to load the video embedding, calculating the video embedding...")
            # for bid, batch in tqdm(enumerate(test_dataloader_video)):
            #     batch = tuple(t.to(device) for t in batch)
            #     video, video_mask = batch
            #     _, video_embedding = model.get_sequence_visual_output(None, None, None, video, video_mask)
            #     batch_visual_output_list.append(video_embedding)
            #     batch_list_v.append((video_mask,))

            # # save the batch_visual_output_list and batch_list_v
            # with open(video_cache_file, 'wb') as f:
            #     pickle.dump((batch_list_v, batch_visual_output_list), f)
            raise FileNotFoundError

        # calculate the similarity
        logger.info("Calculating the similarity matrix...")
        sim_matrix = []
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # Profile both CPU and GPU
        #             record_shapes=True,  # Record input shapes
        #             with_stack=True,  # Record stack traces
        #         ) as prof:
        #     with record_function("similarity_matrix"):
        for idx1, b1 in tqdm(enumerate(batch_list_i)):
            image_ids, *_tmp = b1
            sequence_output = batch_sequence_output_list[idx1]
            each_row = []
            for idx2, b2 in tqdm(enumerate(batch_list_v)):
                video_mask, *_tmp = b2
                visual_output = batch_visual_output_list[idx2]
                visual_TDB = batch_visual_TDB_list[idx2]
                visual_TAB = batch_visual_TAB_list[idx2]
                # calculate the similarity
                b1b2_logits, *_tmp = model.get_inference_logits_image(sequence_output, visual_output, video_mask, visual_TAB, visual_TDB)
                b1b2_logits = b1b2_logits.cpu().detach().numpy()
                each_row.append(b1b2_logits)
            each_row = np.concatenate(tuple(each_row), axis=-1)
            sim_matrix.append(each_row)
    # print(prof.key_averages().table(sort_by="cpu_time_total"))
    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    top_k_indices_list = []
    for i, row in enumerate(sim_matrix):
        sorted_indices = np.argsort(row)[::-1][:args.topk]
        top_k_indices_list.append(sorted_indices)
        k = args.topk
        acc_indices = sorted_indices
        while int(img_names[i].split("image")[-1].split(".")[0]) not in acc_indices:
            acc_indices = np.argsort(row)[::-1][:k]
            k+=1
        print(f"{img_names[i]} found in top {k} indices")
    # read the number of folders in the "retrieval_images" directory
    retrieval_folders = sorted([f for f in os.listdir('retrieval_images') if os.path.isdir(os.path.join('retrieval_images', f))])
    
    video_ids = [ f'video{i}' for i in range(10000)]

    for retrieval_folder, top_k_indices in zip(retrieval_folders, top_k_indices_list):
        # save the result to the txt file
        folder_path = f'retrieval_images/{retrieval_folder}'
        with open(os.path.join(folder_path, 'result.txt'), 'w') as txt_f:
            for topk, index in enumerate(top_k_indices):
                video_id = video_ids[index]
                txt_f.write(f"top-{topk+1}: {video_id}\n")

                # copy the video to the folder
                video_file_path = os.path.join(f'../video-data', video_id + '.mp4')
                target_file_path = os.path.join(folder_path, video_id + '.mp4')
                try:
                    if os.path.exists(video_file_path):
                        import shutil
                        shutil.copyfile(video_file_path, target_file_path)
                except FileNotFoundError:
                    pass

if __name__ == "__main__":
    main()
