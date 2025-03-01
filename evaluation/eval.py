# Adapted from https://github.com/ArrowLuo/CLIP4Clip/blob/668334707c493a4eaee7b4a03b2dae04915ce170/main_task_retrieval.py#L457

import os
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import numpy as np
from evaluation.metrics import compute_metrics
from evaluation.metrics import tensor_text_to_video_metrics
from evaluation.metrics import tensor_video_to_text_sim
from utils.utils import parallel_apply
import torch
from tqdm import tqdm
import pickle
import csv
import shutil

def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
    """run similarity in one single gpu
    Args:
        model: CLIP2Video
        batch_list_t: id of text embedding
        batch_list_v: id of visual embedding
        batch_sequence_output_list: batch text embedding
        batch_visual_output_list: batch visual embedding
    Returns:
        sim_matrix: similarity

    """
    sim_matrix = []
    for idx1, b1 in tqdm(enumerate(batch_list_t)):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in tqdm(enumerate(batch_list_v)):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            # calculate the similarity
            b1b2_logits, *_tmp = model.get_inference_logits(sequence_output, visual_output, input_mask, video_mask)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix

def _run_on_single_gpu_image(model, batch_list_i, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
    """run similarity in one single gpu
    Args:
        model: CLIP2Video
        batch_list_i: id of image embedding
        batch_list_v: id of visual embedding
        batch_sequence_output_list: batch image embedding
        batch_visual_output_list: batch visual embedding
    Returns:
        sim_matrix: similarity

    """
    sim_matrix = []
    for idx1, b1 in tqdm(enumerate(batch_list_i)):
        image_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in tqdm(enumerate(batch_list_v)):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            # calculate the similarity
            b1b2_logits, *_tmp = model.get_inference_logits_image(sequence_output, visual_output, video_mask)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix

def eval_epoch(model, test_dataloader, device, n_gpu, logger):
    """run similarity in one single gpu
    Args:
        model: CLIP2Video
        test_dataloader: data loader for test
        device: device to run model
        n_gpu: GPU number
        batch_sequence_output_list: batch text embedding
        batch_visual_output_list: batch visual embedding
    Returns:
        R1: rank 1 of text-to-video retrieval

    """

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # if multi_sentence_ == True: compute the similarity with multi-sentences retrieval
    multi_sentence_ = False

    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points # used to tag the label when calculate the metric
        sentence_num_ = test_dataloader.dataset.sentence_num # used to cut the sentence representation
        video_num_ = test_dataloader.dataset.video_num # used to cut the video representation
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()

    # embedding_cache_file, need to be modified
    embedding_cache_file = os.path.join('data/msrvtt_data', 'embedding.pkl')

    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0

        # try to load the embedding, or calculate the embedding
        try:
            logger.info("Loading text and video embedding...")
            with open(embedding_cache_file, 'rb') as f:
                batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list = pickle.load(f)

        except FileNotFoundError:
            # extarct the text and video embedding
            logger.info("Fail to load the embedding, calculating the embedding...")
            for bid, batch in tqdm(enumerate(test_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, video, video_mask = batch

                if multi_sentence_:
                    # multi-sentences retrieval means: one frame clip has two or more descriptions.
                    b, *_t = video.shape
                    sequence_output = model.get_sequence_output(input_ids, segment_ids, input_mask)
                    batch_sequence_output_list.append(sequence_output)
                    batch_list_t.append((input_mask, segment_ids,))

                    s_, e_ = total_video_num, total_video_num + b
                    filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                    if len(filter_inds) > 0:
                        video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                        visual_output = model.get_visual_output(video, video_mask)
                        batch_visual_output_list.append(visual_output)
                        batch_list_v.append((video_mask,))
                    total_video_num += b
                else:
                    sequence_output, visual_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask)

                    batch_sequence_output_list.append(sequence_output)
                    batch_list_t.append((input_mask, segment_ids,))

                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))

                # print("{}/{}\r".format(bid, len(test_dataloader)), end="")

            # save the embedding
            logger.info("Saving text and video embedding...")
            with open(embedding_cache_file, 'wb') as f:
                pickle.dump((batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list), f)

        # sim_matrix_cache_file, need to be modified
        sim_matrix_cache_file = os.path.join('data/msrvtt_data', 'sim_matrix.pkl')

        try:
            logger.info("Loading similarity matrix...")
            with open(sim_matrix_cache_file, 'rb') as f:
                sim_matrix = pickle.load(f)

        except FileNotFoundError:
            # calculate the similarity
            logger.info("Fail to load the similarity matrix, calculating the similarity matrix...")
            if n_gpu > 1:
                device_ids = list(range(n_gpu))
                batch_list_t_splits = []
                batch_list_v_splits = []
                batch_t_output_splits = []
                batch_v_output_splits = []
                bacth_len = len(batch_list_t)
                split_len = (bacth_len + n_gpu - 1) // n_gpu
                # split the pairs for multi-GPU
                for dev_id in device_ids:
                    s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                    if dev_id == 0:
                        batch_list_t_splits.append(batch_list_t[s_:e_])
                        batch_list_v_splits.append(batch_list_v)

                        batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                        batch_v_output_splits.append(batch_visual_output_list)
                    else:
                        devc = torch.device('cuda:{}'.format(str(dev_id)))
                        devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]]
                        batch_list_t_splits.append(devc_batch_list)
                        devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_v]
                        batch_list_v_splits.append(devc_batch_list)

                        if isinstance(batch_sequence_output_list[s_], tuple):
                            # for multi_output
                            devc_batch_list = [(b[0].to(devc), b[1].to(devc)) for b in batch_sequence_output_list[s_:e_]]
                        else:
                            # for single_output
                            devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]

                        batch_t_output_splits.append(devc_batch_list)
                        devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                        batch_v_output_splits.append(devc_batch_list)

                parameters_tuple_list = [(batch_list_t_splits[dev_id], batch_list_v_splits[dev_id],
                                            batch_t_output_splits[dev_id], batch_v_output_splits[dev_id]) for dev_id in device_ids]

                # calculate the similarity respectively and concatenate them
                parallel_outputs = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)
                sim_matrix = []
                for idx in range(len(parallel_outputs)):
                    sim_matrix += parallel_outputs[idx]
                sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
            else:
                # calculate the similarity in one GPU
                sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list)
                sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

            # save the similarity
            logger.info("Saving similarity matrix...")
            with open(sim_matrix_cache_file, 'wb') as f:
                pickle.dump(sim_matrix, f)

        # logging the result
        logger.info("Logging the result...")
        R1 = logging_rank(sim_matrix, multi_sentence_, cut_off_points_, logger)
        return R1

def logging_rank(sim_matrix, multi_sentence_, cut_off_points_, logger):
    """run similarity in one single gpu
    Args:
        sim_matrix: similarity matrix
        multi_sentence_: indicate whether the multi sentence retrieval
        cut_off_points_:  tag the label when calculate the metric
        logger: logger for metric
    Returns:
        R1: rank 1 of text-to-video retrieval

    """

    if multi_sentence_:
        # if adopting multi-sequence retrieval, the similarity matrix should be reshaped
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        # compute text-to-video retrieval
        tv_metrics = tensor_text_to_video_metrics(sim_matrix)

        # compute video-to-text retrieval
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    else:
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))

        # compute text-to-video retrieval
        tv_metrics = compute_metrics(sim_matrix)

        # compute video-to-text retrieval
        vt_metrics = compute_metrics(sim_matrix.T)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))


    # logging the result of text-to-video retrieval
    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))

    # logging the result of video-to-text retrieval
    logger.info("Video-to-Text:")
    logger.info(
        '\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.format(
            vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    R1 = tv_metrics['R1']
    return R1

def my_eval_epoch(model, test_dataloader_text, test_dataloader_video, device, n_gpu, logger, dataset_path):
    """run similarity in one single gpu
    Args:
        model: CLIP2Video
        test_dataloader_text: test dataloader of text
        test_dataloader_video: test dataloader of video
        device: device to run model
        n_gpu: GPU number
        batch_sequence_output_list: batch text embedding
        batch_visual_output_list: batch visual embedding
    Returns:
        R1: rank 1 of text-to-video retrieval

    """

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # if multi_sentence_ == True: compute the similarity with multi-sentences retrieval
    multi_sentence_ = False

    cut_off_points_, sentence_num_, video_num_ = [], -1, -1

    model.eval()

    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0

        # extract the text embedding
        logger.info("Calculating the text embedding...")
        for bid, batch in tqdm(enumerate(test_dataloader_text)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids = batch
            text_embedding, _ = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, None, None)
            batch_sequence_output_list.append(text_embedding)
            batch_list_t.append((input_mask, segment_ids,))

        # video_cache_file
        video_cache_file = os.path.join(dataset_path, 'video_embeddings.pkl')

        # extract the video embedding
        try:
            logger.info("Loading video embedding...")
            with open(video_cache_file, 'rb') as f:
                batch_list_v, batch_visual_output_list = pickle.load(f)

        except FileNotFoundError:
            logger.info("Fail to load the video embedding, calculating the video embedding...")
            for bid, batch in tqdm(enumerate(test_dataloader_video)):
                batch = tuple(t.to(device) for t in batch)
                video, video_mask = batch
                _, video_embedding = model.get_sequence_visual_output(None, None, None, video, video_mask)
                batch_visual_output_list.append(video_embedding)
                batch_list_v.append((video_mask,))

            # save the batch_visual_output_list and batch_list_v
            with open(video_cache_file, 'wb') as f:
                pickle.dump((batch_list_v, batch_visual_output_list), f)

        # calculate the similarity
        logger.info("Calculating the similarity matrix...")
        
        # calculate the similarity in one GPU
        sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list)
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    top_3_indices_list = []
    for row in sim_matrix:
        sorted_indices = np.argsort(row)[::-1][:10]
        top_3_indices_list.append(sorted_indices)

    texts = []
    video_ids = []
    with open(f'{dataset_path}/sentences.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            texts.append(row['sentence'])
    
    with open(f'{dataset_path}/video_ids.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_ids.append(row['video_id'])

    text_ids = [i for i in range(1000)]

    for text_id, top_3_indices in zip(text_ids[:10], top_3_indices_list[:10]):
        # 如果 './retrieval_text' 文件夹已经存在，则删除它，没有则创建它
        if os.path.exists('./retrieval_text'):
            import shutil
            shutil.rmtree('./retrieval_text')
        else:
            os.makedirs('./retrieval_text')
        # 创建text_id文件夹
        folder_path = os.path.join('./retrieval_text', str(text_id))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 保存top_3_video_indices到txt文件
        with open(os.path.join(folder_path, 'result.txt'), 'w') as txt_f:
            text = texts[text_id]
            txt_f.write(f"text: {text}\n")
            for index in top_3_indices:
                video_id = video_ids[index]
                txt_f.write(f"{video_id}\n")

                # 复制对应的视频文件
                video_file_path = os.path.join(f'{dataset_path}/all_videos', video_id + '.mp4')
                target_file_path = os.path.join(folder_path, video_id + '.mp4')
                try:
                    if os.path.exists(video_file_path):
                        import shutil
                        shutil.copyfile(video_file_path, target_file_path)
                except FileNotFoundError:
                    pass

def my_eval_epoch_image(model, test_dataloader_image, test_dataloader_video, device, n_gpu, logger, dataset_path):
    """run similarity in one single gpu
    Args:
        model: CLIP2Video
        test_dataloader_image: test dataloader of image
        test_dataloader_video: test dataloader of video
        device: device to run model
        n_gpu: GPU number
        batch_sequence_output_list: batch image embedding
        batch_visual_output_list: batch visual embedding
    Returns:
        R1: rank 1 of image-to-video retrieval

    """

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
        total_video_num = 0

        # extract the image embedding
        logger.info("Calculating the image embedding...")
        for bid, batch in tqdm(enumerate(test_dataloader_image)):
            image_ids, images = batch
            images = images.to(device)
            image_embeddings = model.clip.encode_image(images, return_hidden=True)
            # image_embeddings should be [1, 1, 512] or [1, 512]
            batch_sequence_output_list.append(image_embeddings)
            batch_list_i.append((image_ids,))

        # video_cache_file
        video_cache_file = os.path.join(dataset_path, 'video_embeddings.pkl')

        # extract the video embedding
        try:
            logger.info("Loading video embedding...")
            with open(video_cache_file, 'rb') as f:
                batch_list_v, batch_visual_output_list = pickle.load(f)

        except FileNotFoundError:
            logger.info("Fail to load the video embedding, calculating the video embedding...")
            for bid, batch in tqdm(enumerate(test_dataloader_video)):
                batch = tuple(t.to(device) for t in batch)
                video, video_mask = batch
                _, video_embedding = model.get_sequence_visual_output(None, None, None, video, video_mask)
                batch_visual_output_list.append(video_embedding)
                batch_list_v.append((video_mask,))

            # save the batch_visual_output_list and batch_list_v
            with open(video_cache_file, 'wb') as f:
                pickle.dump((batch_list_v, batch_visual_output_list), f)

        # calculate the similarity
        logger.info("Calculating the similarity matrix...")
        
        # calculate the similarity in one GPU
        sim_matrix = _run_on_single_gpu_image(model, batch_list_i, batch_list_v, batch_sequence_output_list, batch_visual_output_list)
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    top_3_indices_list = []
    for row in sim_matrix:
        sorted_indices = np.argsort(row)[::-1][:10]
        top_3_indices_list.append(sorted_indices)

    # 读取 retrieval_images 下的文件夹个数
    retrieval_folders = sorted([f for f in os.listdir('retrieval_images') if os.path.isdir(os.path.join('retrieval_images', f))])
    video_ids = []
    
    with open(f'{dataset_path}/video_ids.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_ids.append(row['video_id'])

    for retrieval_folder, top_3_indices in zip(retrieval_folders, top_3_indices_list):
        # 保存top_3_video_indices到txt文件
        folder_path = f'retrieval_images/{retrieval_folder}'
        with open(os.path.join(folder_path, 'result.txt'), 'w') as txt_f:
            for topk, index in enumerate(top_3_indices):
                video_id = video_ids[index]
                txt_f.write(f"top-{topk+1}: {video_id}\n")

                # 复制对应的视频文件
                video_file_path = os.path.join(f'{dataset_path}/all_videos', video_id + '.mp4')
                target_file_path = os.path.join(folder_path, video_id + '.mp4')
                try:
                    if os.path.exists(video_file_path):
                        import shutil
                        shutil.copyfile(video_file_path, target_file_path)
                except FileNotFoundError:
                    pass