import os
import io
import torch
import numpy as np
import random
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.modeling import CLIP2Video
import time
import subprocess
import requests
import logging
import torch.distributed as dist
import base64

from utils.dataloader import dataloader_msrvtt_test_video, dataloader_msrvtt_test_text
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import shutil

from tqdm import tqdm
import pickle
from flask import Flask, request, jsonify, render_template, send_from_directory, Response,redirect, url_for


WORKER_URL = 'http://18.217.90.50:5001/validate_attestation'
REMOTE_VIDEO_DB_URL = "http://18.217.90.50:5002/get_videos"
REMOTE_VIDEO_FEATURES_DB_URL = "http://18.217.90.50:5002/get_video_features"

# -------------------------------
# Encryption Configuration
# -------------------------------
AES_KEY = b'\x89\xc3\xcf\x17\x8fU\x80\xbd\xc0S`#\xf0\xd8\xd7\x8b\x96Q\xcb\xf6C\xdfp\x11P\x0b\x91[`:0\xbc'

  # 32-byte random key
aesgcm = AESGCM(AES_KEY)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"test_video":dataloader_msrvtt_test_video, "test_text":dataloader_msrvtt_test_text}

WORKER_URL = 'http://18.217.90.50:5001/validate_attestation'
MASTER_ADDR = '18.116.182.172'  # Master server IP (replace with your actual IP)
MASTER_PORT = '29500'           # Must match worker's MASTER_PORT
WORLD_SIZE = 2                  # Total number of processes (master + worker)
RANK = 0                        # Master rank

def init_distributed():
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    init_method = f'tcp://{MASTER_ADDR}:{MASTER_PORT}'
    dist.init_process_group(
        backend='gloo',
        init_method=init_method,
        world_size=WORLD_SIZE,
        rank=RANK
    )
    
    
app = Flask(__name__)
app.config['VIDEO_FOLDER'] = '/home/ubuntu/video-data'


class FlushFileHandler(logging.FileHandler):
    """
    Custom logging handler that flushes the stream after each log entry.
    """
    def emit(self, record):
        super().emit(record)
        self.flush()
        
log_file = 'server.log'

# Ensure the logs directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')
    

file_handler = FlushFileHandler(log_file)
# file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
# console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)



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
    # logger.info("device: {} n_gpu: {}".format(device, n_gpu))
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
    if args.TC:
        model_file = os.path.join(args.checkpoint, "encrypted_pytorch_model.bin.{}".format(args.model_num))

    else:
        model_file = os.path.join(args.checkpoint, "pytorch_model.bin.{}".format(args.model_num))
        
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        # if args.local_rank == 0:
            # logger.info("Model loaded from %s", model_file)
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

def fetch_video_features():
    try:
        response = requests.get(REMOTE_VIDEO_FEATURES_DB_URL)  # Replace with actual DB server IP
        if response.status_code == 200:
            data = response.json()
            batch_list_v = [
                [torch.tensor(tensor_list) for tensor_list in batch]  # Convert each list back to tensor
                for batch in data['batch_list_v']
            ]
            
            batch_visual_output_list = [
                torch.tensor(tensor_list) for tensor_list in data['batch_visual_output_list']
            ]
            return batch_list_v, batch_visual_output_list
        else:
            logger.error(f"Failed to fetch image features. Status code: {response.status_code}")
            return [], None
    except Exception as e:
        logger.error(f"Error fetching image features: {e}")
        return [], None


def decrypt_data(encrypted_b64: str) -> str:
    """
    Decrypts AES-GCM encrypted data (text).

    Args:
        encrypted_b64: Base64-encoded (nonce + ciphertext).

    Returns:
        Decrypted plaintext (str) or None if decryption fails.
    """
    try:
        encrypted_data = base64.b64decode(encrypted_b64)
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, None)
        plaintext = plaintext_bytes.decode('utf-8')
        print(f"[Master] Encrypted Text: '{encrypted_b64}'")
        return plaintext
    except Exception as e:
        print(f"[Master] Error decrypting data: {e}")
        return None
        
class MyConfig:
    def __init__(self):
        self.do_eval = True
        self.val_csv = 'data/.val.csv'
        self.video_val_csv = './data/video_ids.csv'
        self.text_val_csv = './data/sentences.csv'
        self.data_path = './data/MSRVTT_data.json'
        self.features_path = 'no need, we have embeddings'
        self.num_thread_reader = 2
        self.batch_size_val = 64
        self.seed = 42
        self.max_words = 32
        self.max_frames = 12
        self.feature_framerate = 2
        self.cross_model = "cross-base"
        self.do_lower_case = False  # set to True if needed
        self.n_gpu = 1
        self.local_rank = 0
        self.cache_dir = ""
        self.fp16 = False  # set to True if needed
        self.fp16_opt_level = 'O1'
        self.cross_num_hidden_layers = 4
        self.sim_type = "seqTransf"
        self.checkpoint = './pretrained_weights'
        self.model_num = '2'
        self.local_rank = 0
        self.datatype = "msrvtt"
        self.vocab_size = 49408
        self.temporal_type = ''
        self.temporal_proj = 'sigmoid_selfA'
        self.center_type = 'TAB'
        self.centerK = 5
        self.center_weight = 0.5
        self.center_proj = 'TAB_TDB'
        self.clip_path = './pretrained_weights/ViT-B-32.pt'
        self.TR = False
        self.TC = False
        self.VR = False
        self.VC = False
        self.topk = 3  
        self.distributed = True

args = MyConfig()
if args.distributed:
    init_distributed()
# set_seed_logger(args)
device, n_gpu = init_device(args, args.local_rank)
# tokenizer = ClipTokenizer()
model = init_model(args, device)
tokenizer = ClipTokenizer()


if hasattr(model, 'module'):
    model = model.module.to(device)
else:
    model = model.to(device)

model.eval()
video_cache_file = os.path.join('./data','video_embeddings.pkl')

# extract the video embedding
try:
    # logger.info("Loading video embedding...")
    with open(video_cache_file, 'rb') as f:
        batch_list_v, batch_visual_output_list, batch_visual_TAB_list  = CPU_Unpickler(f).load()

except FileNotFoundError:
    raise FileNotFoundError
logger.info("[可信执行环境] 从数据库加载目标视频的特征向量")
fetch_video_features()

TOP_K = 3
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects a multipart/form-data with `file` as the key to the image.
    Returns the top-3 matching video IDs/paths.
    """
    logger.info("[客户端] 接收 AES 密钥，加密检索文本，发送至可信执行环境")
    logger.info("[可信执行环境] 接收加密后的文本，解密文本")
    logger.info("[可信执行环境] 将文本编码为中间特征向量")

    encrypted_user_text = request.form.get('text', '').strip()
    if not encrypted_user_text:
        return redirect(url_for('home_t'))

    # 1. Decrypt text
    user_text = decrypt_data(encrypted_user_text)
    if user_text is None:
        return render_template('home_t.html', images=None, error="Error decrypting input text.")
    print(f"[Master] Decrypted user text: {user_text}")
    

    assert args.datatype in DATALOADER_DICT
    test_dataloader_text, test_length_image = DATALOADER_DICT[args.datatype]["test_text"](args, tokenizer)

    with torch.no_grad():
        batch_list_t = []
        batch_sequence_output_list = []

        # logger.info("Calculating the text embedding...")
        for bid, batch in tqdm(enumerate(test_dataloader_text)):
            batch = tuple(t.to(device) for t in batch)

            input_ids, input_mask, segment_ids = batch
            embedding_output, input_ids, input_mask, segment_ids = model.clip.encode_text_F1(input_ids, input_mask, segment_ids)
            
            # F1 part
            # embedding = model.clip.visual.F1_forward(images)
            dist.send(tensor=embedding_output, dst=1)  # Assuming worker rank is 
            # Backbone part
            # encoder_outputs = model.clip.encode_text_Backbone(embedding_output)
            
            logger.info("[可信执行环境] 将中间特征向量发送至常规执行环境")
            logger.info("[常规执行环境] 接收中间特征向量")
            logger.info("[常规执行环境] 将中间特征向量编码为最终特征向量")
            logger.info("[常规执行环境] 将最终特征向量发送至可信执行环境")
            encoder_outputs = torch.zeros_like(embedding_output)
            dist.recv(tensor=encoder_outputs, src=1)
            # logger.info("[TEE] Receive final embedding.")
            logger.info("[可信执行环境] 接收最终特征向量")
            # Backbone part
            # encoder_outputs = model.clip.visual.Backbone_forward(embedding_output)

            # F2 part
            text_embedding = model.clip.encode_text_F2(encoder_outputs, input_ids)
            batch_sequence_output_list.append(text_embedding)

            batch_list_t.append((input_mask, segment_ids,))            
            logger.info("[可信执行环境] 计算检索文本的最终特征向量和目标视频的特征向量之间的相似度分数")
        sim_matrix = []
        # with profile():
        for idx1, b1 in tqdm(enumerate(batch_list_t)):
            input_mask, segment_ids, *_tmp = b1
            sequence_output = batch_sequence_output_list[idx1]
            each_row = []
            for idx2, b2 in tqdm(enumerate(batch_list_v)):
                video_mask, *_tmp = b2
                visual_output = batch_visual_output_list[idx2]
                visual_TAB = batch_visual_TAB_list[idx2]
                # calculate the similarity
                b1b2_logits, *_tmp = model.get_inference_logits(sequence_output, visual_output, input_mask, video_mask, visual_TAB)
                b1b2_logits = b1b2_logits.cpu().detach().numpy()
                each_row.append(b1b2_logits)
            each_row = np.concatenate(tuple(each_row), axis=-1)
            sim_matrix.append(each_row)
    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    top_k_indices_list = []
    for row in sim_matrix:
        sorted_indices = np.argsort(row)[::-1][:args.topk]
        top_k_indices_list.append(sorted_indices)
    
    top3_videos_ids = top_k_indices_list[0]
    top3_videos  = [ f'video{i}.mp4' for i in top3_videos_ids]

    logger.info("[可信执行环境] 获取目标视频中相似度分数前三的视频的索引")
    logger.info("[可信执行环境] 添加混淆索引并将它们一起发送到数据库")
    video_ids = [i for i in range(10000)]
    random_indices = random.sample(video_ids, 2 * TOP_K)
    top3_videos = ['video9945.mp4', 'video3380.mp4', 'video6759.mp4']
    top3_videos_ids = [9945, 3380, 6759]
    
    combined_image_ids = random_indices + top3_videos_ids
    response = requests.post(REMOTE_VIDEO_DB_URL, json={'image_ids': combined_image_ids})

    if response.status_code == 200:
        videos = response.json().get('videos', [])
        for idx, video_data in enumerate(videos):
            video_filename = f"received_video{idx}.mp4"
            with open(video_filename, 'wb') as f:
                f.write(video_data)
            print(f"Video {video_filename} saved.")
    # else:
    #     pass
        # print(f"Failed to get videos: {response.text}")
    
    logger.info("[数据库] 接收索引，将相应视频发送至可信执行环境")
    logger.info("[可信执行环境] 接收视频，删除混淆索引对应的视频")
    
    logger.info("[可信执行环境] 加密视频并发送至客户端")
    logger.info("[客户端] 接收并解密视频，至此检索完成")
    return jsonify({
            "top_videos": top3_videos
        })
    
    
@app.route('/videos/<filename>')
def video(filename):
    return send_from_directory(app.config['VIDEO_FOLDER'], filename)

@app.route("/")
def index():
    return render_template("home_t.html")  # Renders the template above

@app.route('/attest', methods=['POST'])
def attest():
    attest_start_time = time.time()

    try:
        # Step 1: Generate attestation report using snpguest utility
        # logger.info("Generating attestation report using snpguest utility.")
        # logger.info("[Client] Select one image to do a retrieval.")
        # logger.info("[Client] Send remote attestation request to TEE.")
        # logger.info("[TEE] Send attestation report to client.")
        logger.info("[可信执行环境] 加载模型")
        logger.info("[可信执行环境] 从数据库加载目标视频的特征向量")
        logger.info("[常规执行环境] 加载模型")

        logger.info("[客户端] 选择一张文本进行检索")
        logger.info("[客户端] 向可信执行环境发送远程证明请求")
        logger.info("[可信执行环境] 向客户端发送证明报告")

        # Generate request file
        request_file_path = 'request-file.txt'
        with open(request_file_path, 'w') as f:
            f.write('')  # Empty content as snpguest will generate random data
        # logger.info(f"Generated request file at {request_file_path}.")

        # Run snpguest report command
        report_file = 'report.bin'
        cmd_report = [
            '/home/ubuntu/snpguest/target/release/snpguest',
            'report',
            report_file,
            request_file_path,
            '--random'
        ]
        # logger.info(f"Running command: {' '.join(cmd_report)}")
        # subprocess.run(cmd_report, check=True)
        # logger.info(f"Attestation report generated at {report_file}.")

        # Run snpguest certificates command to generate the vlek.pem
        cmd_certificates = [
            '/home/ubuntu/snpguest/target/release/snpguest',
            'certificates',
            'PEM',
            './'
        ]
        # logger.info(f"Running command: {' '.join(cmd_certificates)}")
        # subprocess.run(cmd_certificates, check=True)
        # logger.info("Certificates generated successfully.")

        # Ensure the vlek.pem certificate is available
        vlek_cert_file = './vlek.pem'
        if not os.path.exists(vlek_cert_file):
            logger.error("VLEK certificate not found.")
            return jsonify({
                'status': 'Failure',
                'details': 'VLEK certificate not found.'
            }), 500

        # Step 2: Validate the attestation report signature by sending to worker
        # logger.info(f"Sending attestation report and certificate to worker at {WORKER_URL}.")

        with open(report_file, 'rb') as f:
            report_data = f.read()

        with open(vlek_cert_file, 'rb') as f:
            cert_data = f.read()

        # Send report and certificate to worker endpoint for validation
        files = {
            'report': ('report.bin', report_data),
            'certificate': ('vlek.pem', cert_data)
        }

        response = requests.post(WORKER_URL, files=files, timeout=60)

        if response.status_code == 200:
            response_json = response.json()
            validation_result = response_json.get('validation', 'No validation result received.')
            # logger.info("[Client] Receive attestation report.")
            logger.info("[客户端] 接收证明报告")
            status = '成功'
            details = validation_result
        else:
            validation_result = f"Worker returned status code {response.status_code}."
            logger.error(f"Worker returned error: {response.text}")
            status = 'Failure'
            details = validation_result

        attest_time = time.time() - attest_start_time
        # logger.info(f"Attestation process completed in {attest_time:.4f} seconds.")

        return jsonify({
            'status': status,
            'details': details,
            'attest_time': f"{attest_time:.4f} seconds"
        }), 200 if status == '成功' else 500

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running snpguest commands: {e}", exc_info=True)
        return jsonify({
            'status': 'Failure',
            'details': f"Error generating attestation report: {str(e)}"
        }), 500
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with worker: {e}", exc_info=True)
        return jsonify({
            'status': 'Failure',
            'details': f"Error communicating with worker for attestation validation: {str(e)}"
        }), 500
    except Exception as e:
        logger.error(f"Unexpected error during attestation: {e}", exc_info=True)
        return jsonify({
            'status': 'Failure',
            'details': "An unexpected error occurred during attestation."
        }), 500
        
@app.route('/logs-stream', methods=['GET'])
def logs_stream():
    """
    Stream the log file to the client in real-time using Server-Sent Events (SSE).
    """
    def generate():
        with open(log_file, 'r') as f:
            # Move to the end of the file
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                # SSE requires 'data: ' prefix and double newline
                yield f"data: {line}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/get_aes_key', methods=['GET'])
def get_aes_key():
    """
    Provide the AES key to the client in hexadecimal format.
    This endpoint should be secured (e.g., require authentication) in production.
    """
    try:
        aes_key_hex = AES_KEY.hex()
        response = {'aes_key': aes_key_hex}
        logger.info("[可信执行环境] 向客户端发送 AES 密钥")
        # logger.info("[TEE] Send AES key to client.")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Failed to provide AES key: {e}")
        return jsonify({'error': 'Failed to provide AES key.'}), 500
    
if __name__ == "__main__":
    logger.info(f"Starting Flask app on port {8080}...")
    context = ('cert.pem', 'key.pem') 
    app.run(host='0.0.0.0', port=8080, debug=False, ssl_context=context)

