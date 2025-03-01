import base64
import os
import logging
import numpy as np
import requests
import csv
import torch
from flask import Flask, request, jsonify, send_file
from tqdm import tqdm
import json
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from urllib3.exceptions import InsecureRequestWarning
import sys
import pickle
import io
import time 

import flask.cli
flask.cli.show_server_banner = lambda *args: None

import sys 
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(MAIN_DIR)

# Logger setup
logger = logging.getLogger(__name__)
aesgcm = None

# Constants
IMAGE_FEAT_PATH = os.path.join(MAIN_DIR, "database/valid_imgs.img_feat.jsonl")
IMAGE_DATA_PATH = os.path.join(MAIN_DIR, "database/valid_imgs.tsv")
REMOTE_AES_KEY_URL = "https://18.116.182.172:8080/get_aes_key"  # Replace with the actual remote server URL


import warnings

warnings.simplefilter('ignore', InsecureRequestWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*development server.*")

# Initialize Flask app
app = Flask(__name__)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class Database:
    def __init__(self):
        self.batch_list_v = []
        self.batch_visual_output_list = []
        self.serialized_batch_list_v = []
        self.serialized_batch_visual_output_list = []
        self.load_video_features()
        # Fetch AES key from remote server
        # global aesgcm
        # aes_key = self.fetch_aes_key()
        # aesgcm = AESGCM(aes_key)
        # if not aesgcm:
        #     logger.error("Failed to access AES key")

    def load_video_features(self):
        logger.info("Logging video features")
        video_cache_file = os.path.join(MAIN_DIR, './data/video_embeddings.pkl')

        with open(video_cache_file, 'rb') as f:
            self.batch_list_v, self.batch_visual_output_list = CPU_Unpickler(f).load()
        
        self.serialized_batch_list_v = [
            [(tensor.tolist()) for tensor in batch]  # Convert each tensor to a list
            for batch in self.batch_list_v
        ]
        
        self.serialized_batch_visual_output_list = [ tensor.tolist()  for tensor in self.batch_visual_output_list ]

    
    def fetch_aes_key(self):
        """
        Fetch the AES key from a remote server.
        """
        try:
            response = requests.get(REMOTE_AES_KEY_URL, verify=False)  # Replace with secure flag in production
            if response.status_code == 200:
                aes_key_hex = response.json().get('aes_key')
                return bytes.fromhex(aes_key_hex)
            else:
                logger.error("Failed to fetch AES key.")
        except Exception as e:
            logger.error(f"Error fetching AES key: {e}")
        return None
    
    def fetch_and_encrypt_images(self, image_ids):
        """
        Fetch and encrypt images based on image IDs.

        Args:
            image_ids (list): List of image IDs to fetch and encrypt.

        Returns:
            List of encrypted image data in base64 format.
        """
        encrypted_images = {}
        for img_id in image_ids:
            if img_id in self.image_id_to_data:
                try:
                    image_data_b64 = self.image_id_to_data[img_id]
                    encrypted_data = self.encrypt_image_bytes(image_data_b64)
                    if encrypted_data:
                        encrypted_images[img_id] = encrypted_data
                    else:
                        logger.error(f"Error encrypting image {img_id}.")
                except Exception as e:
                    logger.error(f"Error encrypting image {img_id}: {e}")
            else:
                logger.warning(f"Image ID {img_id} not found in database.")
        logger.info("Receive indexes, send images back to TEE.")
        return encrypted_images
    
    def encrypt_image_bytes(self, image_b64):
        """
        Encrypts image bytes using AES-GCM.

        Args:
            image_b64: Base64-encoded image data.

        Returns:
            Base64-encoded (nonce + ciphertext) or empty string if encryption fails.
        """
        try:
            raw_image_bytes = base64.b64decode(image_b64.encode("utf-8"))
            nonce = os.urandom(12)
            ciphertext = aesgcm.encrypt(nonce, raw_image_bytes, None)
            combined = nonce + ciphertext
            encrypted_b64 = base64.b64encode(combined).decode('utf-8')
            return encrypted_b64
        except Exception as e:
            logger.error(f"Error encrypting image bytes: {e}")
            return ""
    
    def get_video_features(self):
        """
        Return image IDs and feature tensors.
        """
        # Mock retrivel: please note that it will take some time to send over the embeddings
        time.sleep(3)
        return {}
        return jsonify({
            'batch_list_v': self.serialized_batch_list_v,
            'batch_visual_output_list': self.serialized_batch_visual_output_list
        })
        
 

# Initialize the Database instance
database = Database()



@app.route("/get_video_features", methods=["GET"])
def get_image_features():
    return database.get_video_features()


@app.route("/get_videos", methods=["POST"])
def get_videos():        
    # Mock retrivel: please note that it will take some time to send over the embeddings
    # Please note that this code down below is not tested
    # Get the list of image IDs from the request
    # image_ids = request.json.get('image_ids', [])
    time.sleep(5)
    return {}
    # Hard-coded video path, TODO change to your own data path 
    VIDEO_FOLDER = '/home/ubuntu/video-data/'
    # video_files = []
    # for image_id in image_ids:
    #     video_path = os.path.join(VIDEO_FOLDER, f"video{image_id}.mp4")
    #     if os.path.exists(video_path):
    #         video_files.append(video_path)
    
    # if not video_files:
    #     return "No videos found for the given IDs", 404

    # # Send videos as binary data
    # video_responses = []
    # for video_path in video_files:
    #     with open(video_path, 'rb') as f:
    #         video_responses.append(f.read())

    # return {
    #     "videos": [video_data for video_data in video_responses]
    # }, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
