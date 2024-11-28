import pdb
import sys
videollama_dir = "./Video-LLaMA"
sys.path.append(videollama_dir)
import ast
import os
import math
import base64
import traceback
from io import BytesIO

import re
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import gradio as gr
from PIL import Image

from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.processors.video_processor import *
from video_llama.runners import *
from video_llama.tasks import *
from video_llama.models.ImageBind.data import load_and_transform_audio_data, load_and_transform_vision_data
from omegaconf import OmegaConf

class videollama_evaluation(nn.Module):
    def __init__(self, model_path="./avlm_model_weight/Video-LLaMA-Series"):
        super().__init__()  # Ensure this is called first
        self.model_path = model_path
        self.cfg = OmegaConf.load("./Video-LLaMA/eval_configs/video_llama_eval_withaudio.yaml")
        self.runner_config = Config.build_runner_config(self.cfg)
        self.model_config = Config.build_model_config(self.cfg)
        self.dataset_config = Config.build_dataset_config(self.cfg)
        self.cfg = OmegaConf.merge(self.runner_config, self.model_config, self.dataset_config)

        self.gpu_id = 0
        self.cfg.device_8bit = self.gpu_id
        self.model_cls = registry.get_model_class(self.cfg.model.arch)
        self.model = self.model_cls.from_config(self.cfg.model).to('cuda:{}'.format(self.gpu_id)).eval()

        self.vis_processor_cfg = self.dataset_config.datasets.webvid.vis_processor.train
        self.vis_processor = registry.get_processor_class(self.vis_processor_cfg.name).from_config(self.vis_processor_cfg)
        self.chat = Chat(self.model, self.vis_processor)

        self.question_prompt = "Answer with the option's letter from the given choices directly."
    

    def evaluate_image_audio_text(self, image_path, audio_path, question, options):
        # image_path for list of image path
        # audio_path for list of audio path
        # question for question
        # options for [option_A, option_B, option_C, option_D]

        content = []
        # fixed image token and no audio in text
        for index in range(1, 1 + len(image_path)):
            question = question.replace(f"[img{index}]", "<Video><ImageHere></Video>") 
        for index in range(1, 1 + len(audio_path)):
            question = question.replace(f'[audio{index}]', '<ImageHere>')
        option_text = "A:" + options[0] + "\n" + "B:" + options[1] + "\n" + "C:" + options[2] + "\n" + "D:" + options[3] + "\n" 
        text = question + "\n" + option_text + self.question_prompt
        
        img_list = []
        conv = conv_llava_llama_2.copy()
        conv.system = ""
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)

        for image in image_path:
            current_image = load_and_transform_vision_data([image], "cpu").to(self.model.device).permute(1,0,2,3).unsqueeze(0)
            image_emb, _ = self.model.encode_videoQformer_visual(current_image)
            img_list.append(image_emb)
        for audio in audio_path:
            current_audio = load_and_transform_audio_data([audio], "cpu", clips_per_video=8).to(self.model.device)
            audio_emb,_  = self.model.encode_audioQformer(current_audio)
            img_list.append(audio_emb)
        
        response, _ = self.chat.answer(conv, img_list)

        return response

    def evaluate_video_audio_text(self, video_path, audio_path, question, options):
        content = []
        # fixed image token and no audio in text
        for index in range(1, 1 + len(video_path)):
            question = question.replace(f"[video{index}]", "<Video><ImageHere></Video>") 
        for index in range(1, 1 + len(audio_path)):
            question = question.replace(f'[audio{index}]', '<ImageHere>')
        option_text = "A:" + options[0] + "\n" + "B:" + options[1] + "\n" + "C:" + options[2] + "\n" + "D:" + options[3] + "\n" 
        text = question + "\n" + option_text + self.question_prompt
        
        img_list = []
        conv = conv_llava_llama_2.copy()
        conv.system = ""
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)

        for video in video_path:
            current_video, msg = load_video(video_path = video, n_frms=8, height=224, width=224, sampling ="uniform", return_msg = True)
            current_video = self.vis_processor.transform(current_video).to(self.model.device).unsqueeze(0)
            image_emb, _ = self.model.encode_videoQformer_visual(current_video)
            img_list.append(image_emb)
        for audio in audio_path:
            current_audio = load_and_transform_audio_data([audio], "cpu", clips_per_video=8).to(self.model.device)
            audio_emb,_  = self.model.encode_audioQformer(current_audio)
            img_list.append(audio_emb)
        
        response, _ = self.chat.answer(conv, img_list)

        return response