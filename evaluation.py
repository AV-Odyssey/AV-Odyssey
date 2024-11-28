import json
import os
import torch
from tqdm import tqdm
import pdb
import numpy as np

def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " " # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'{choice}' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices: # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        # pred_index = random.choice(all_choices)
        pred_index = 'A'
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack: 
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index) # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else: # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index

def read_json(file_path):
    """读取 JSON 文件并返回数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return None
    except json.JSONDecodeError:
        print(f"文件 {file_path} 不是有效的 JSON 格式。")
        return None

def save_json(data, file_path):
    """将数据保存为 JSON 文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"数据已保存到 {file_path}")
    except Exception as e:
        print(f"保存数据时发生错误: {e}")

def vlm_task_process(model, json_data):
    result_data = []
    for current_data in tqdm(json_data):
        try:
            if "image" in current_data['data_type']:
                # for image
                prediction = model.evaluate_image_text(current_data['image_path'], current_data['audio_path'], current_data['question'], [current_data['option_A'], current_data['option_B'], current_data['option_C'], current_data['option_D']])
            if "video" in current_data['data_type']:
                # for video
                prediction = model.evaluate_video_text(current_data['video_path'], current_data['audio_path'], current_data['question'], [current_data['option_A'], current_data['option_B'], current_data['option_C'], current_data['option_D']])
        except Exception as e:
            prediction = "error!"
            print(e)

        print(prediction)
        result_data.append({
            "question_id": current_data['question_id'],
            "answer": current_data['answer'],
            "prediction": prediction
        })

    return result_data


def alm_task_process(model, json_data):
    result_data = []
    for current_data in tqdm(json_data):
        '''
        try:
            # for audio
            if "image" in current_data['data_type']:
                prediction = model.evaluate_image_audio_text(current_data['image_path'], current_data['audio_path'], current_data['question'], [current_data['option_A'], current_data['option_B'], current_data['option_C'], current_data['option_D']])
            if "video" in current_data['data_type']:
                prediction = model.evaluate_video_audio_text(current_data['video_path'], current_data['audio_path'], current_data['question'], [current_data['option_A'], current_data['option_B'], current_data['option_C'], current_data['option_D']])
            print(prediction)
            
        except Exception as e:
            prediction = "error!"
            print(e)
        '''
        if "image" in current_data['data_type']:
            prediction = model.evaluate_image_audio_text(current_data['image_path'], current_data['audio_path'], current_data['question'], [current_data['option_A'], current_data['option_B'], current_data['option_C'], current_data['option_D']])
        if "video" in current_data['data_type']:
            prediction = model.evaluate_video_audio_text(current_data['video_path'], current_data['audio_path'], current_data['question'], [current_data['option_A'], current_data['option_B'], current_data['option_C'], current_data['option_D']])
        
        print(prediction)
        result_data.append({
                "question_id": current_data['question_id'],
                "answer": current_data['answer'],
                "prediction": prediction
            })
    return result_data

def avlm_task_process(model, json_data):
    result_data = []
    for current_data in tqdm(json_data):
        '''
        try:
            if "image" in current_data['data_type']:
                # for image and audio
                prediction = model.evaluate_image_audio_text(current_data['image_path'], current_data['audio_path'], current_data['question'], [current_data['option_A'], current_data['option_B'], current_data['option_C'], current_data['option_D']])
            if "video" in current_data['data_type']:
                # for video and audio
                prediction = model.evaluate_video_audio_text(current_data['video_path'], current_data['audio_path'], current_data['question'], [current_data['option_A'], current_data['option_B'], current_data['option_C'], current_data['option_D']])
        except Exception as e:
            prediction = "error!"
            print(e) 
        '''
        if "image" in current_data['data_type']:
            # for image and audio
            prediction = model.evaluate_image_audio_text(current_data['image_path'], current_data['audio_path'], current_data['question'], [current_data['option_A'], current_data['option_B'], current_data['option_C'], current_data['option_D']])
        if "video" in current_data['data_type']:
            # for video and audio
            prediction = model.evaluate_video_audio_text(current_data['video_path'], current_data['audio_path'], current_data['question'], [current_data['option_A'], current_data['option_B'], current_data['option_C'], current_data['option_D']])
        
        
        print(prediction)
        result_data.append({
            "question_id": current_data['question_id'],
            "answer": current_data['answer'],
            "prediction": prediction
        })

    return result_data

def vlm_model_select(model_name, image_folder=None, video_folder=None):
    if model_name == "Internvl2_8B":
        from vlm_model.internvl_2 import Internvl2_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/vlm_model_weight/InternVL2-8B"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder)
    elif model_name == "Qwen2vl":   # latest transformers
        from vlm_model.qwen2_vl import Qwen2vl_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/vlm_model_weight/Qwen2-VL-7B-Instruct"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder)
    elif model_name == "minicpm_v":
        from vlm_model.minicpm_v import Minicpm_v_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/vlm_model_weight/MiniCPM-V-2_6"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder)
    elif model_name == "llava": # not implement
        from vlm_model.llava import llava_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/vlm_model_weight/llava-onevision-qwen2-7b-ov"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder)
    elif model_name == "molmo":  # not implement
        from vlm_model.molmo import Molmo_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/vlm_model_weight/Molmo-7B-D-0924"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder)
    elif model_name == "blip3":  # transformers==4.41.1
        from vlm_model.blip3 import BLIP3_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/vlm_model_weight/blip3_weight/xgen-mm-phi3-mini-base-r-v1.5.pt"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder)
    elif model_name == "vila":  # transformers==4.41.1
        from vlm_model.vila import vila_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/vlm_model_weight/Llama-3-VILA1.5-8B"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder)
    
    return model

def alm_model_select(model_name, audio_folder=None):
    if model_name == "Qwen2audio":   # latest transformers
        from alm_model.qwen2_audio import Qwen2audio_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/alm_model_weight/Qwen2-Audio-7B-Instruct"
        model = model_build(model_path=model_path, audio_folder=audio_folder)
    elif model_name == "Qwenaudio":   # latest transformers
        from alm_model.qwen_audio import Qwenaudio_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/alm_model_weight/Qwen-Audio-Chat"
        model = model_build(model_path=model_path, audio_folder=audio_folder)
    elif model_name == "SALMONN":   
        from alm_model.salmonn import SALMON_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/alm_model_weight/SALMONN-7B/salmonn_7b_v0.pth"
        model = model_build(model_path=model_path, audio_folder=audio_folder)
    elif model_name == "typhoonaudio":  
        from alm_model.typhoon_audio import typhoon_audio_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/alm_model_weight/llama-3-typhoon-v1.5-8b-audio-preview"
        model = model_build(model_path=model_path, audio_folder=audio_folder)
    elif model_name == "GAMA":   
        from alm_model.GAMA import GAMA_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/alm_model_weight/GAMA/Llama-2-7b-chat-hf-qformer"
        model = model_build(model_path=model_path, audio_folder=audio_folder)
    
    return model

def avlm_model_select(model_name, image_folder=None, video_folder=None, audio_folder=None):
    if model_name == "unified-io-large":   
        from avlm_model.unio2_large import Unio2_large_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/avlm_model_weight/uio2-large"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder, audio_folder=audio_folder)
    elif model_name == "unified-io-xl":   
        from avlm_model.unio2_xl import Unio2_xl_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/avlm_model_weight/uio2-xl"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder, audio_folder=audio_folder)
    elif model_name == "unified-io-xxl":   
        from avlm_model.unio2_xxl import Unio2_xxl_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/avlm_model_weight/uio2-xxl"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder, audio_folder=audio_folder)
    elif model_name == "onellm":   
        from avlm_model.onellm import onellm_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/avlm_model_weight/OneLLM-7B"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder, audio_folder=audio_folder)
    elif model_name == "videollama":   
        from avlm_model.videollama import videollama_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/avlm_model_weight/Video-LLaMA-Series"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder, audio_folder=audio_folder)
    elif model_name == "videollama2":   
        from avlm_model.videollama2 import videollama2_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/avlm_model_weight/VideoLLaMA2.1-7B-AV"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder, audio_folder=audio_folder)
    elif model_name == "pandagpt":   
        from avlm_model.pandagpt import pandagpt_evaluation as model_build
        model_path = "./audio_visual_model_evaluation/avlm_model_weight/pandagpt_7b_max_len_1024/pytorch_model.pt"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder, audio_folder=audio_folder)
    elif model_name == "vita":   # must used as CUDA_VISIBLE_DEVICES=0,1 python3.10 evaluation.py
        from avlm_model.vita import vita_evaluation as model_build    
        model_path = "./audio_visual_model_evaluation/avlm_model_weight/VITA/VITA_ckpt"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder, audio_folder=audio_folder)
    elif model_name == "videosalmonn":   
        from avlm_model.video_salmonn import videosalmonn_evaluation as model_build    
        model_path = "./audio_visual_model_evaluation/avlm_model_weight/video_salmonn_weight"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder, audio_folder=audio_folder)
    elif model_name == "mini_omni2":   
        from avlm_model.mini_omni2 import mini_omni2_evaluation as model_build    
        model_path = "./audio_visual_model_evaluation/avlm_model_weight/mini-omni2"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder, audio_folder=audio_folder)
    elif model_name == "crema":   
        from avlm_model.crema import crema_evaluation as model_build    
        model_path = "./audio_visual_model_evaluation/avlm_model_weight/CREMA"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder, audio_folder=audio_folder)
    elif model_name == "nextgpt":   
        from avlm_model.nextgpt import nextgpt_evaluation as model_build    
        model_path = "./audio_visual_model_evaluation/avlm_model_weight/nextgpt_7b_tiva_v0"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder, audio_folder=audio_folder)
    elif model_name == "vita":   
        from avlm_model.vita import VITA_evaluation as model_build    
        model_path = "./audio_visual_model_evaluation/avlm_model_weight/VITA"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder, audio_folder=audio_folder)
    elif model_name == "macaw":   
        from avlm_model.macaw import Macaw_evaluation as model_build    
        model_path = "./audio_visual_model_evaluation/avlm_model_weight/VITA"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder, audio_folder=audio_folder)
    elif model_name == "anygpt":   
        from avlm_model.anygpt import anygpt_evaluation as model_build    
        model_path = "./audio_visual_model_evaluation/avlm_model_weight/anygpt_weight"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder, audio_folder=audio_folder)
    elif model_name == "gemini":   
        from avlm_model.gemini import gemini_evaluation as model_build    
        model_path = "./"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder, audio_folder=audio_folder)
    elif model_name == "favor":   
        from avlm_model.favor import favor_evaluation as model_build    
        model_path = "./"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder, audio_folder=audio_folder)
    elif model_name == "reka":   
        from avlm_model.reka import reka_evaluation as model_build    
        model_path = "./"
        model = model_build(model_path=model_path, image_folder=image_folder, video_folder=video_folder, audio_folder=audio_folder)
    
    return model

if __name__ == "__main__":

    task_folder = "./audio_visual_model_evaluation/MLLM_evaluation/data/SSBench_v1"
    json_file = "./audio_visual_model_evaluation/MLLM_evaluation/data/SSBench_v1/ssbench.json"
    json_data = read_json(json_file)
    # print(json_data)
    # print(json_data[:10])

    import pyarrow.parquet as pq

    def read_parquet_file(file_path):
        # 读取Parquet文件
        table = pq.read_table(file_path)
        return table

    # 定义Parquet文件路径
    file_path = [
                './audio_visual_model_evaluation/data/AV_Odyssey_Bench/av_odyssey_part1.parquet',
                 './audio_visual_model_evaluation/data/AV_Odyssey_Bench/av_odyssey_part2.parquet',
                 './audio_visual_model_evaluation/data/AV_Odyssey_Bench/av_odyssey_part3.parquet',
                 './audio_visual_model_evaluation/data/AV_Odyssey_Bench/av_odyssey_part4.parquet',
                 './audio_visual_model_evaluation/data/AV_Odyssey_Bench/av_odyssey_part5.parquet',
                 './audio_visual_model_evaluation/data/AV_Odyssey_Bench/av_odyssey_part6.parquet'
                 ]
    question_type_dict = {}

    for par_file in file_path:

        # 读取Parquet文件
        table = read_parquet_file(par_file)
        # 将PyArrow Table转换为Pandas DataFrame（如果需要）
        df = table.to_pandas()

        for index, row in df.iterrows():

            row_dict = row.to_dict()
            question_type_id = row_dict.get('question_type_id')
            row_dict['subpart'] = row_dict.pop('subfield')

            row_dict['image_path'] = [row_dict['image_1'], row_dict['image_2'],  row_dict['image_3'], row_dict['image_4']] if row_dict['image_2'] else [row_dict['image_1']]
            row_dict['audio_path'] = [row_dict['audio_1'], row_dict['audio_2'] , row_dict['audio_3'], row_dict['audio_4']] if row_dict['audio_2'] else [row_dict['audio_1']]
            row_dict['option_A'] = row_dict['options'][0]
            row_dict['option_B'] = row_dict['options'][1]
            row_dict['option_C'] = row_dict['options'][2]
            row_dict['option_D'] = row_dict['options'][3]
            row_dict['video_path'] = [row_dict.pop('video_1')]
            
            row_dict.pop('options')

            if question_type_id not in question_type_dict:
                question_type_dict[question_type_id] = []
            question_type_dict[question_type_id].append(row_dict)


    
    # question_type_dict = {}
    # for item in json_data:
    #     question_type_id = item.get('question_type_id')
    #     if question_type_id not in question_type_dict:
    #         question_type_dict[question_type_id] = []
    #     question_type_dict[question_type_id].append(item)

    vlm_model_list = ["Internvl2_8B", "Qwen2vl", "minicpm_v", "blip3", "vila" ]
    alm_model_list = ["Qwen2audio", "Qwenaudio", "SALMONN", "typhoonaudio"]
    avlm_model_list = ["unified-io-large", "unified-io-xl", "unified-io-xxl", "onellm", "pandagpt", "videollama", "videollama2", "anygpt", "nextgpt", "gemini", "gpt", "vita", "reka"]
    model_type_list = ["audio-visual-llm", "visionllm", "audiollm"]
    question_id_list = [i for i in range(1, 27)]

    model_type = model_type_list[0] # audio-visual-llm, visionllm, audiollm
    # current_model = avlm_model_list[0]
    current_model_list = [avlm_model_list[5]]
    for current_model in current_model_list:
        if model_type == 'visionllm':
            # for vlm
            model = vlm_model_select(current_model, image_folder = task_folder, video_folder = task_folder)
        elif model_type == 'audiollm':
            # for alm
            model = alm_model_select(current_model, audio_folder = task_folder)
        elif model_type == 'audio-visual-llm':
            # for avlm
            model = avlm_model_select(current_model, image_folder = task_folder, video_folder = task_folder, audio_folder = task_folder)


        all_evaluation_results = []
        for current_question_id in question_id_list:
            current_json_data = question_type_dict[current_question_id]
            task_name = 'task' + str(current_question_id)

            with torch.no_grad():
                if model_type == 'visionllm':
                    evaluation_result = vlm_task_process(model, current_json_data)
                    result_save_path = "./audio_visual_model_evaluation/MLLM_evaluation/code/new_avlm_results/" + task_name + "_" + current_model + ".json"
                elif model_type == 'audiollm':
                    evaluation_result = alm_task_process(model, current_json_data)
                    result_save_path = "./audio_visual_model_evaluation/MLLM_evaluation/code/new_avlm_results/" + task_name + "_" + current_model + ".json"
                elif model_type == 'audio-visual-llm':
                    evaluation_result = avlm_task_process(model, current_json_data)
                    result_save_path = "./audio_visual_model_evaluation/MLLM_evaluation/code/new_avlm_results/" + task_name + "_" + current_model + ".json"
            
            # clean the answer
            cleaned_evaluation_data = []
            for data, prediction in zip(current_json_data, evaluation_result):
                option_list = {'A': data['option_A'], 'B': data['option_B'], 'C': data['option_C'], 'D': data['option_D']}
                answer = parse_multi_choice_response(prediction, ['A', 'B', 'C', 'D'], option_list)
                prediction['prediction'] = answer
                cleaned_evaluation_data.append(prediction)

            all_evaluation_results = all_evaluation_results + cleaned_evaluation_data

            

            # save_json(evaluation_result, result_save_path)

        with open('./audio_visual_model_evaluation/MLLM_evaluation/code/new_avlm_results/'+'_'+current_model+'.jsonl', 'w') as f:
            for item in all_evaluation_results:
                item.pop("answer")
                f.write(json.dumps(item) + '\n')
        
