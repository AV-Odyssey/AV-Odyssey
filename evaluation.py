import json
import os
import torch
from tqdm import tqdm
import pdb
import numpy as np
import argparse
import pyarrow.parquet as pq

vlm_model_list = ["Internvl2_8B", "Qwen2vl", "minicpm_v", "blip3", "vila" ]
alm_model_list = ["Qwen2audio", "Qwenaudio", "SALMONN", "typhoonaudio"]
avlm_model_list = ["unified-io-large", "unified-io-xl", "unified-io-xxl", "onellm", "pandagpt", "videollama", "videollama2", "anygpt", "nextgpt", "gemini", "gpt", "vita", "reka"]
model_type_list = ["audio-visual-llm", "visionllm", "audiollm"]

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

def vlm_model_select(model_name):
    raise NotImplementedError("Not implement vision language models.")

def alm_model_select(model_name):
    raise NotImplementedError("Not implement audio language models.")

def avlm_model_select(model_name):
    if model_name == "videollama":   
        from avlm_model.videollama.videollama import videollama_evaluation as model_build
        model_path = "./avlm_model_weight/Video-LLaMA-Series"
        model = model_build(model_path=model_path)
    
    return model

def read_parquet_file(file_path):
        table = pq.read_table(file_path)
        return table

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--model',default="", type=str, help ='input files')
    args = parser.parse_args()

    
    model_name = args.model
    
    if model_name in vlm_model_list: # vision language model
        model = vlm_model_select(model_name)
    elif model_name in alm_model_list: # audio language model
        model = alm_model_select(model_name)
    elif model_name in avlm_model_list: # audio-visual language model
        model = avlm_model_select(model_name)

    file_path = [
                './data/av_odyssey_part1.parquet',
                './data/av_odyssey_part2.parquet',
                './data/av_odyssey_part3.parquet',
                './data/av_odyssey_part4.parquet',
                './data/av_odyssey_part5.parquet',
                './data/av_odyssey_part6.parquet'
                ]
    question_type_dict = {}    
    for par_file in file_path:
        table = read_parquet_file(par_file)
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
    question_id_list = [i for i in range(1, 27)] 

    all_evaluation_results = []
    for current_question_id in question_id_list:
        current_json_data = question_type_dict[current_question_id]
        task_name = 'task' + str(current_question_id)

        with torch.no_grad():
            if model_name in vlm_model_list: # vision language model
                evaluation_result = vlm_task_process(model, current_json_data)
            elif model_name in alm_model_list: # audio language model
                evaluation_result = alm_task_process(model, current_json_data)
            elif model_name in avlm_model_list: # audio-visual language model
                evaluation_result = avlm_task_process(model, current_json_data)

        # clean the answer, following MMMU (https://github.com/MMMU-Benchmark/MMMU)
        cleaned_evaluation_data = []
        for data, prediction in zip(current_json_data, evaluation_result):
            option_list = {'A': data['option_A'], 'B': data['option_B'], 'C': data['option_C'], 'D': data['option_D']}
            answer = parse_multi_choice_response(prediction['prediction'], ['A', 'B', 'C', 'D'], option_list)
            prediction['prediction'] = answer
            cleaned_evaluation_data.append(prediction)

        all_evaluation_results = all_evaluation_results + cleaned_evaluation_data


    with open('avlm_results/'+'_'+model_name+'.jsonl', 'w') as f:
        for item in all_evaluation_results:
            item.pop("answer")
            f.write(json.dumps(item) + '\n')
    
