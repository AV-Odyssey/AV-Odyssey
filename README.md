# <img src="assets/logo.png" width="5%" />  AV-Odyssey: Can Your Multimodal LLMs Really Understand Audio-Visual Information?

![AVQA](https://img.shields.io/badge/Task-AVQA-red) 
![Multi-Modal](https://img.shields.io/badge/Task-Multi--Modal-red) 
![AV-Odyssey](https://img.shields.io/badge/Dataset-AV--Odyssey-blue)  
![Gemini](https://img.shields.io/badge/Model-Gemini-green) 
![Reka](https://img.shields.io/badge/Model-Reka-green) 
![GPT-4o](https://img.shields.io/badge/Model-GPT--4o-green)

Official repository for the paper "[AV-Odyssey: Can Your Multimodal LLMs Really Understand Audio-Visual Information?]()".

🌟 For more details, please refer to the project page with data examples: [https://av-odyssey.github.io/](https://av-odyssey.github.io/).

[[🌐 Webpage](https://av-odyssey.github.io/)] [[📖 Paper]()] [[🤗 AV-Odyssey Dataset](https://huggingface.co/datasets/AV-Odyssey/AV_Odyssey_Bench)] [[🤗 Deaftest Dataset](https://huggingface.co/datasets/AV-Odyssey/Deaftest_dataset)] [[🏆 Leaderboard](https://huggingface.co/spaces/AV-Odyssey/AV_Odyssey_Bench_Leaderboard)]


---

## 🔥 News
* **`2024.11.24`** 🌟 We release AV-Odyssey, the first-ever comprehensive evaluation benchmark to explore whether MLLMs really understand audio-visual information.


## 👀 About AV-Odyssey

Recently, multimodal large language models (MLLMs), such as GPT-4o, Gemini 1.5 Pro, and Reka Core, have expanded their capabilities to include vision and audio modalities. While these models demonstrate impressive performance across a wide range of audio-visual applications, our proposed **DeafTest** reveals that MLLMs often struggle with simple tasks humans find trivial: 1) determining which of two sounds is louder, and 2) determining which of two sounds has a higher pitch. Motivated by these observations, we introduce **AV-Odyssey Bench**. This benchmark encompasses **26** different tasks and **4,555** carefully crafted problems, each incorporating text, visual, and audio components. All data are **newly collected and annotated by humans**, not from any existing audio-visual dataset. AV-Odyssey Bench demonstrates three major features: 1. **Comprehensive** Audio Attributes; 2. **Extensive** Domains; 3. **Interleaved** Text, Audio, and Visual components.

<img src="/assets/intro.png" style="zoom:50%;" />

## 📐 Data Examples

Please refer to our project page https://av-odyssey.github.io/ for exploring more examples.


### 📍AV-Odyssey Bench
<div align="center">
  <img src="assets/demo-1.svg" width="100%" />
</div>


## 🔍 Dataset

**License**:
```
AV-Odyssey is only used for academic research. Commercial use in any form is prohibited.
The copyright of all videos belongs to the video owners.
If there is any infringement in AV-Odyssey, please email libohao1998@gmail.com and we will remove it immediately.
Without prior approval, you cannot distribute, publish, copy, disseminate, or modify AV-Odyssey in whole or in part. 
You must strictly comply with the above restrictions.
```

Please send an email to **[libohao1998@gmail.com](mailto:libohao1998@gmail.com)**. 🌟


## 🔮 Evaluation Pipeline

### Run Evaluation on AV-Odyssey

We now provide an example code for the evaluation of the [Video-Llama](https://github.com/DAMO-NLP-SG/Video-LLaMA) model. 

1. Download the AV-Odyssey data from [[🤗 AV-Odyssey Dataset](https://huggingface.co/datasets/AV-Odyssey/AV_Odyssey_Bench)] and put it into your specified folder. In our code, we download AV-Odyssey data into [data](https://github.com/AV-Odyssey/AV-Odyssey/tree/main/data).

2. Download the pre-trained weights of the evaluated model. In our code, we download [Video-Llama weight](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-Series) into [avlm_model_weight](https://github.com/AV-Odyssey/AV-Odyssey/tree/main/avlm_model_weight). You need to install all the required packages of the evaluated model.

Then, run

```
python evaluation.py --model videollama
```

We specify the model in evaluate.py.

The result will be collected into [avlm_results](https://github.com/AV-Odyssey/AV-Odyssey/tree/main/avlm_results).



## 🏆 Leaderboard

### Contributing to the AV-Odyssey Leaderboard

🚨 The [Leaderboard](https://huggingface.co/spaces/AV-Odyssey/AV_Odyssey_Bench_Leaderboard) for AV-Odyssey is continuously being updated, welcoming the contribution of your excellent MLLMs! 



## :black_nib: Citation

If you find our work helpful for your research, please consider citing our work.   

```bibtex

```
