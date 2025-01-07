# Are They the Same? Exploring Visual Correspondence Shortcomings of Multimodal LLMs

### [Paper](https://arxiv.org/abs/***) | [Project Page](https://zhouyiks.github.io/***) | [MMVM Benchmark](https://huggingface.co/zhouyik/MMVMBench) | [Huggingface](https://huggingface.co/zhouyik/colva_internvl2_4b)

![Teaser](imgs/benchmark_00.png)

We build the evaluation tool **MMVMEvalKit** based on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

Before running evaluation: 

1. Clone down our **MMVMEvalKit**.
2. Download the `match_bench.zip` and `mllm_match_eval_full.tsv` from [here](https://huggingface.co/zhouyik/MMVMBench) and put them under the **MMVMEvalKit** folder and `match_bench.zip`
3. Evironment requirements follow that of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
4. Note: Your OpenAI API Key should be setted in the **.env** file:
```
# OpenAI API
OPENAI_API_KEY=
OPENAI_API_BASE=
```

To evaluate the existing MLLMs on MMVM benchmark, e.g. InternVL2-2B, run
```
python run.py --data MMatch --model InternVL2-2B --verbose
```

To evaluate CoLVA-InternVL2-4B on MMVM benchmark, download the pretrained weights from [here](https://huggingface.co/zhouyik/colva_ablation) and run
```
python run.py --data MMatch --model colva_internvl2_4b --verbose
```

To evaluate CoLVA-Qwen2VL-2B on MMVM benchmark, download the pretrained weights from [here](https://huggingface.co/zhouyik/colva_ablation) and run
```
python run.py --data MMatch --model colva_qwen2vl_2b --verbose
```

To evaluate CoLVA-Qwen2VL-7B on MMVM benchmark, download the pretrained weights from [here](https://huggingface.co/zhouyik/colva_ablation) and run
```
python run.py --data MMatch --model colva_qwen2vl_7b --verbose
```