import torch
import torch.amp
from transformers import AutoTokenizer, AutoConfig, AutoModel, CLIPImageProcessor
import warnings
from PIL import Image
from .base import BaseModel
from mmengine.model import BaseTTAModel, is_model_wrapper
from mmengine.logging import print_log
from mmengine.dist import get_dist_info
from mmengine.config import Config, ConfigDict
from peft import get_peft_model, prepare_model_for_kbit_training
from ..smp import *
from ..dataset import DATASET_TYPE
import pandas as pd
import string
import torch.distributed as dist
import torchvision.transforms as T
import transformers
import torch.nn as nn
from types import MethodType
import pycocotools.mask as mask_util
from typing import Any, Iterator, List, Union

from torchvision.transforms.functional import InterpolationMode
import re
import torch.nn.functional as F

from collections import OrderedDict, namedtuple


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VPT_CONTEXT_TOKEN = '<VPT_CONTEXT>'


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6, upscale=False):
    image = Image.open(image_file).convert('RGB')
    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# This function is used to split InternVL2-Llama3-76B
def split_model(model_name):
    import math
    device_map = {}
    num_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = num_gpus // world_size

    num_layers = {'InternVL2-8B': 32, 'InternVL2-26B': 48,
                  'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as 0.8 GPU.
    num_layers_per_gpu = math.ceil(num_layers / (num_gpus - 0.2))
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.8)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = rank + world_size * i
            layer_cnt += 1
    device_map['vision_model'] = rank
    device_map['mlp1'] = rank
    device_map['language_model.model.tok_embeddings'] = rank
    device_map['language_model.model.embed_tokens'] = rank
    device_map['language_model.output'] = rank
    device_map['language_model.model.norm'] = rank
    device_map['language_model.lm_head'] = rank
    device_map[f'language_model.model.layers.{num_layers - 1}'] = rank
    return device_map

class _IncompatibleKeys(
        namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):

    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super().__repr__()

    __str__ = __repr__


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    if 'output_layer' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('output_layer')
    return list(lora_module_names)

def polygons_to_bitmask(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    if len(polygons) == 0:
        # COCOAPI does not support empty polygons
        return np.zeros((height, width)).astype(bool)
    rles = mask_util.frPyObjects(polygons, height, width)
    masks = mask_util.decode(rles)
    reduced = np.add.reduce(masks, axis=2)
    m = np.where(reduced>=2, 0, reduced)
    # rle = mask_util.merge(rles)
    return m.astype(bool)


class WrapInternVL(BaseModel):
    def __init__(self, 
                 mllm,
                 tokenizer=None,
                 pretrained_pth=None,
                 llm_lora=None,
                 object_tokenizer=None,
                 ot_image_processor=None,
                 ):
        # super().__init__()

        config = AutoConfig.from_pretrained(mllm["pretrained_model_name_or_path"], trust_remote_code=True)
        self.config = config
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'

        model_clazz = mllm.pop('type')
        mllm.update(dict(config=config))
        self.model = model_clazz(**mllm)
        self.model.language_model.config.use_cache = False

        ot_config = AutoConfig.from_pretrained(object_tokenizer["pretrained_model_name_or_path"], trust_remote_code=True)
        self.ot_config = ot_config
        ot_clazz = object_tokenizer.pop('type')
        self.object_tokenizer = ot_clazz(**object_tokenizer)
        ot_hidden_size = self.object_tokenizer.model.num_features
        self.ot_mlp1 = nn.Sequential(
            nn.LayerNorm(ot_hidden_size,),
            nn.Linear(ot_hidden_size, config.llm_config.hidden_size,),
            nn.GELU(),
            nn.Linear(config.llm_config.hidden_size, config.llm_config.hidden_size)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(mllm["pretrained_model_name_or_path"], trust_remote_code=True, padding_side='right')
        img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.model.img_context_token_id = img_context_token_id
        self._add_special_tokens()

        process_clazz = ot_image_processor.pop('type')
        self.ot_image_processor = process_clazz(**ot_image_processor)
        
        if llm_lora is not None:
            self._prepare_llm_for_lora(llm_lora)
        
        if pretrained_pth is not None:
            assert osp.isfile(pretrained_pth)
            state_dict = torch.load(pretrained_pth, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            mllm_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    mllm_state_dict[k[len('model.'):]] = v
            if len(mllm_state_dict) != 0:
                self.model.load_state_dict(mllm_state_dict, strict=False)
            
            ot_adapter_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('ot_mlp1.'):
                    ot_adapter_state_dict[k[len('ot_mlp1.'):]] = v
            if len(ot_adapter_state_dict) != 0:
                self.ot_mlp1.load_state_dict(ot_adapter_state_dict, strict=False)
         
            for k, v in self.ot_mlp1.named_parameters():
                assert v.equal(ot_adapter_state_dict[k])

            # for k, v in new_state_dict.items():
            #     print(f"{k} : {v.shape}")
            # exit(0)

            # load_state_dict(self, state_dict, strict=False)  # TODO, check whether the internvl2 weights are loaded correctly.
            print(f"Load pretrained weight from {pretrained_pth}")
        
        self.model.to("cuda")

        self.patch_token = int(
            (config.force_image_size // config.vision_config.patch_size)**2 * (config.downsample_ratio**2))

        self._count = 0
        self.max_num=6
        self.device = 'cuda'
        kwargs_default = dict(do_sample=False, max_new_tokens=1024, top_p=None, num_beams=1)
        self.kwargs = kwargs_default
    
    def _add_special_tokens(self):
        assert hasattr(self, "tokenizer")

        special_tokens = [VPT_CONTEXT_TOKEN, ]
        num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        print_log(f"Added {num_new_tokens} special tokens.")
        
        self.vpt_content_token_idx = self.tokenizer(VPT_CONTEXT_TOKEN, add_special_tokens=False).input_ids[0]

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
            lora_config, Config) or isinstance(lora_config, ConfigDict):
            config_clazz = lora_config.pop('type')
            lora_config = config_clazz(**lora_config)
            # lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if self.config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.config.llm_config.architectures[0] == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.config.llm_config.architectures[0] in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplementedError
        lora_config.target_modules = target_modules
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)

    def llm_forward_my_imple(self, tokenizer, pixel_values, num_patches_list, question, generation_config, verbose=False,
                             ot_pixel_values=None, ot_visual_prompts=None):
        if ot_visual_prompts is not None and len(ot_visual_prompts) > 0:
            ot_pixel_values = ot_pixel_values.to("cuda").to(self.object_tokenizer.dtype)
            ot_h, ot_w = ot_pixel_values.shape[-2:]
            ot_num_tokens_h, ot_num_tokens_w = ot_h // self.ot_config.patch_size, ot_w // self.ot_config.patch_size
            summary, ot_embeds = self.object_tokenizer(ot_pixel_values)
            with torch.amp.autocast_mode(device='cuda', dtype=self.model.vision_model.dtype):
                ot_embeds = self.ot_mlp1(ot_embeds)
            
            ot_object_embeds_list = []
            for fidx, ot_visual_prompts_fi in enumerate(ot_visual_prompts):
                nvp, h, w = ot_visual_prompts_fi.shape
                ot_visual_prompts_fi = ot_visual_prompts_fi[:, None, :, :]
                ot_visual_prompts_fi = F.interpolate(ot_visual_prompts_fi.to(ot_embeds.dtype), (ot_num_tokens_h, ot_num_tokens_w), mode="bilinear")
                ot_visual_prompts_fi = (ot_visual_prompts_fi > 0.55).to(ot_embeds.dtype)
                ot_visual_prompts_fi = ot_visual_prompts_fi.reshape(nvp, -1)

                num_vp_tokens = torch.sum(ot_visual_prompts_fi, dim=-1, keepdim=False)
                ot_embeds_fi = ot_embeds[fidx]
                object_embeds = (ot_visual_prompts_fi[:, :, None] / (num_vp_tokens[:, None, None] + 1e-4) * ot_embeds_fi[None, :, :])
                object_embeds = torch.sum(object_embeds, dim=1)
                ot_object_embeds_list.append(object_embeds)
            ot_object_embeds = torch.cat(ot_object_embeds_list)
        else:
            ot_object_embeds = None
        
        tokenizer.padding_side = 'left'
        model_input = tokenizer(question, return_tensors='pt', padding=True)
        input_ids = model_input['input_ids'].cuda()
        attention_mask = model_input['attention_mask'].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids()



        
    
    def generate_inner(self, message, dataset=None):
        image_num = len([x for x in message if x['type'] == 'image'])
        if image_num == 1:
            prompt = '<image>\n' + '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        else:
            prompt, image_idx = '', 1
            for x in message:
                if x['type'] == 'text':
                    prompt += x['value']
                elif x['type'] == 'image':
                    prompt += f'<Image-{image_idx}>'
                    image_idx += 1
            prompt = '\n'.join([f'Image-{i + 1}: <image>' for i in range(image_num)]) + '\n' + prompt
        
        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            num_patches_list = []
            pixel_values_list = []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset)
                curr_pixel_values = load_image(
                    file_name, max_num=self.max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)

            json_path = [image_name.split('.')[0]+'.json' for image_name in image_path]
            json_exists = [os.path.exists(json_file) for json_file in json_path]
            if any(json_exists):
                region_list = []
                for query_json_file in json_path[:-1]:
                    with open(query_json_file, 'r') as f:
                        query_anno = json.load(f)
                    ori_height, ori_width = query_anno[0]['height'], query_anno[0]['width']
                    segm = query_anno[0]['segmentation']
                    segm = [np.array(poly) for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    mask = polygons_to_bitmask(segm, ori_height, ori_width)
                    region_list.append(mask.unsqueeze(0))
                with open(json_path[-1], 'r') as f:
                    query_anno = json.load(f)
                all_masks, all_region_ids = [], []
                for idx in range(len(query_anno)):
                    ori_height, ori_width = query_anno[idx]['height'], query_anno[idx]['width']
                    region_id = query_anno[idx]['id']
                    segm = query_anno[idx]['segmentation']
                    segm = [np.array(poly) for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    mask = polygons_to_bitmask(segm, ori_height, ori_width)
                    all_masks.append(mask.unsqueeze(0))
                    all_region_ids.append(region_id)
                all_masks = np.concatenate(all_region_ids, axis=0)
                region_list.append(all_masks)

                ot_image_path = [image_name[:len('FRAME00')]+'_ORI.'+image_name.split('.')[-1] for image_name in image_path]
                ot_images = [Image.open(image_name).convert('RGB') for image_name in ot_image_path]

                ot_pixel_values, ot_visual_prompts = [], []
                for fi, image in enumerate(ot_images):
                    w, h = image.size
                    if w > h:
                        target_size = (1024, int(h/w*1024))
                    else:
                        target_size = (int(w/h*1024), 1024)
                    resized_image = image.resize(target_size)
                    cur_w, cur_h = resized_image.size
                    padded_image = np.ones(shape=(1024, 1024, 3), dtype=np.uint8) * 255
                    padded_image[:cur_h, :cur_w, :] = np.array(resized_image)

                    ot_pixel_values.append(self.ot_image_processor(images=Image.fromarray(padded_image), return_tensors='pt').pixel_values)
                ot_pixel_values = torch.cat(ot_pixel_values)

                for regions in region_list:
                    h, w = regions.shape[-2:]
                    regions = torch.from_numpy(regions).to(ot_pixel_values.dtype).to(ot_pixel_values.device)
                    if h > w:
                        padded_regions = regions.new_zeros((regions.shape[0], h, h))
                    else:
                        padded_regions = regions.new_zeros((regions.shape[0], w, w))
                    padded_regions[:, :h, :w] = regions
                    resized_padded_regions = F.interpolate(padded_regions.unsqueeze(0), size=(1024, 1024), mode='bilinear').squeeze(0)
                    ot_visual_prompts.append(resized_padded_regions)

                object_token_str = ""
                for fidx in range(image_num-1):
                    object_token_str = object_token_str + f"Objects in Image-{fidx+1}: <query object>{VPT_CONTEXT_TOKEN}\n"
                object_token_str = object_token_str + f"Objects in Image-{image_num}: "
                for region_id in all_region_ids:
                    object_token_str = object_token_str + f"<object-{region_id}>{VPT_CONTEXT_TOKEN}, "
                object_token_str = object_token_str[:-2] + '.\n'

            else:
                region_list, all_region_ids = [], []
                ot_pixel_values, ot_visual_prompts = None, None
                object_token_str = ""
            
            prompt, image_idx = '', 1
            for x in message:
                if x['type'] == 'text':
                    prompt += x['value']
                elif x['type'] == 'image':
                    prompt += f'<Image-{image_idx}>'
                    image_idx += 1
            prompt = '\n'.join([f'Image-{i + 1}: <image>' for i in range(image_num)]) + '\n' + object_token_str + prompt
        elif image_num == 1:
            prompt = '<image>\n' + '\n'.join([x['value'] for x in message if x['type'] == 'text'])

            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            upscale_flag = listinstr(['MMMU_DEV_VAL'], dataset)
            pixel_values = load_image(
                image_path, max_num=self.max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]

            region_list, all_region_ids = [], []
            ot_pixel_values, ot_visual_prompts = None, None
        else:
            pixel_values = None
            num_patches_list = []

            region_list, all_region_ids = [], []
            ot_pixel_values, ot_visual_prompts = None, None

        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                question=prompt,
                generation_config=self.kwargs,
                verbose=False
            )
        return response
