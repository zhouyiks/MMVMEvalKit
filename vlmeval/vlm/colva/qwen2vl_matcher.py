import torch
import torch.amp
from transformers import AutoTokenizer, AutoConfig, AutoModel, CLIPImageProcessor
import warnings
from PIL import Image
from ..base import BaseModel
from mmengine.model import BaseTTAModel, is_model_wrapper
from mmengine.logging import print_log
from mmengine.dist import get_dist_info
from mmengine.config import Config, ConfigDict
from peft import get_peft_model, prepare_model_for_kbit_training
from ...smp import *
from ...dataset import DATASET_TYPE
import pandas as pd
import string
import torch.distributed as dist
import torchvision.transforms as T
import transformers
import torch.nn as nn
from types import MethodType
import pycocotools.mask as mask_util
from typing import Any, Iterator, List, Union, Optional
from transformers import GenerationConfig, AutoProcessor
from transformers.modeling_outputs import BaseModelOutputWithPast

from torchvision.transforms.functional import InterpolationMode
import re
import torch.nn.functional as F

from collections import OrderedDict, namedtuple

from xtuner.model.utils import find_all_linear_names

from .conversation import get_conv_template


VPT_CONTEXT_TOKEN = '<VPT_CONTEXT>'



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

class WrapQwen2VL(BaseModel):
    def __init__(self, 
                 mllm,
                 tokenizer=None,
                 pretrained_pth=None,
                 radio_pretrained_pth=None,
                 llm_lora=None,
                 object_tokenizer=None,
                 ot_image_processor=None,
                 ):
        # super().__init__()

        # transformers == 4.47.0.dev0
        
        config = AutoConfig.from_pretrained(mllm["pretrained_model_name_or_path"], trust_remote_code=True)
        model_clazz = mllm.pop('type')
        self.model = model_clazz(**mllm)
        self.model.model.config.use_cache = False

        self.model.model.forward = MethodType(Qwen2VLModel_forward, self.model.model)

        # self.model.lm_head.bias.to(torch.float32)

        ot_config = AutoConfig.from_pretrained(object_tokenizer["pretrained_model_name_or_path"], trust_remote_code=True)
        self.ot_config = ot_config
        ot_clazz = object_tokenizer.pop('type')
        self.object_tokenizer = ot_clazz(**object_tokenizer)
        ot_hidden_size = self.object_tokenizer.model.num_features
        llm_hidden_size = self.model.model.config.hidden_size
        self.ot_mlp1 = nn.Sequential(
            nn.LayerNorm(ot_hidden_size,),
            nn.Linear(ot_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.processor = AutoProcessor.from_pretrained(mllm["pretrained_model_name_or_path"])
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

            # for k, v in state_dict.items():
            #     if 'radio_model.summary_idxs' in k:
            #         self.object_tokenizer.radio_model.summary_idxs = v
            #     if 'radio_model.input_conditioner.norm_mean' in k:
            #         self.object_tokenizer.radio_model.input_conditioner.norm_mean = v
            #     if 'radio_model.input_conditioner.norm_std' in k:
            #         self.object_tokenizer.radio_model.input_conditioner.norm_std = v

            mllm_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    mllm_state_dict[k[len('model.'):]] = v
            if len(mllm_state_dict) != 0:
                self.model.load_state_dict(mllm_state_dict, strict=False)
            
            pretrained_state_dict = torch.load(radio_pretrained_pth, map_location='cpu')
            if 'state_dict' in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict['state_dict']
            ot_adapter_state_dict = {}
            for k, v in pretrained_state_dict.items():
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
        
        # self.model.to("cuda")
        self.object_tokenizer.to("cuda")
        self.ot_mlp1.to("cuda")

        self.device = 'cuda'
    
    def _add_special_tokens(self):
        assert hasattr(self, "processor")

        special_tokens = [VPT_CONTEXT_TOKEN, ]
        num_new_tokens = self.processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
        print_log(f"Added {num_new_tokens} special tokens.")
        
        self.vpt_content_token_idx = self.processor.tokenizer(VPT_CONTEXT_TOKEN, add_special_tokens=False).input_ids[0]
        image_token = "<|image_pad|>" if not hasattr(self.processor.tokenizer, "image_token") else self.processor.tokenizer.image_token
        self.img_context_token_idx = self.processor.tokenizer(image_token, add_special_tokens=False).input_ids[0]

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
            lora_config, Config) or isinstance(lora_config, ConfigDict):
            config_clazz = lora_config.pop('type')
            lora_config = config_clazz(**lora_config)
            # lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.model.model = prepare_model_for_kbit_training(self.model.model, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.model)
            lora_config.target_modules = modules
        
        self.model.model = get_peft_model(self.model.model, lora_config)


    def llm_forward_my_imple(self, inputs, ot_pixel_values=None, ot_visual_prompts=None):
        if ot_visual_prompts is not None and len(ot_visual_prompts) > 0:
            ot_pixel_values = ot_pixel_values.to("cuda").to(self.object_tokenizer.dtype)
            ot_h, ot_w = ot_pixel_values.shape[-2:]
            ot_num_tokens_h, ot_num_tokens_w = ot_h // self.ot_config.patch_size, ot_w // self.ot_config.patch_size
            summary, ot_embeds = self.object_tokenizer(ot_pixel_values)
            ot_embeds = self.ot_mlp1(ot_embeds)
            
            ot_object_embeds_list = []
            for fidx, ot_visual_prompts_fi in enumerate(ot_visual_prompts):
                nvp, h, w = ot_visual_prompts_fi.shape
                ot_visual_prompts_fi = ot_visual_prompts_fi[:, None, :, :].to("cuda").to(self.object_tokenizer.dtype)
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
        
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")
        pixel_values = inputs.pixel_values.to("cuda").to(self.model.visual.dtype)
        image_grid_thw = inputs.image_grid_thw.to("cuda")

        vit_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)

        B, N = input_ids.shape
        temp_input_ids = input_ids.clone().flatten()
        temp_input_ids[temp_input_ids == self.vpt_content_token_idx] = self.img_context_token_idx
        # temp_input_ids.to(self.model.get_input_embeddings().weight.device)
        input_embeds = self.model.get_input_embeddings()(temp_input_ids.reshape(B, N)).clone()

        B, N, C  = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)
        
        if ot_object_embeds is not None:
            selected = (input_ids == self.vpt_content_token_idx)
            selected = selected.to(input_embeds.device)
            ot_object_embeds = ot_object_embeds.to(input_embeds.device)
            input_embeds[selected] = input_embeds[selected] * 0.0 + ot_object_embeds

        selected = (input_ids == self.img_context_token_idx)
        vit_embeds = vit_embeds.to(input_embeds.device)
        selected = selected.to(input_embeds.device)
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        input_embeds = input_embeds.reshape(B, N, C)

        output_ids = self.model.generate(
            inputs_embeds=input_embeds.to('cuda:0'),
            attention_mask=attention_mask.to('cuda:0'),
            use_cache=True,
            max_new_tokens=512,
        )
        input_ids = input_ids.reshape(B, N)
        generated_ids = [_output_ids[-200:] for _input_ids, _output_ids in zip(input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
        return output_text[0]
    
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
        
            json_path = ['.'.join(image_name.split('.')[:-1])+'.json' for image_name in image_path]
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
                    region_list.append(mask[np.newaxis, :, :])
                with open(json_path[-1], 'r') as f:
                    query_anno = json.load(f)
                all_masks, all_region_ids = [], []
                for idx in range(len(query_anno)):
                    ori_height, ori_width = query_anno[idx]['height'], query_anno[idx]['width']
                    region_id = query_anno[idx]['id']
                    segm = query_anno[idx]['segmentation']
                    segm = [np.array(poly) for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    mask = polygons_to_bitmask(segm, ori_height, ori_width)
                    all_masks.append(mask)
                    all_region_ids.append(region_id)
                all_masks = np.stack(all_masks, axis=0)
                region_list.append(all_masks)
                
                image_names = [os.path.basename(image_name) for image_name in image_path]
                dir_names = [os.path.dirname(image_name) for image_name in image_path]
                ot_image_path = [os.path.join(dir_name, image_name[:len('FRAME00')]+'_ORI.'+image_name.split('.')[-1])
                                  for image_name, dir_name in zip(image_names, dir_names)]
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
                
                sorted_indices = sorted(range(len(all_region_ids)), key=lambda k: all_region_ids[k])
                for sorted_idx in sorted_indices:
                    object_token_str = object_token_str + f"<object-{all_region_ids[sorted_idx]}>{VPT_CONTEXT_TOKEN}, "
                # for region_id in all_region_ids:
                #     object_token_str = object_token_str + f"<object-{region_id}>{VPT_CONTEXT_TOKEN}, "
                object_token_str = object_token_str[:-2] + '.\n'

                cand_visual_prompts = []
                for sort_idx in sorted_indices:
                    cand_visual_prompts.append(ot_visual_prompts[-1][sort_idx])
                cand_visual_prompts = torch.stack(cand_visual_prompts, dim=0)
                ot_visual_prompts[-1] = cand_visual_prompts

            else:
                region_list, all_region_ids = [], []
                ot_pixel_values, ot_visual_prompts = None, None
                object_token_str = ""

            out_conversation_list = [{
                "role": "system", 
                "content": [{
                    "type": "text", 
                    "text": "You are a helpful assistant."}]
            }]
            
            out_contents = []
            image_idx = 0
            for element in message:
                if element["type"] == "image":
                    image_idx += 1
                    out_contents.append({
                        "type": "text",
                        "text": f"Image-{image_idx}: "
                    })
                    out_contents.append({
                        "type": "image",
                    })
                else:
                    msg_value = element["value"]
                    out_contents.append({
                        "type": "text",
                        "text": object_token_str
                    })
                    out_contents.append({
                        "type": "text",
                        "text": msg_value,
                    })
            out_conversation_list.append({
                "role": "user",
                "content": out_contents,
            })
            text_prompt = self.processor.apply_chat_template(out_conversation_list, add_generation_prompt=True)

            images = [Image.open(ele).convert('RGB') for ele in image_path]
            resized_padded_images = []
            for image in images:
                w, h = image.size
                if w > h:
                    target_size = (1024, int(h/w*1024))
                else:
                    target_size = (int(w/h*1024), 1024)
                resized_image = image.resize(target_size)
                cur_w, cur_h = resized_image.size
                padded_image = np.zeros(shape=(1024, 1024, 3), dtype=np.uint8) * 255
                padded_image[:cur_h, :cur_w, :] = np.array(resized_image)
                padded_image = Image.fromarray(padded_image)
                resized_padded_images.append(padded_image)
            inputs = self.processor(text=[text_prompt], images=resized_padded_images, padding=True, return_tensors="pt")
          
            # prompt, image_idx = '', 1
            # for x in message:
            #     if x['type'] == 'text':
            #         prompt += x['value']
            #     elif x['type'] == 'image':
            #         prompt += f'<Image-{image_idx}>'
            #         image_idx += 1
            # prompt = '\n'.join([f'Image-{i + 1}: <image>' for i in range(image_num)]) + '\n' + object_token_str + prompt
        else:
            pixel_values = None
            num_patches_list = []

            region_list, all_region_ids = [], []
            ot_pixel_values, ot_visual_prompts = None, None
            inputs = None

        with torch.no_grad():
            # response = self.model.chat(
            #     self.tokenizer,
            #     pixel_values=pixel_values,
            #     num_patches_list=num_patches_list,
            #     question=prompt,
            #     generation_config=self.kwargs,
            #     verbose=False,
            # )
            response = self.llm_forward_my_imple(
                inputs=inputs,
                ot_pixel_values=ot_pixel_values,
                ot_visual_prompts=ot_visual_prompts,
            )
        return response



def Qwen2VLModel_forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # logger.warning_once(
                #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                # )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )