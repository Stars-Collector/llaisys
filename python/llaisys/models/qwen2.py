from typing import Sequence
from ..libllaisys import LIB_LLAISYS, DeviceType, LlaisysQwen2Meta, LlaisysQwen2Model_t, LlaisysQwen2Weights, llaisysTensor_t
from ..tensor import Tensor
from ..libllaisys import DataType
import ctypes
import numpy as np
from pathlib import Path
import json
import struct
import safetensors


def bf16_to_fp16(bf16_array):
    """Convert bfloat16 array to float16 array."""
    original_shape = bf16_array.shape
    flat_array = bf16_array.ravel()
    bf16_as_uint32 = flat_array.astype(np.uint32) << 16
    fp32 = np.frombuffer(bf16_as_uint32.tobytes(), dtype=np.float32)
    fp16 = fp32.astype(np.float16)
    return fp16.reshape(original_shape)


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        
        # Load config
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Map config to meta
        meta = LlaisysQwen2Meta()
        meta.dtype = DataType.BF16.value
        meta.nlayer = config["num_hidden_layers"]
        meta.hs = config["hidden_size"]
        meta.nh = config["num_attention_heads"]
        meta.nkvh = config["num_key_value_heads"]
        meta.dh = meta.hs // meta.nh
        meta.di = meta.hs
        meta.maxseq = 4096 
        meta.voc = config["vocab_size"]
        meta.epsilon = config["rms_norm_eps"]
        meta.theta = config["rope_theta"]
        meta.end_token = config["eos_token_id"]
        meta.intermediate_size = config.get("intermediate_size", 4 * meta.hs)
        
        self.end_token = meta.end_token
        
        # Create model
        device_id = 0
        self.device = device
        self.model_ptr = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta),
            device.value,
            ctypes.byref(ctypes.c_int(device_id)),
            1
        )
        
        # Get weights
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model_ptr)
        self.weights = weights_ptr.contents
        
        # Load safetensors
        weight_map = {
            "model.embed_tokens.weight": self.weights.in_embed,
            "lm_head.weight": self.weights.out_embed,
            "model.norm.weight": self.weights.out_norm_w,
        }
        
        for layer in range(meta.nlayer):
            weight_map[f"model.layers.{layer}.input_layernorm.weight"] = self.weights.attn_norm_w[layer]
            weight_map[f"model.layers.{layer}.self_attn.q_proj.weight"] = self.weights.attn_q_w[layer]
            weight_map[f"model.layers.{layer}.self_attn.q_proj.bias"] = self.weights.attn_q_b[layer]
            weight_map[f"model.layers.{layer}.self_attn.k_proj.weight"] = self.weights.attn_k_w[layer]
            weight_map[f"model.layers.{layer}.self_attn.k_proj.bias"] = self.weights.attn_k_b[layer]
            weight_map[f"model.layers.{layer}.self_attn.v_proj.weight"] = self.weights.attn_v_w[layer]
            weight_map[f"model.layers.{layer}.self_attn.v_proj.bias"] = self.weights.attn_v_b[layer]
            weight_map[f"model.layers.{layer}.self_attn.o_proj.weight"] = self.weights.attn_o_w[layer]
            weight_map[f"model.layers.{layer}.post_attention_layernorm.weight"] = self.weights.mlp_norm_w[layer]
            weight_map[f"model.layers.{layer}.mlp.gate_proj.weight"] = self.weights.mlp_gate_w[layer]
            weight_map[f"model.layers.{layer}.mlp.up_proj.weight"] = self.weights.mlp_up_w[layer]
            weight_map[f"model.layers.{layer}.mlp.down_proj.weight"] = self.weights.mlp_down_w[layer]
        
        for file in sorted(model_path.glob("*.safetensors")):
            with open(file, 'rb') as f:
                header_len = int.from_bytes(f.read(8), byteorder='little')
                header_data = f.read(header_len)
                header = json.loads(header_data)
                
                for name_, tensor_info in header.items():
                    if name_ in weight_map:
                        try:
                            dtype_str = tensor_info['dtype']
                            shape = tensor_info['shape']
                            data_offset = tensor_info['data_offsets'][0]
                            data_length = tensor_info['data_offsets'][1] - data_offset
                            
                            f.seek(8 + header_len + data_offset)
                            raw_data = f.read(data_length)
                            
                            if dtype_str == 'BF16':
                                tensor_data = np.frombuffer(raw_data, dtype=np.uint16)
                                tensor_data = tensor_data.reshape(shape)
                            elif dtype_str == 'F16':
                                tensor_data = np.frombuffer(raw_data, dtype=np.float16)
                                tensor_data = tensor_data.reshape(shape)
                            elif dtype_str == 'F32':
                                tensor_data = np.frombuffer(raw_data, dtype=np.float32)
                                tensor_data = tensor_data.reshape(shape)
                            elif dtype_str == 'I64':
                                tensor_data = np.frombuffer(raw_data, dtype=np.int64)
                                tensor_data = tensor_data.reshape(shape)
                            elif dtype_str == 'I32':
                                tensor_data = np.frombuffer(raw_data, dtype=np.int32)
                                tensor_data = tensor_data.reshape(shape)
                            else:
                                raise ValueError(f"Unsupported dtype: {dtype_str}")
                            
                            tensor_ptr = weight_map[name_]
                            
                            # CRITICAL FIX: DO NOT TRANSPOSE.
                            # PyTorch Linear weights are saved as [out_features, in_features].
                            # C++ backend expects [out_features, in_features] for Y = X * W^T.
                            
                            LIB_LLAISYS.tensorLoad(tensor_ptr, tensor_data.ctypes.data_as(ctypes.c_void_p))
                            # 如果是输入 Embedding，且配置启用了 tie_word_embeddings，则同时也加载到输出 Embedding (lm_head)
                            if name_ == "model.embed_tokens.weight" and config.get("tie_word_embeddings", False):
                                print(f"  -> Tying weights: loading {name_} into lm_head")
                                LIB_LLAISYS.tensorLoad(self.weights.out_embed, tensor_data.ctypes.data_as(ctypes.c_void_p))
                        except Exception as e:
                            print(f"Error loading tensor {name_}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
    
    def __del__(self):
        if hasattr(self, 'model_ptr') and self.model_ptr:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model_ptr)
    
    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128
        
        input_array = (ctypes.c_int64 * len(inputs))(*inputs)
        output_tokens = list(inputs)
        
        # Prefill and first token generation
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.model_ptr,
            input_array,
            len(input_array)
        )

        if next_token == self.end_token:
            return output_tokens

        output_tokens.append(next_token)

        # Decoding loop
        for _ in range(max_new_tokens - 1):
            if next_token == self.end_token:
                break
            
            new_token_array = (ctypes.c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model_ptr,
                new_token_array,
                1
            )
            
            output_tokens.append(next_token)
        
        return output_tokens