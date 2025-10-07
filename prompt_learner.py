from typing import Union, List

from PIL.ImageFilter import Kernel
from pkg_resources import packaging
import torch
from VIP2CLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy
import torch.nn as nn
import numpy as np

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[
    torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]

    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)
    # token不够的位置填充为0，eot后面开始
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

#tokenize--encode--归一化
def encode_text_with_prompt_ensemble(model,obj_list,device):
    prompt_normal=['a photo of a normal {}.']
    prompt_abnormal=['a photo of an abnormal {}.']

    text_features_pos=[]
    text_features_neg=[]
    for obj in obj_list:
        #1,d
        prompt_pos=[state.format(obj) for state in prompt_normal]
        prompt_pos=tokenize(prompt_pos)
        prompt_pos_embedding=model.encode_text(prompt_pos.to(device))
        prompt_pos_embedding=prompt_pos_embedding/prompt_pos_embedding.norm(dim=-1,keepdim=True)
        text_features_pos.append(prompt_pos_embedding)

        prompt_neg=[state.format(obj) for state in prompt_abnormal]
        prompt_neg=tokenize(prompt_neg)
        prompt_neg_embedding=model.encode_text(prompt_neg.to(device))
        prompt_neg_embedding=prompt_neg_embedding/prompt_neg_embedding.nrom(dim=-1,keepdim=True)
        text_features_neg.append(prompt_neg_embedding)
    #n,d
    prompt_pos_embeddings=torch.cat(text_features_pos,dim=0)
    prompt_neg_embeddings=torch.cat(text_features_neg,dim=0)
    #n,2,d
    prompt_embeddings=torch.stack((prompt_pos_embeddings,prompt_neg_embeddings),dim=1)
    return prompt_embeddings

class Linearlayer(nn.Module):
    def __init__(self, dim_in, dim_out, k=3, hidden_dim=384, use_bais=False):
        super(Linearlayer, self).__init__()
        self.num_layer = k

        if use_bais:
            bias_vectors_pos = torch.empty(1, dim_in)
            nn.init.normal_(bias_vectors_pos, std=0.02)
            self.bias = nn.Parameter(bias_vectors_pos)
        else:
            self.bias = 0

        self.fc = nn.ModuleList()
        for _ in range(k):
            layers = [
                nn.Linear(dim_in, hidden_dim),
                nn.ReLU(), 
                nn.Linear(hidden_dim, dim_out)  
            ]
            self.fc.append(nn.Sequential(*layers))

    def forward(self, tokens):

        tokens = tokens + self.bias
        outputs = [fc(tokens) for fc in self.fc]

        return torch.cat(outputs, dim=1)

class ICA(nn.Module):
    def __init__(self,obj_list,clip_model,design_details,use_bias=True):
        super().__init__()

        #准备可学习参数和固定text的token，forward中替换
        ctx_dim=clip_model.ln_final.weight.shape[0]
        dtype=clip_model.transformer.get_cast_dtype() #transformer部分的数值精度
        self.dtype=dtype

        self.normal_state_prompt=['good {}']
        self.abnormal_state_prompt=['damaged {}']

        self.n_cls=design_details['Prompt_cls_length']
        n_cls=self.n_cls
        cls_token=' '.join(['X']*n_cls)
        cls_ctx=torch.empty(n_cls,ctx_dim,dtype=dtype)
        nn.init.normal_(cls_ctx,std=0.02)
        self.cls_ctx=nn.Parameter(cls_ctx)

        self.n_ctx=design_details['Prompt_length']
        n_ctx=self.n_ctx
        ctx_token=' '.join(['X']*n_ctx)
        ctx_pos=torch.empty(n_ctx,ctx_dim,dtype=dtype)
        nn.init.normal_(ctx_pos,std=0.02)
        self.ctx_pos=nn.Parameter(ctx_pos)
        ctx_neg=torch.empty(n_ctx,ctx_dim,dtype=dtype)
        nn.init.normal_(ctx_neg,std=0.02)
        self.ctx_neg=nn.Parameter(ctx_neg)

        self.cls_prompter=Linearlayer(dim_in=768,dim_out=768,k=self.n_cls,use_bais=use_bias)

        normal_sentences=[
            ctx_token+' '+state.format(cls_token) for state in self.normal_state_prompt
        ]
        abnormal_sentences=[
            ctx_token+' '+state.format(cls_token) for state in self.abnormal_state_prompt
        ]

        pos_prompts_tokens=[]
        neg_prompts_tokens=[]
        for prompt in normal_sentences:
            pos_promtp_token=tokenize(prompt)
            pos_prompts_tokens.append(pos_promtp_token)
        for prompt in abnormal_sentences:
            neg_promtp_token=tokenize(prompt)
            neg_prompts_tokens.append(neg_promtp_token)
        pos_prompts_tokens=torch.cat(pos_prompts_tokens,dim=0)
        neg_prompts_tokens=torch.cat(neg_prompts_tokens,dim=0)

        with torch.no_grad():
            pos_prompt_embeddings=clip_model.token_embedding(neg_prompts_tokens).type(dtype)
            neg_prompt_embeddings=clip_model.token_embedding(neg_prompts_tokens).type(dtype)

        self.register_buffer('token_prefix_pos',pos_prompt_embeddings[:,:1,:])
        self.register_buffer('token_middle_pos',pos_prompt_embeddings[:,1+self.n_ctx:1+self.n_ctx+1,:])
        self.register_buffer('token_suffix_pos',pos_prompt_embeddings[:,1+self.n_ctx+1+self.n_cls:,:])

        self.register_buffer('token_prefix_neg',neg_prompt_embeddings[:,:1,:])
        self.register_buffer('token_middle_neg',neg_prompt_embeddings[:,1+self.n_ctx:1+self.n_ctx+1,:])
        self.register_buffer('token_suffix_neg',neg_prompt_embeddings[:,1+self.n_ctx+1+self.n_cls:,:])

        self.register_buffer('tokenized_prompts_pos',pos_prompts_tokens)
        self.register_buffer('tokenized_prompts_neg',neg_prompts_tokens)

        print(f'pos_prompts_tokens.shape:{pos_prompts_tokens.shape}')
        print(f'neg_prompts_tokens.shape:{neg_prompts_tokens.shape}')


    #返回prompts及tokens 2n,l  2n,l,d
    def forward(self,img_features):

        #获取可学习量和固定embedding
        #L,D
        ctx_cls=self.cls_ctx
        ctx_pos=self.ctx_pos
        ctx_neg=self.ctx_neg

        #N,L,D
        pos_ctx_prefix=self.token_prefix_pos
        pos_ctx_middle=self.token_middle_pos
        pos_ctx_suffix=self.token_suffix_pos

        neg_ctx_prefix=self.token_prefix_neg
        neg_ctx_middle=self.token_middle_neg
        neg_ctx_suffix=self.token_suffix_neg

        N,D=img_features.shape
        img_ctx_cls=self.cls_prompter(img_features.reshape(N,1,D)).type(self.dtype)
        ctx_cls=ctx_cls.unsqueeze(0)+img_ctx_cls

        pos_prompt_embeddings=torch.cat([
            pos_ctx_prefix.expand(N,-1,-1),
            ctx_pos.unsqueeze(0).expand(N,-1,-1),
            pos_ctx_middle.expand(N,-1,-1),
            ctx_cls,
            pos_ctx_suffix.expand(N,-1,-1)
        ],dim=1)

        neg_prompt_embeddings=torch.cat([
            neg_ctx_prefix.expand(N,-1,-1),
            ctx_neg.unsqueeze(0).expand(N,-1,-1),
            neg_ctx_middle.expand(N,-1,-1),
            ctx_cls,
            neg_ctx_suffix.expand(N,-1,-1)
        ],dim=1)

        promtps=torch.cat((pos_prompt_embeddings,neg_prompt_embeddings),dim=0)

        pos_prompts_tokens=self.tokenized_prompts_pos.expand(N,-1)
        neg_prompts_tokens=self.tokenized_prompts_neg.expand(N,-1)
        prompts_tokens=torch.cat((pos_prompts_tokens,neg_prompts_tokens),dim=0)

        return promtps,prompts_tokens


class FGP(nn.Module):
    def __init__(self, dim_v,dim_t, dim_out, num_heads = 8, qkv_bias = False, qk_scale = None, attn_drop = 0., proj_drop = 0.):
        super().__init__()
        self.num_heads = num_heads 
        self.head_dim = dim_out // num_heads
        self.dim_out = dim_out

        self.q_proj_pre = nn.Conv1d(dim_t, dim_out, kernel_size=1)
        self.k_proj_pre_1 =nn.Conv1d(dim_v, dim_out, kernel_size=1)
        self.v_proj_pre_1 = nn.Conv1d(dim_v, dim_out, kernel_size=1)
        self.proj_post_t = nn.Conv1d(dim_out, dim_out, kernel_size=1)
        
        self.prompt_temp_l1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.beta_t = 1 

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std= 0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(self, F_t, F_s):
        B1, N1, C1 = F_t.shape
        B2, N2, C2 = F_s.shape
        assert B1 == B2
        q_t = self.q_proj_pre(F_t.permute(0, 2, 1)).permute(0, 2, 1).reshape(B1, N1, self.num_heads, self.head_dim)  #1
        k_s = self.k_proj_pre_1(F_s.permute(0, 2, 1)).permute(0, 2, 1).reshape(B2, N2, self.num_heads, self.head_dim)  #1
        v_s = self.v_proj_pre_1(F_s.permute(0, 2, 1)).permute(0, 2, 1).reshape(B2, N2, self.num_heads, self.head_dim)  #1
        attn_t = torch.einsum('bnkc,bmkc->bknm', q_t, k_s) * self.beta_t
        attn_t = attn_t.softmax(dim = -1)
        F_t_a = torch.einsum('bknm,bmkc->bnkc', attn_t, v_s).reshape(B1, N1, self.dim_out)
        F_t_a = self.proj_post_t(F_t_a.permute(0, 2, 1)).permute(0, 2, 1) 
        F_t_a = F_t_a / F_t_a.norm(dim=-1, keepdim = True)
        
        return F_t_a









        


