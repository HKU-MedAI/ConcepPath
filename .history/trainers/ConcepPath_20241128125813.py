import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip, attention
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def orthogonal_loss(features, gamma=0.5):
    if len(features.shape)<3:
        features = features.unsqueeze(0)
        
    device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
    #  features are normalized
    features = F.normalize(features, p=2, dim=1)
    labels = torch.arange(features.shape[-1])
    labels = labels[:, None]  # extend dim

    mask = torch.eq(labels, labels.t()).bool().to(device)
    eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

    mask_pos = mask.masked_fill(eye, 0).float()
    mask_neg = (~mask).float()
    dot_prod = torch.matmul(features.transpose(1,2), features)
    pos_pairs_mean = (mask_pos * dot_prod).sum(dim=(1,2)) / (mask_pos.sum() + 1e-6)
    neg_pairs_mean = (mask_neg * dot_prod).sum(dim=(1,2)) / (mask_neg.sum() + 1e-6)  # TODO: removed abs
    
    loss = (1.0 - pos_pairs_mean) + gamma * neg_pairs_mean

    return loss.sum()

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.ln_final.weight.dtype

    def forward(self, prompts, tokenized_prompts):
        
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class PromptLearner(nn.Module):
    
    def __init__(self, classnames, clip_model, n_ctx, n_ddp=0, num_patch_prompt=0, is_shared=False):
        super().__init__()
        n_cls = len(classnames)
        
        self.n_ddp = n_ddp
        self.num_patch_prompt = num_patch_prompt

        # ===============
        n_ctx = n_ctx # cfg.TRAINER.COOP.N_CTX
        ctx_init = "" # cfg.TRAINER.COOP.CTX_INIT
        # ===============

        dtype = clip_model.ln_final.weight.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = 224

        # ===============
        cfg_imsize = 224 #cfg.INPUT.SIZE[0]
        # ===============
        
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
        # random initialization
            if not is_shared:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                if n_ddp>0 and num_patch_prompt>0:
                    ddp_vectors = torch.empty(int(n_cls/num_patch_prompt)*n_ddp, 75, ctx_dim, dtype=dtype)
                
            else:
                print("Initializing a generic context")
                if n_ddp>0 and num_patch_prompt>0:
                    ddp_vectors = torch.empty(75, ctx_dim, dtype=dtype)
                    
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        # else:
        # print(prompts)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        if n_ddp>0 and num_patch_prompt>0:
            nn.init.normal_(ddp_vectors, std=0.02)
            self.ddp = nn.Parameter(ddp_vectors)
            ddp_prefix = " ".join(["X"] * 75)
            tokenized_ddp = torch.cat([clip.tokenize(ddp_prefix+".") for _ in range(int(n_cls/num_patch_prompt)*n_ddp)]).to(device)
            
            with torch.no_grad():
                embedding_ddp = clip_model.token_embedding(tokenized_ddp).type(dtype)
            
            self.register_buffer("ddp_token_prefix", embedding_ddp[:, :1, :])  # SOS
            self.register_buffer("ddp_token_suffix", embedding_ddp[:, 1 + 75 :, :])  # CLS, EOS
            
            tokenized_prompts_ = []
            for i in range(n_cls):
                if i%num_patch_prompt==0:
                    cur_i_ = int(i/num_patch_prompt)
                    tokenized_prompts_.append(tokenized_ddp[cur_i_:cur_i_+n_ddp])
                tokenized_prompts_.append(tokenized_prompts[i].unsqueeze(0))
            self.tokenized_prompts = torch.cat(tokenized_prompts_, dim=0).to(device)
        else:
            self.tokenized_prompts = tokenized_prompts
        
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens
        
        # ===============
        self.class_token_position = "top" #cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        # ===============

    def forward(self):
        ctx = self.ctx
        
        if self.n_ddp>0 and self.num_patch_prompt>0:
            ddp_prefix = self.ddp_token_prefix
            ddp_suffix = self.ddp_token_suffix
            ddp = self.ddp
            if ddp.dim() == 2:
                ddp = ddp.unsqueeze(0).expand(int(self.n_cls/self.num_patch_prompt)*self.n_ddp, -1, -1)
            
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        
        # ==== 
        if self.class_token_position == "top":
            prompts = []
            for i in range(self.n_cls):
                if self.n_ddp>0 and self.num_patch_prompt>0:
                    if i%self.num_patch_prompt==0:

                        cur_i_ = int(i/self.num_patch_prompt)
                    
                        ddp_i = ddp[cur_i_: cur_i_+self.n_ddp, :, :]
                        ddp_prefix_i = ddp_prefix[cur_i_: cur_i_+self.n_ddp, :, :]
                        ddp_suffix_i = ddp_suffix[cur_i_: cur_i_+self.n_ddp, :, :]
                        
                        prompt_ddp = torch.cat(
                                [
                                    ddp_prefix_i,
                                    ddp_i,
                                    ddp_suffix_i
                                ],
                                dim=1,
                                ) 
                        
                        prompts.append(
                            prompt_ddp
                        )
                # tokenized_prompts.append(clip.tokenize(prompts[cur_i]))
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt_i = torch.cat(
                    [
                        prefix_i,  
                        class_i,   
                        ctx_i,     
                        suffix_i,  
                    ],
                    dim=1,
                )
                prompts.append(prompt_i)
            prompts = torch.cat(prompts, dim=0)

        return prompts


class ConcepPath(nn.Module):
    def __init__(self, 
                 slide_prompt, 
                 patch_prompt, 
                 clip_model, 
                 loss_func, 
                 n_classes, 
                 num_patch_prompt=26, 
                 n_ctx=16, 
                 n_ddp=0, 
                 is_shared=False, 
                 orth_ratio=2, 
                 weighted_type="p2c", 
                 is_adapted=True
                 ):
        
        super().__init__()
        
        self.num_patch_prompt_ = num_patch_prompt+n_ddp
        self.weighted_type = weighted_type
            
        self.orth_ratio = orth_ratio
        
        self.clip_model = clip_model
        
        self.n_ddp = n_ddp
        self.patch_prompt_learner = PromptLearner(patch_prompt, clip_model, n_ctx, n_ddp, num_patch_prompt, is_shared=is_shared)
        self.slide_prompt_learner = PromptLearner(slide_prompt, clip_model, n_ctx, is_shared=is_shared)
       
        self.patch_tokenized_prompts = self.patch_prompt_learner.tokenized_prompts
        self.slide_tokenized_prompts = self.slide_prompt_learner.tokenized_prompts

        self.text_encoder = TextEncoder(clip_model)
        
        self.loss_func = loss_func
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.ln_final.weight.dtype
        
        self.is_adapted = is_adapted
        
        self.patch_adapt_ratio = nn.Parameter(torch.tensor(0.2))
        self.slide_adapt_ratio = nn.Parameter(torch.tensor(0.2))
        
        if is_adapted:
            self.slide_text_adapter = Adapter(512)
            self.slide_feature_adapter = Adapter(512)
            

    def forward(self, patch_features, label, result_fp=None, test=False):
        
        patch_features = patch_features.squeeze(0)

        patch_features = patch_features.type(self.dtype)
        patch_prompts = self.patch_prompt_learner()
        patch_tokenized_prompts = self.patch_tokenized_prompts

        slide_prompts = self.slide_prompt_learner()
        slide_tokenized_prompts = self.slide_tokenized_prompts
        
        patch_text_features = self.text_encoder(patch_prompts, patch_tokenized_prompts)
        slide_text_features = self.text_encoder(slide_prompts, slide_tokenized_prompts)
        
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
        
        slide_adapt_ratio = 0.2
        
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
        patch_text_features = patch_text_features / patch_text_features.norm(dim=-1, keepdim=True)
        slide_text_features = slide_text_features / slide_text_features.norm(dim=-1, keepdim=True)
        
        if test:
            import pickle
            out_put = {"patch_features": patch_text_features, "slide_features": slide_text_features}
            feature_fp = result_fp.replace("attn_score", "feature")
            import os
            feature_rp = "/".join(feature_fp.split("/")[:-1])
            os.makedirs(feature_rp, exist_ok=True)
            with open(feature_rp+"/feature.pkl", 'wb') as file:
                pickle.dump(out_put, file)

        logit_scale = self.logit_scale.exp()
        patch_text_features = patch_text_features.type(self.dtype)
        slide_text_features = slide_text_features.type(self.dtype)
        
        # oth loss for patch text features
        loss_i = orthogonal_loss(patch_text_features.reshape(-1, self.num_patch_prompt_, 512).transpose(1,2))
        
        sim_matrix = F.softmax(logit_scale*patch_features@patch_text_features.t(), dim=1)
        
        if test:
            import pickle
            attention_score = sim_matrix
            out_put = {"name": result_fp.split("/")[-1], "attention_score": attention_score}
            with open(result_fp.replace(".pkl", "_attention_score.pkl"), 'wb') as file:
                pickle.dump(out_put, file)
        
        num_patchs, _ = sim_matrix.shape
        
        slide_features = sim_matrix.t() @ patch_features
        _, embedding_len = slide_features.shape
        slide_features = slide_features/slide_features.norm(dim=-1, keepdim=True)
        slide_features = slide_features.reshape(-1, self.num_patch_prompt_, embedding_len)

        slide_features_ = torch.bmm(
                slide_text_features.unsqueeze(1), 
                patch_text_features.reshape(-1, self.num_patch_prompt_, embedding_len).transpose(1, 2),
            ).squeeze(1)
        slide_features_ = slide_features_/torch.sum(slide_features_, dim=-1, keepdim=True)
        slide_features = slide_features+slide_features_.unsqueeze(-1)*slide_features
            
            
        if test:
            import pickle
            patch_prompt_score = slide_features_.unsqueeze(-1)
            out_put = {"name": result_fp.split("/")[-1], "patch_prompt_score": patch_prompt_score}
            with open(result_fp.replace(".pkl", "_patch_prompt_score.pkl"), 'wb') as file:
                pickle.dump(out_put, file)
            
        slide_features = slide_features/slide_features.norm(dim=-1, keepdim=True)
        
        slide_features = torch.mean(slide_features, dim=1)
        slide_features = slide_features / slide_features.norm(dim=-1, keepdim=True)
        
        if self.is_adapted:
            adapted_slide_features = self.slide_feature_adapter(slide_features)
            slide_features = slide_adapt_ratio * adapted_slide_features/adapted_slide_features.norm(dim=-1, keepdim=True) + (1 - slide_adapt_ratio) * slide_features
            slide_features = slide_features / slide_features.norm(dim=-1, keepdim=True)
            
            adapted_slide_text_features = self.slide_feature_adapter(slide_text_features)
            slide_text_features = slide_adapt_ratio * adapted_slide_text_features/adapted_slide_text_features.norm(dim=-1, keepdim=True) + (1 - slide_adapt_ratio) * slide_text_features
            slide_text_features = slide_text_features / slide_text_features.norm(dim=-1, keepdim=True)
        
        logits = torch.diag(logit_scale*slide_features @ slide_text_features.t()).unsqueeze(0)
        
        Y_prob = F.softmax(logits)

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        
        if test:
            # print(f" \
            #     概率为：{Y_prob}\n \
            #     分类为：{Y_hat}\n \
            #     attention score 为：{attention_score}\n \
            #     patch_prompt_score 为：{patch_prompt_score} \
            # ")
            return Y_prob, Y_hat, attention_score, patch_prompt_score
        
        loss = self.loss_func(logits, label)

        return logits, Y_prob, Y_hat, loss+self.orth_ratio*loss_i.sum()