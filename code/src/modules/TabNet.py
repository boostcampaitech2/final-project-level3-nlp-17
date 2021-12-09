import torch
import torch.nn as nn
import numpy as np

import unittest

from src.modules.GhostBatchNorm import GhostBatchNorm
from src.modules.activations import Sparsemax

def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim+output_dim) / np.sqrt(4*input_dim))
    nn.init.xavier_normal_(module.weight, gain=gain_value)
    return

def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim+output_dim) / np.sqrt(input_dim))
    nn.init.xavier_normal_(module.weight, gain=gain_value)
    return

class EmbeddingGenerator(nn.Module):
    def __init__(self, input_dim, cat_dims, cat_idxs, cat_emb_dim):
        super(EmbeddingGenerator, self).__init__()
        if cat_dims == [] and cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            return
        elif (cat_dims == []) ^ (cat_idxs == []):
            if cat_dims == []:
                msg = "If cat_idxs is non-empty, cat_dims must be defined as a list of same length."
            else:
                msg = "If cat_dims is non-empty, cat_idxs must be defined as a list of same length."
            raise ValueError(msg)
        elif len(cat_dims) != len(cat_idxs):
            msg = "The lists cat_dims and cat_idxs must have the same length."
            raise ValueError(msg)

        self.skip_embedding = False
        if isinstance(cat_emb_dim, int):
            self.cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
        else:
            self.cat_emb_dims = cat_emb_dim

        # check that all embeddings are provided
        if len(self.cat_emb_dims) != len(cat_dims):
            msg = f"""cat_emb_dim and cat_dims must be lists of same length, got {len(self.cat_emb_dims)}
                      and {len(cat_dims)}"""
            raise ValueError(msg)
        self.post_embed_dim = int(
            input_dim + np.sum(self.cat_emb_dims) - len(self.cat_emb_dims)
        )

        self.embeddings = nn.ModuleList()

        # Sort dims by cat_idx
        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        self.cat_emb_dims = [self.cat_emb_dims[i] for i in sorted_idxs]

        for cat_dim, emb_dim in zip(cat_dims, self.cat_emb_dims):
            self.embeddings.append(nn.Embedding(cat_dim, emb_dim))

        # record continuous indices
        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[cat_idxs] = 0

    def forward(self, x):
        """
        Apply embeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            # no embeddings required
            return x

        cols = []
        cat_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            # Enumerate through continuous idx boolean mask to apply embeddings
            if is_continuous:
                cols.append(x[:, feat_init_idx].float().view(-1, 1))
            else:
                cols.append(
                    self.embeddings[cat_feat_counter](x[:, feat_init_idx].long())
                )
                cat_feat_counter += 1
        # concat
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings

class RandomObfuscator(nn.Module):
    def __init__(self, pretraining_ratio):
        super(RandomObfuscator, self).__init__()
        self.pretraining_ratio = pretraining_ratio

    def forward(self, x):
        obfuscated_vars = torch.bernoulli(
            self.pretraining_ratio * torch.ones(x.shape)
        ).to(x.device)
        masked_input = torch.mul(1 - obfuscated_vars, x)
        return masked_input, obfuscated_vars


class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, virtual_batch_size, momentum, device=None):
        super(AttentiveTransformer,self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        initialize_non_glu(self.fc, input_dim, output_dim)
        self.bn = GhostBatchNorm(output_dim, virtual_batch_size, momentum)
        self.selector = Sparsemax(dim=-1)
    
    def forward(self, priors, processed_feat):
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = torch.mul(x, priors)
        x = self.selector(x)
        return x

class GLU_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, virtual_batch_size, momentum, fc=None):
        super(GLU_Layer, self).__init__()

        self.output_dim = output_dim
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(input_dim, 2 * output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2 * output_dim)

        self.bn = GhostBatchNorm(2 * output_dim, virtual_batch_size, momentum)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        
        #GLU
        out = torch.mul(x[:, : self.output_dim], torch.sigmoid(x[:,self.output_dim :]))
        return out

class GLU_Block(nn.Module):
    def __init__(self, input_dim, output_dim, n_glu, first, shared_layers, virtual_batch_size, momentum):
        super(GLU_Block, self).__init__()
        self.first = first
        self.shared_layers = shared_layers
        self.n_glu = n_glu
        self.glu_layers = nn.ModuleList()

        params = {'virtual_batch_size': virtual_batch_size, 'momentum': momentum}

        fc = shared_layers[0] if shared_layers else None
        self.glu_layers.append(GLU_Layer(input_dim, output_dim, fc=fc, **params))
        for glu_id in range(1, self.n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(GLU_Layer(output_dim, output_dim, fc=fc, **params))

    def forward(self, x):
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))
        if self.first:
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)

        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x * scale
        return x

class FeatTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, shared_layers, n_glu_independent, virtual_batch_size, momentum):
        super(FeatTransformer,self).__init__()
        params = {
            "n_glu": n_glu_independent,
            "virtual_batch_size": virtual_batch_size,
            "momentum": momentum
        }

        if shared_layers is None:
            self.shared = nn.Identity()
            is_first = True
        else:
            self.shared = GLU_Block(
                input_dim,
                output_dim,
                first=True,
                shared_layers=shared_layers,
                n_glu=len(shared_layers),
                virtual_batch_size=virtual_batch_size,
                momentum=momentum,
                )
            is_first = False
        
        if n_glu_independent == 0:
            self.specifics = nn.Identity()
        else:
            spec_input_dim = input_dim if is_first else output_dim
            self.specifics = GLU_Block(spec_input_dim, output_dim, first=is_first, shared_layers=None, **params)
    
    def forward(self, x):
        x = self.shared(x)
        x = self.specifics(x)
        return x


class TabNetEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_d, n_a, n_steps, gamma, n_independent, n_shared, virtual_batch_size, momentum, epsilon=1e-15):
        super(TabNetEncoder,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_multi_task = isinstance(output_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.initial_bn = nn.BatchNorm1d(self.input_dim, momentum=0.01)

        if self.n_shared > 0:
            shared_feat_transform = nn.ModuleList()
            for i in range(self.n_shared):
                if i == 0:
                    shared_feat_transform.append(nn.Linear(self.input_dim, 2 * (n_d + n_a), bias=False))
                else:
                    shared_feat_transform.append(nn.Linear(n_d + n_a, 2 * (n_d + n_a), bias = False))
        else:
            shared_feat_transform = None
        
        self.initial_splitter = FeatTransformer(
            self.input_dim, 
            n_d + n_a, 
            shared_feat_transform, 
            n_glu_independent=self.n_independent,
            virtual_batch_size = self.virtual_batch_size,
            momentum = momentum
            )

        self.feat_transformers = nn.ModuleList()
        self.att_transformers = nn.ModuleList()

        for step in range(n_steps):
            transformer = FeatTransformer(
                self.input_dim,
                n_d + n_a,
                shared_feat_transform,
                n_glu_independent = self.n_independent,
                virtual_batch_size = virtual_batch_size,
                momentum = momentum,
            )
            attention = AttentiveTransformer(
                n_a,
                self.input_dim,
                virtual_batch_size = virtual_batch_size,
                momentum = momentum,
            )

            self.feat_transformers.append(transformer)
            self.att_transformers.append(attention)
    
    def forward(self, x, prior=None):
        x = self.initial_bn(x)

        if prior is None:
            prior = torch.ones(x.shape).to(x.device)
        
        M_loss = 0
        att = self.initial_splitter(x)[:, self.n_d : ]

        step_output = []
        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            M_loss += torch.mean(torch.sum(torch.mul(M, torch.log(M + self.epsilon)), dim=1))

            prior = torch.mul(self.gamma - M, prior)
            masked_x = torch.mul(M, x)
            out = self.feat_transformers[step](masked_x)
            d = nn.ReLU()(out[:, : self.n_d])
            step_output.append(d)
            att = out[:, self.n_d:]
        
        M_loss /= self.n_steps
        return step_output, M_loss
    def forward_masks(self, x):
        x = self.initial_bn(x)

        prior = torch.ones(x.shape).to(x.device)
        M_explain = torch.zeros(x.shape).to(x.device)
        att = self.initial_splitter(x)[:, self.n_d :]
        masks = {}

        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            masks[step] = M
            prior = torch.mul(self.gamma - M, prior)
            masked_x = torch.mul(M, x)
            out = self.feat_transformers[step](masked_x)
            d = nn.ReLU()(out[:, : self.n_d])
            step_importance = torch.sum(d, dim=1)
            M_explain += torch.mul(M, step_importance.unsqueeze(dim=1))
            att = out[:, self.n_d:]
        
        return M_explain, masks

class TabNetDecoder(nn.Module):
    def __init__(self, input_dim, n_d, n_steps, n_independent, n_shared, virtual_batch_size, momentum):
        super(TabNetDecoder,self).__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_steps = n_steps
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        
        self.feat_transformers = nn.ModuleList()

        if self.n_shared > 0:
            shared_feat_transform = nn.ModuleList()
            for i in range(self.n_shared):
                if i == 0:
                    shared_feat_transform.append(nn.Linear(n_d, 2 * n_d, bias = False))
                else:
                    shared_feat_transform.append(nn.Linear(n_d, 2 * n_d, bias = False))
        else:
            shared_feat_transform = None

        for step in range(n_steps):
            transformer = FeatTransformer(
                n_d,
                n_d,
                shared_feat_transform,
                n_glu_independent = self.n_independent,
                virtual_batch_size = virtual_batch_size,
                momentum = momentum,
            )
            self.feat_transformers.append(transformer)
        
        self.reconstruction_layer = nn.Linear(n_d, self.input_dim, bias=False)
        initialize_non_glu(self.reconstruction_layer, n_d, self.input_dim)
    
    def forward(self, steps_output):
        res = 0
        for step_nb, step_output in enumerate(steps_output):
            x = self.feat_transformers[step_nb](step_output)
            res = torch.add(res, x)
        res = self.reconstruction_layer(res)
        return res

class TabNetNoEmbeddings(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super(TabNetNoEmbeddings, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_multi_task = isinstance(output_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.initial_bn = nn.BatchNorm1d(self.input_dim, momentum=0.01)

        self.encoder = TabNetEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
        )

        if self.is_multi_task:
            self.multi_task_mappings = nn.ModuleList()
            for task_dim in output_dim:
                task_mapping = nn.Linear(n_d, task_dim, bias=False)
                initialize_non_glu(task_mapping, n_d, task_dim)
                self.multi_task_mappings.append(task_mapping)
        else:
            self.final_mapping = nn.Linear(n_d, output_dim, bias=False)
            initialize_non_glu(self.final_mapping, n_d, output_dim)

    def forward(self, x):
        res = 0
        steps_output, M_loss = self.encoder(x)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)

        if self.is_multi_task:
            # Result will be in list format
            out = []
            for task_mapping in self.multi_task_mappings:
                out.append(task_mapping(res))
        else:
            out = self.final_mapping(res)
        return out, M_loss

    def forward_masks(self, x):
        return self.encoder.forward_masks(x)

class TabNet(nn.Module):
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        n_d, 
        n_a, 
        n_steps, 
        gamma, 
        cat_idxs, 
        cat_dims, 
        cat_emb_dim, 
        n_independent, 
        n_shared, 
        virtual_batch_size, 
        momentum, 
        epsilon=1e-15
        ):
        super(TabNet,self).__init__()
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.ouptut_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim
        self.tabnet = TabNetNoEmbeddings(
            self.post_embed_dim,
            output_dim,
            n_d,
            n_a,
            n_steps,
            gamma,
            n_independent,
            n_shared,
            epsilon,
            virtual_batch_size,
            momentum,
        )
    def forward(self, x, self_train=False):
        x = self.embedder(x)
        return self.tabnet(x)
    def forward_masks(self, x):
        x = self.embedder(x)
        return self.tabnet.forward_masks(x)


class TabNetPretraining(nn.Module):
    def __init__(
        self,
        input_dim,
        pretraining_ratio=0.2,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        n_shared_decoder=1,
        n_indep_decoder=1,
    ):
        super(TabNetPretraining, self).__init__()

        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.pretraining_ratio = pretraining_ratio
        self.n_shared_decoder = n_shared_decoder
        self.n_indep_decoder = n_indep_decoder

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim

        self.masker = RandomObfuscator(self.pretraining_ratio)
        self.encoder = TabNetEncoder(
            input_dim=self.post_embed_dim,
            output_dim=self.post_embed_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
        )
        self.decoder = TabNetDecoder(
            self.post_embed_dim,
            n_d=n_d,
            n_steps=n_steps,
            n_independent=self.n_indep_decoder,
            n_shared=self.n_shared_decoder,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
        )

    def forward(self, x):
        embedded_x = self.embedder(x)
        if self.training:
            masked_x, obf_vars = self.masker(embedded_x)
            # set prior of encoder with obf_mask
            prior = 1 - obf_vars
            steps_out, _ = self.encoder(masked_x, prior=prior)
            res = self.decoder(steps_out)
            return res, embedded_x, obf_vars
        else:
            steps_out, _ = self.encoder(embedded_x)
            res = self.decoder(steps_out)
            return res, embedded_x, torch.ones(embedded_x.shape).to(x.device)

    def forward_masks(self, x):
        embedded_x = self.embedder(x)
        return self.encoder.forward_masks(embedded_x)
