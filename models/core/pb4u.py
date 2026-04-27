import functools
from collections import defaultdict

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.typing import Size

from models import networks
from models.core.base import BaseBlock, make_edgesets_dict

class GraphNetBlock(BaseBlock):
    def __init__(self, node_processor_fn, edge_processor_fn, edge_sets, latent_size):

        self.edge_sets = edge_sets
        self.latent_size = latent_size
        self.edge_keys = sorted(list(set([v['edge_key'] for v in edge_sets.values()])))
        edge_processor_dict = {v['edge_key']: edge_processor_fn for v in self.edge_sets.values()}
        node_processor_dict = dict(node=node_processor_fn)
        super().__init__(edge_processor_dict, node_processor_dict)

    def get_updated_edge_features(self, sample):
        updated_features = {}

        for edgeset_key, edgeset in self.edge_sets.items():
            edge_features_updated = self.update_edge_features(sample, edgeset['edge_key'], edgeset['source'],
                                                              edgeset['edge_key'], edgeset['target'])
            updated_features[edgeset_key] = edge_features_updated

        return updated_features

    def aggregate_node_features(self, sample, updated_edge_features):
        nodeset_input_features = defaultdict(dict)

        for edgeset_key, edgeset in self.edge_sets.items():
            source_nodes = sample[edgeset['source']]
            target_nodes = sample[edgeset['target']]

            N_source = source_nodes.node_features.shape[0]
            N_target = target_nodes.node_features.shape[0]
            size = (N_source, N_target)

            edge_index = sample[edgeset['source'], edgeset['edge_key'], edgeset['target']].edge_index
            edge_features = updated_edge_features[edgeset_key]
            aggregated_features = self.aggregate_nodes(edge_features, edge_index, size)

            nodeset_input_features[edgeset['target']][edgeset['edge_key']] = aggregated_features

        return nodeset_input_features

    def get_updated_node_features(self, sample, nodeset_input_features):
        updated_node_features = dict()
        for node_key, features_dict in nodeset_input_features.items():
            nodes = sample[node_key]

            node_features = nodes.node_features
            device = node_features.device
            N_nodes = node_features.shape[0]

            input_features_list = [node_features]

            for edge_key in self.edge_keys:
                if edge_key in features_dict:
                    input_features_list.append(features_dict[edge_key])
                else:
                    dummy_features = torch.zeros(N_nodes, self.latent_size, device=device)
                    input_features_list.append(dummy_features)

            updated_features = self.update(input_features_list, 'node')
            updated_node_features[node_key] = updated_features

        return updated_node_features

    def update_node_features_sample(self, sample, updated_node_features):
        for node_key, updated_features in updated_node_features.items():
            nodes = sample[node_key]
            prev_features = nodes.node_features
            sample[node_key].node_features = prev_features + updated_features

        return sample

    def update_edge_features_sample(self, sample, updated_edge_features):
        for edgeset_key, edgeset in self.edge_sets.items():
            prev_features = sample[edgeset['source'], edgeset['edge_key'], edgeset['target']].features
            update_features = updated_edge_features[edgeset_key]

            sample[edgeset['source'], edgeset['edge_key'], edgeset['target']].features = prev_features + update_features

        return sample
    
    def detach_state(self, sample):
        for node_type in sample.node_types:
            sample[node_type].node_features = sample[node_type].node_features.detach()

        for edgeset_key, edgeset in self.edge_sets.items():
            source = edgeset['source']
            target = edgeset['target']
            edge_key = edgeset['edge_key']
            sample[source, edge_key, target].features = sample[source, edge_key, target].features.detach()
        
        return sample

    def propagate(self, sample, detach: bool = False, size: Size = None, **kwargs):
        # FOR EACH EDGE SET GET FEATURES
        updated_edge_features = self.get_updated_edge_features(sample)

        # FOR EACH NODE TYPE
        # FOR EACH EDGESET
        # AGGREGATE FEATURES
        nodeset_input_features = self.aggregate_node_features(sample, updated_edge_features)

        # FOR EACH NODE TYPE
        # PASS AGGREGATED FEATURES THROUGH MLP TO GET UPDATED
        updated_node_features = self.get_updated_node_features(sample, nodeset_input_features)

        # FOR EACH NODE TYPE
        # UPDATE FEATURES
        sample = self.update_node_features_sample(sample, updated_node_features)

        # FOR EACH EDGESET
        # UPDATE FEATURES
        sample = self.update_edge_features_sample(sample, updated_edge_features)

        return sample
    
class GraphAggrBlock(BaseBlock):
    def __init__(self, node_processor_fn, edge_processor_fn, edge_sets, latent_size, num_layers):

        self.edge_sets = edge_sets
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.edge_keys = sorted(list(set([v['edge_key'] for v in edge_sets.values()])))
        edge_processor_dict = {v['edge_key']: edge_processor_fn for v in self.edge_sets.values()}
        node_processor_dict = dict(node=node_processor_fn)

        super().__init__(edge_processor_dict, node_processor_dict)

        self.message_mlp = self._make_mlp(self.latent_size * 3, self.latent_size)
        self.fuse_mlp = self._make_mlp(self.latent_size * 2, self.latent_size)
        self.norm = nn.LayerNorm(self.latent_size)
        self.gamma = 0.9

    def _make_mlp(self, input_size: int, output_size: int, layer_norm: bool = True) -> nn.Module:
        """Builds an MLP."""
        widths = [input_size] + [self.latent_size] * self.num_layers + [output_size]
        network = networks.MLP(widths, activate_final=None)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(output_size))
        return network

    def propagate_node_features(self, sample, repeat):
        source_nodes = sample['cloth']
        target_nodes = sample['cloth']

        N_source = source_nodes.node_features.shape[0]
        N_target = target_nodes.node_features.shape[0]
        size = (N_source, N_target)
        edge_index = sample['cloth', 'mesh_edge', 'cloth'].edge_index
        edge_features = sample['cloth', 'mesh_edge', 'cloth'].features
        
        target_nodes.aggregated_features = target_nodes.node_features
        source_features = source_nodes.node_features[edge_index[0]]
        target_features = target_nodes.node_features[edge_index[1]]
        
        # Sum
        for i in range(repeat):
            message_features = self.message_mlp(torch.cat([source_features, edge_features, target_features], dim=-1))
            aggregated_features = self.aggregate_nodes(message_features, edge_index, size)
            aggregated_features = self.gamma * target_nodes.aggregated_features + aggregated_features
            target_nodes.aggregated_features = self.norm(aggregated_features)
            source_features = source_nodes.aggregated_features[edge_index[0]]
            target_features = target_nodes.aggregated_features[edge_index[1]]
        return sample
    
    def update_node_features(self, sample):
        aggregated_features = sample['cloth'].aggregated_features
        node_features = sample['cloth'].node_features
        updated_features = self.fuse_mlp(torch.cat([node_features, aggregated_features], dim=-1))
        sample['cloth'].node_features = updated_features
        return sample
    
    def forward(self, sample, repeat):
        sample = self.propagate(sample, repeat)
        return sample
                
    def propagate(self, sample, repeat, detach: bool = False, size: Size = None, **kwargs):
        # Aggregate features
        sample = self.propagate_node_features(sample, repeat)

        # Update node features
        sample = self.update_node_features(sample)
        
        return sample

class EncodeProcessDecode(nn.Module):
    def __init__(self, mcfg):
        """Encode-Process-Decode GraphNet model."""
        super().__init__()
        self._latent_size = mcfg.latent_size
        self._output_size = mcfg.output_size
        self._num_layers = mcfg.num_layers
        self.n_nodefeatures = mcfg.n_nodefeatures
        self.n_edgefeatures_mesh = mcfg.n_edgefeatures_mesh
        self.n_edgefeatures_world = mcfg.n_edgefeatures_world
        self._message_passing_steps = mcfg.message_passing_steps

        self.node_encoder = self._make_mlp(self.n_nodefeatures, self._latent_size)
        self.decoder = self._make_mlp(self._latent_size, self._output_size, layer_norm=False)

        self.detach = mcfg.detach
        self.save_at = mcfg.save_at

        edgeset_encoders = {}
        edgeset_encoders['mesh'] = self._make_mlp(self.n_edgefeatures_mesh, self._latent_size)
        edgeset_encoders['world'] = self._make_mlp(self.n_edgefeatures_world, self._latent_size)
        self.edgeset_encoders = nn.ModuleDict(edgeset_encoders)

        node_proc_model = functools.partial(self._make_mlp, input_size=self._latent_size * (1 + 2),
                                            output_size=self._latent_size)
        edge_proc_model = functools.partial(self._make_mlp, input_size=self._latent_size * 3,
                                            output_size=self._latent_size)

        self.edge_sets_full = make_edgesets_dict(0)
        edgesets_list = ['world_direct', 'world_inverse', 'mesh']
        edgeset_dict = {k: self.edge_sets_full[k] for k in edgesets_list}

        self.aggr_processor = GraphAggrBlock(node_proc_model, edge_proc_model, edgeset_dict, self._latent_size, self._num_layers)
        processor_steps = []
        for i in range(self._message_passing_steps):
            processor_steps.append(GraphNetBlock(node_proc_model, edge_proc_model, edgeset_dict, self._latent_size))
        self.processor_steps = nn.ModuleList(processor_steps)

    def _make_mlp(self, input_size: int, output_size: int, layer_norm: bool = True) -> nn.Module:
        """Builds an MLP."""
        widths = [input_size] + [self._latent_size] * self._num_layers + [output_size]
        network = networks.MLP(widths, activate_final=None)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(output_size))
        return network
    
    def _encode_nodes(self, sample):
        cloth_features = sample['cloth'].node_features
        obstacle_features = sample['obstacle'].node_features
        obstacle_active_mask = sample['obstacle'].active_mask[:, 0]
        obstacle_features_active = obstacle_features[obstacle_active_mask]

        N_cloth = cloth_features.shape[0]
        N_obstacle = obstacle_features.shape[0]

        combined_features = torch.cat([cloth_features, obstacle_features_active], dim=0)
        combined_latents = self.node_encoder(combined_features)

        cloth_latents = combined_latents[:N_cloth]
        obstacle_active_latents = combined_latents[N_cloth:]
        latent_features = obstacle_active_latents.shape[1]

        obstacle_latents = torch.zeros(N_obstacle, latent_features).to(obstacle_active_latents.device)
        obstacle_latents[obstacle_active_mask] = obstacle_active_latents

        sample['cloth'].node_features = cloth_latents
        sample['obstacle'].node_features = obstacle_latents

        return sample

    def _encode_edges(self, sample):
        mesh_edge_features = sample['cloth', 'mesh_edge', 'cloth'].features
        mesh_edge_latents = self.edgeset_encoders['mesh'](mesh_edge_features)
        sample['cloth', 'mesh_edge', 'cloth'].features = mesh_edge_latents

        cloth_edge_features_direct = sample['cloth', 'world_edge', 'obstacle'].features
        cloth_edge_features_inverse = sample['obstacle', 'world_edge', 'cloth'].features
        N_world_edges = cloth_edge_features_direct.shape[0]

        cloth_edge_features_cat = torch.cat([cloth_edge_features_direct, cloth_edge_features_inverse], dim=0)
        cloth_edge_latents_cat = self.edgeset_encoders['world'](cloth_edge_features_cat)
        cloth_edge_latents_direct = cloth_edge_latents_cat[:N_world_edges]
        cloth_edge_latents_inverse = cloth_edge_latents_cat[N_world_edges:]
        sample['cloth', 'world_edge', 'obstacle'].features = cloth_edge_latents_direct
        sample['obstacle', 'world_edge', 'cloth'].features = cloth_edge_latents_inverse
        return sample

    def _encode(self, sample: Batch) -> Batch:
        """Encodes node and edge features into latent features."""
        sample = self._encode_nodes(sample)
        sample = self._encode_edges(sample)
        return sample

    def _decode(self, sample):
        """Decodes node features from graph."""
        cloth_features = sample['cloth'].node_features
        out_features = self.decoder(cloth_features)
        sample['cloth'].node_features = out_features
        return sample

    def forward(self, sample, repeat) -> torch.Tensor:
        """Encodes and processes a multigraph, and returns node features."""
        sample = self._encode(sample)

        sample = self.aggr_processor(sample, repeat)

        for i in range(self._message_passing_steps):
            sample = self.processor_steps[i](sample)

        return self._decode(sample)