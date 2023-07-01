from torch.utils import data
import numpy as np
import random
import torch
from copy import deepcopy
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import matplotlib.pyplot as plt

class NodeTypeDataset(data.Dataset):
    def __init__(self, env, node_type, state, pred_state, node_freq_mult,
                 scene_freq_mult, hyperparams, augment=False, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']

        self.augment = augment

        self.node_type = node_type
        self.edge_types = [edge_type for edge_type in env.get_edge_types() if edge_type[0] is node_type]
        self.index = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)
        self.len = len(self.index)

        # print(self.edge_types)

    def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
        index = list()
        for scene in self.env.scenes:
            present_node_dict = scene.present_nodes(np.arange(0, scene.timesteps), type=self.node_type, **kwargs)
            for t, nodes in present_node_dict.items():
                for node in nodes:
                    index += [(scene, t, node)] *\
                             (scene.frequency_multiplier if scene_freq_mult else 1) *\
                             (node.frequency_multiplier if node_freq_mult else 1)

        return index

    def __len__(self):
        return self.len

    def __getitem__(self, i, get_first_idx_only=False):
        (scene, t, node) = self.index[i]

        if self.augment:
            scene = scene.augment()
            node = scene.get_node_by_id(node.id)

        if not get_first_idx_only:
            return get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                      self.edge_types, self.max_ht, self.max_ft, self.hyperparams)
        else:
            first_history_index = (self.max_ht - node.history_points_at(t)).clip(0)
            return first_history_index

def get_node_timestep_data(env, scene, t, node, state, pred_state,
                                      edge_types, max_ht, max_ft, hyperparams,
                                      scene_graph=None):
    """
    The Original function in Trajectron++ which extracts the neighbors' state

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node: Node
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbours are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :param scene_graph: If scene graph was already computed for this scene and time you can pass it here
    :return: Batch Element
    """

    # Node
    timestep_range_x = np.array([t - max_ht, t])
    timestep_range_y = np.array([t + 1, t + max_ft])

    x = node.get(timestep_range_x, state[node.type])
    y = node.get(timestep_range_y, pred_state[node.type])
    first_history_index = (max_ht - node.history_points_at(t)).clip(0)
    x_st_t = deepcopy(x)
    x_st_t = x_st_t - x[-1]
    y_st_t = y

    x_t = torch.tensor(x, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.float)

    x_st_t = torch.tensor(x_st_t, dtype=torch.float)
    y_st_t = torch.tensor(y_st_t, dtype=torch.float)

    # Neighbors
    timestep_range_x_all = list(range(timestep_range_x[0], timestep_range_x[1]+1))
    temporal_scene_graph = [scene.get_scene_graph(t,
                                        env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter']) if scene_graph is None else scene_graph \
                            for t in timestep_range_x_all[first_history_index:]]

    scene_graph = temporal_scene_graph[-1]

    neighbors_data_st = dict()
    neighbors_data = dict()
    neighbors_edge_value = dict()
    for edge_type in edge_types:
        neighbors_data_st[edge_type] = list()
        neighbors_data[edge_type] = list()
        neighbors_edge_value[edge_type] = list()
        # We get all nodes which are connected to the current node for the current timestep
        connected_nodes = set(scene_graph.get_neighbors(node, edge_type[1]))
        for past_scene_graph in temporal_scene_graph[:-1]:
            past_connected_nodes = set(past_scene_graph.get_neighbors(node, edge_type[1]))
            connected_nodes.intersection_update(past_connected_nodes)

        for connected_node in connected_nodes:
            neighbor_state_np = connected_node.get(np.array([t - max_ht, t]),
                                                   state[connected_node.type],
                                                   # padding=0.0) # not show -> 0 as padding
                                                   padding=np.nan)  # not show -> nan as padding

            neighbors_data[edge_type].append(torch.tensor(neighbor_state_np, dtype=torch.float))

    return (first_history_index, x_t, y_t, x_st_t, y_st_t, None, neighbors_data, None, None, None)

