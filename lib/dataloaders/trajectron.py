from torch.utils import data
import numpy as np
import random 
import torch
from copy import deepcopy

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
                    valid = True
                    data = [(scene, t, node)] *\
                             (scene.frequency_multiplier if scene_freq_mult else 1) *\
                             (node.frequency_multiplier if node_freq_mult else 1)
                    (scene, t, node) = data[0]
                    if self.augment:
                        scene = scene.augment()
                        node = scene.get_node_by_id(node.id)
                    first_history_index, x_t, y_t, x_st_t, y_st_t,scene_name, timestep = get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,\
                                      self.edge_types, self.max_ht, self.max_ft, self.hyperparams)

                    all_t = torch.cat((x_t[:,:2], y_t),dim=0)
                    if valid:
                        index += [ (first_history_index, x_t, y_t, x_st_t, y_st_t,scene_name, timestep)]
                    else: 
                        pass
        return index

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        (first_history_index, x_t, y_t, x_st_t, y_st_t,scene_name, timestep) = self.index[i]
        return first_history_index, x_t, y_t, x_st_t, y_st_t,scene_name, timestep


def get_node_timestep_data(env, scene, t, node, state, pred_state,
                           edge_types, max_ht, max_ft, hyperparams,
                           scene_graph=None):
    """
    Pre-processes the data for a single batch element: node state over time for a specific time in a specific scene
    as well as the neighbour data for it.

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

    return (first_history_index, x_t, y_t, x_st_t, y_st_t, scene.name, t)


def get_node_timestep_data_w_neighbor(env, scene, t, node, state, pred_state,
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

    _, std = env.get_standardize_params(state[node.type], node.type)
    std[0:2] = env.attention_radius[(node.type, node.type)]
    rel_state = np.zeros_like(x[0])
    rel_state[0:2] = np.array(x)[-1, 0:2]
    x_st = env.standardize(x, state[node.type], node.type, mean=rel_state, std=std)
    if list(pred_state[node.type].keys())[0] == 'position':  # If we predict position we do it relative to current pos
        y_st = env.standardize(y, pred_state[node.type], node.type, mean=rel_state[0:2])
    else:
        y_st = env.standardize(y, pred_state[node.type], node.type)

    x_t = torch.tensor(x, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.float)
    x_st_t = torch.tensor(x_st, dtype=torch.float)
    y_st_t = torch.tensor(y_st, dtype=torch.float)

    # Neighbors
    neighbors_data_st = None
    neighbors_edge_value = None
    if hyperparams['edge_encoding']:
        # Scene Graph
        scene_graph = scene.get_scene_graph(t,
                                            env.attention_radius,
                                            hyperparams['edge_addition_filter'],
                                            hyperparams['edge_removal_filter']) if scene_graph is None else scene_graph

        neighbors_data_st = dict()
        neighbors_edge_value = dict()
        for edge_type in edge_types:
            neighbors_data_st[edge_type] = list()
            # We get all nodes which are connected to the current node for the current timestep
            connected_nodes = scene_graph.get_neighbors(node, edge_type[1])

            if hyperparams['dynamic_edges'] == 'yes':
                # We get the edge masks for the current node at the current timestep
                edge_masks = torch.tensor(scene_graph.get_edge_scaling(node), dtype=torch.float)
                neighbors_edge_value[edge_type] = edge_masks

            for connected_node in connected_nodes:
                neighbor_state_np = connected_node.get(np.array([t - max_ht, t]),
                                                       state[connected_node.type],
                                                       padding=0.0)

                # Make State relative to node where neighbor and node have same state
                _, std = env.get_standardize_params(state[connected_node.type], node_type=connected_node.type)
                std[0:2] = env.attention_radius[edge_type]
                equal_dims = np.min((neighbor_state_np.shape[-1], x.shape[-1]))
                rel_state = np.zeros_like(neighbor_state_np)
                rel_state[:, ..., :equal_dims] = x[-1, ..., :equal_dims]
                neighbor_state_np_st = env.standardize(neighbor_state_np,
                                                       state[connected_node.type],
                                                       node_type=connected_node.type,
                                                       mean=rel_state,
                                                       std=std)

                neighbor_state = torch.tensor(neighbor_state_np_st, dtype=torch.float)
                neighbors_data_st[edge_type].append(neighbor_state)

    # Robot
    robot_traj_st_t = None
    timestep_range_r = np.array([t, t + max_ft])
    if hyperparams['incl_robot_node']:
        x_node = node.get(timestep_range_r, state[node.type])
        if scene.non_aug_scene is not None:
            robot = scene.get_node_by_id(scene.non_aug_scene.robot.id)
        else:
            robot = scene.robot
        robot_type = robot.type
        robot_traj = robot.get(timestep_range_r, state[robot_type], padding=np.nan)
        robot_traj_st_t = get_relative_robot_traj(env, state, x_node, robot_traj, node.type, robot_type)
        robot_traj_st_t[torch.isnan(robot_traj_st_t)] = 0.0

    # Map
    map_tuple = None
    if hyperparams['use_map_encoding']:
        if node.type in hyperparams['map_encoder']:
            if node.non_aug_node is not None:
                x = node.non_aug_node.get(np.array([t]), state[node.type])
            me_hyp = hyperparams['map_encoder'][node.type]
            if 'heading_state_index' in me_hyp:
                heading_state_index = me_hyp['heading_state_index']
                # We have to rotate the map in the opposit direction of the agent to match them
                if type(heading_state_index) is list:  # infer from velocity or heading vector
                    heading_angle = -np.arctan2(x[-1, heading_state_index[1]],
                                                x[-1, heading_state_index[0]]) * 180 / np.pi
                else:
                    heading_angle = -x[-1, heading_state_index] * 180 / np.pi
            else:
                heading_angle = None

            scene_map = scene.map[node.type]
            map_point = x[-1, :2]

            patch_size = hyperparams['map_encoder'][node.type]['patch_size']
            map_tuple = (scene_map, map_point, heading_angle, patch_size)

    return (first_history_index, x_t, y_t, x_st_t, y_st_t, neighbors_data_st,
            neighbors_edge_value, robot_traj_st_t, map_tuple)

