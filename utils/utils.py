import random
import torch
import numpy as np

def neuron_to_cover(layer_neuron_activated_dic, cover_ratio):
    not_covered_all = [(layer_name, index) for (layer_name, index), v in layer_neuron_activated_dic.items() if not v]

    cover_size = (int) (len(not_covered_all) * cover_ratio)
    if not_covered_all:
        neurons_2be_covered = random.sample(not_covered_all, cover_size)
    else: # All neurons are activated
        neurons_2be_covered = []
    return neurons_2be_covered, not_covered_all

def cal_neurons_cov_loss(layers_output_dict, neurons_2be_covered):
    if len(neurons_2be_covered) == 0:
        return 0.0
    cov_loss = 0
    for (layer_name, neuron_idx) in neurons_2be_covered:
        cov_loss += torch.mean(layers_output_dict[layer_name][:,neuron_idx,...])
    return cov_loss #/ len(neurons_2be_covered)

def update_coverage(layers_output_dict, threshold=0):
    layer_neuron_activated_dic = {}

    total_activated_count = 0
    total_neuron_count = 0
    for layer_name in layers_output_dict.keys():
        # B X Num_neutrons X H X W
        scaled = scale(layers_output_dict[layer_name])

        # neurons_activation = [torch.mean(scaled[:,neuron_idx,...]) >= threshold for neuron_idx in range(scaled.shape[1])]
        # layer_activation_dic[layer] = neurons_activation
        activated_count = 0
        for neuron_idx in range(scaled.shape[1]):
            activated = torch.mean(scaled[:,neuron_idx,...]) > threshold
            if activated:
                activated_count += 1
            layer_neuron_activated_dic[(layer_name, neuron_idx)] = activated
        # print("Layer-%s %d/%d neurons are inactivated" % (layer_name, inactivated_count, scaled.shape[1]))
        total_activated_count += activated_count
        total_neuron_count += scaled.shape[1]

    return layer_neuron_activated_dic, total_activated_count, total_neuron_count

def update_coverage_v2(layers_output_dict, threshold=0, layer_neuron_activated_dic={}):
    total_activated_count = count_activated_neurons(layer_neuron_activated_dic)
    total_neuron_count = 0
    first = (len(layer_neuron_activated_dic) == 0)

    activated_cur_update = 0
    for layer_name in layers_output_dict.keys():
        # B X Num_neutrons X H X W
        scaled = scale(layers_output_dict[layer_name])

        # neurons_activation = [torch.mean(scaled[:,neuron_idx,...]) >= threshold for neuron_idx in range(scaled.shape[1])]
        # layer_activation_dic[layer] = neurons_activation
        activated_count = 0
        for neuron_idx in range(scaled.shape[1]):
            if (layer_name, neuron_idx) not in layer_neuron_activated_dic.keys():
                layer_neuron_activated_dic[(layer_name, neuron_idx)] = False
            
            activated = torch.mean(scaled[:,neuron_idx,...]).item() > threshold
            if activated and not layer_neuron_activated_dic[(layer_name, neuron_idx)]:
                layer_neuron_activated_dic[(layer_name, neuron_idx)] = True
                activated_count += 1

        activated_cur_update += activated_count
        total_neuron_count += scaled.shape[1]
    total_activated_count += activated_cur_update
    return layer_neuron_activated_dic, total_activated_count, total_neuron_count

def count_activated_neurons(layer_neuron_activated_dic):
    return np.sum(list(map(lambda x: x == True, layer_neuron_activated_dic.values())))

def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled