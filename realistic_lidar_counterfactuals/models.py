import zipfile
import torch.nn as nn
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import SAC
import gymnasium as gym
import json


class CombinedActor(nn.Module):
    def __init__(self, features_extractor, latent_pi, mu):
        super().__init__()
        # Initialize components from given feature extractor and policy layers
        self.features_extractor = features_extractor
        self.cnn = self.features_extractor.cnn
        self.linear = self.features_extractor.linear
        self.latent_pi = latent_pi
        self.mu = mu
        self.tanh = nn.Tanh()
        self.lidar_dim = 180

    def no_flatten_features_extractor(self, observations):
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)

        cnn_input = observations[:, :self.lidar_dim].unsqueeze(1)
        cnn_output = self.cnn(cnn_input)

        output_size = cnn_output.size(-1)
        cnn_output = torch.reshape(cnn_output, (-1, output_size))

        other_input = observations[:, self.lidar_dim:]

        combined_output = torch.cat((cnn_output, other_input), dim=1)
        features = self.linear(combined_output)
        return features

    def forward(self, observations):
        device = next(self.parameters()).device
        observations = observations.to(device)

        with torch.no_grad():
            features = self.no_flatten_features_extractor(observations)
            logits = self.latent_pi(features)
            actions = self.mu(logits)
            actions = self.tanh(actions)
            return_val = actions.cpu().squeeze()
        return return_val


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128, cnn_dict: dict = {}):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        num_features = observation_space.shape[0]

        self.lidar_dim = cnn_dict['lidar_dim']
        self.other_features_dim = num_features - self.lidar_dim

        cnn_layers = []
        in_channels = 1

        if isinstance(cnn_dict['out_channels'], int):
            cnn_dict['out_channels'] = [cnn_dict['out_channels']]
            cnn_dict['kernel_size'] = [cnn_dict['kernel_size']]
            cnn_dict['stride'] = [cnn_dict['stride']]
            cnn_dict['padding'] = [cnn_dict['padding']]

        for i in range(len(cnn_dict['out_channels'])):
            out_channels = cnn_dict['out_channels'][i]
            kernel_size = cnn_dict['kernel_size'][i]
            stride = cnn_dict['stride'][i]
            padding = cnn_dict['padding'][i]

            cnn_layers.append(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, padding_mode='circular'))
            cnn_layers.append(nn.ReLU())
            in_channels = out_channels

        cnn_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn_layers)

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()).float()
            sample = sample[None, :self.lidar_dim].unsqueeze(1)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + self.other_features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cnn_input = observations[:, :self.lidar_dim].unsqueeze(1)
        cnn_output = self.cnn(cnn_input)

        other_input = observations[:, self.lidar_dim:]

        combined_output = torch.cat((cnn_output, other_input), dim=1)
        final_output = self.linear(combined_output)
        return final_output


def load_model(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open('data') as data_file:
            model_data = json.load(data_file)

    policy_kwargs = model_data['policy_kwargs']
    cnn_dict = policy_kwargs['features_extractor_kwargs']['cnn_dict']

    if 'net_arch' in policy_kwargs:
        if ':type:' in policy_kwargs:
            del policy_kwargs[':type:']
            del policy_kwargs[':serialized:']
        policy_kwargs['features_extractor_class'] = CustomFeatureExtractor
        policy_kwargs['features_extractor_kwargs'] = dict(cnn_dict=cnn_dict)
    else:
        policy_kwargs = dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(cnn_dict=cnn_dict)
        )

    custom_objects = {
        "policy_kwargs": policy_kwargs,
    }
    model = SAC.load(zip_path, custom_objects=custom_objects)

    policy_model = CombinedActor(model.actor.features_extractor, model.actor.latent_pi, model.actor.mu)
    return policy_model, model
