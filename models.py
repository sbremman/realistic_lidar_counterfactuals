
import torch.nn as nn
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import SAC
import gymnasium as gym


class CombinedActor(nn.Module):
    def __init__(self, features_extractor, latent_pi, mu):
        super().__init__()
        self.features_extractor = features_extractor
        # print("self.features_extractor", self.features_extractor)

        self.cnn = self.features_extractor.cnn[:2]
        # print("self.cnn", self.cnn)
        self.linear = self.features_extractor.linear
        self.latent_pi = latent_pi
        self.mu = mu
        self.tanh = nn.Tanh()
        self.lidar_dim = 180

    def no_flatten_features_extractor(self, observations):
        if observations.dim() == 1:  # for 1D tensor
            observations = observations.unsqueeze(0)
        cnn_input = observations[:, :self.lidar_dim].unsqueeze(1)
        cnn_output = self.cnn(cnn_input)

        cnn_output = th.reshape(cnn_output, (-1, 35))

        other_input = observations[:, self.lidar_dim:]

        # Concatenate the outputs
        combined_output = th.cat((cnn_output, other_input), dim=1)

        features = self.linear(combined_output)

        return features

    def forward(self, observations):
        # Extract features from observations
        """print("FLATTEN")
        features = self.features_extractor(observations)"""
        device = next(self.parameters()).device
        observations = observations.to(device)

        # print("NO FLATTEN")
        features = self.no_flatten_features_extractor(observations)

        # Compute policy logits (latent_pi)
        logits = self.latent_pi(features)

        # Compute mean action (mu)
        actions = self.mu(logits)

        # Apply Tanh activation
        actions = self.tanh(actions)

        return_val = actions.cpu().squeeze()

        return return_val


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128, cnn_dict: dict = {}):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        num_features = observation_space.shape[0]
        self.lidar_dim = cnn_dict['lidar_dim']
        self.other_features_dim = num_features - self.lidar_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_dict['out_channels'], kernel_size=cnn_dict['kernel_size'],
                      stride=cnn_dict['stride'], padding=cnn_dict['padding'], padding_mode='circular'),
            nn.ReLU(),
            nn.Flatten()
        )
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1] * cnn_dict[
                'out_channels']
        self.linear = nn.Sequential(nn.Linear(n_flatten + self.other_features_dim, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        cnn_input = observations[:, :self.lidar_dim].unsqueeze(1)
        cnn_output = self.cnn(cnn_input)
        other_input = observations[:, self.lidar_dim:]
        combined_output = th.cat((cnn_output, other_input), dim=1)
        final_output = self.linear(combined_output)
        return final_output



def get_icinco_model():
    cnn_dict = {
        'lidar_dim': 180,
        'out_channels': 1,
        'kernel_size': 20,
        'stride': 5,
        'padding': 5
    }

    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(cnn_dict=cnn_dict),
    )

    custom_objects = {
        "policy_kwargs": policy_kwargs,
    }

    model = SAC.load('models_and_data/ICINCO_model', custom_objects=custom_objects)
    policy_model = CombinedActor(model.actor.features_extractor, model.actor.latent_pi, model.actor.mu)

    return policy_model