"""
Generative AI Models for Cell Morphology and Trajectory Synthesis
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class MorphologyVAE(nn.Module):
    """Variational autoencoder for learning cell morphology distributions"""
    
    def __init__(self, input_dim: int = 64*64, latent_dim: int = 32):
        super(MorphologyVAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        """VAE loss: reconstruction + KL divergence"""
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    def generate(self, n_samples: int = 1) -> torch.Tensor:
        """Generate new cell morphologies"""
        z = torch.randn(n_samples, self.latent_dim)
        with torch.no_grad():
            samples = self.decode(z)
        return samples


class TrajectoryGenerator(nn.Module):
    """Generator network for trajectory GAN"""
    
    def __init__(self, noise_dim: int = 100, condition_dim: int = 10,
                 trajectory_length: int = 50, state_dim: int = 3):
        super(TrajectoryGenerator, self).__init__()
        
        self.trajectory_length = trajectory_length
        self.state_dim = state_dim
        
        # LSTM for temporal generation
        self.lstm = nn.LSTM(
            input_size=noise_dim + condition_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        
        self.fc = nn.Linear(128, state_dim)
    
    def forward(self, noise, condition):
        """
        Generate trajectory
        
        Args:
            noise: Random noise (batch_size, trajectory_length, noise_dim)
            condition: Conditioning information (batch_size, condition_dim)
        
        Returns:
            Generated trajectory (batch_size, trajectory_length, state_dim)
        """
        batch_size = noise.size(0)
        
        # Expand condition to match trajectory length
        condition_expanded = condition.unsqueeze(1).repeat(1, self.trajectory_length, 1)
        
        # Concatenate noise and condition
        lstm_input = torch.cat([noise, condition_expanded], dim=2)
        
        # Generate through LSTM
        lstm_out, _ = self.lstm(lstm_input)
        
        # Map to state space
        trajectory = self.fc(lstm_out)
        
        return trajectory


class TrajectoryDiscriminator(nn.Module):
    """Discriminator network for trajectory GAN"""
    
    def __init__(self, trajectory_length: int = 50, state_dim: int = 3,
                 condition_dim: int = 10):
        super(TrajectoryDiscriminator, self).__init__()
        
        # LSTM for temporal discrimination
        self.lstm = nn.LSTM(
            input_size=state_dim + condition_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, trajectory, condition):
        """
        Discriminate real vs fake trajectories
        
        Args:
            trajectory: Input trajectory (batch_size, trajectory_length, state_dim)
            condition: Conditioning information (batch_size, condition_dim)
        
        Returns:
            Probability of being real (batch_size, 1)
        """
        batch_size = trajectory.size(0)
        trajectory_length = trajectory.size(1)
        
        # Expand condition
        condition_expanded = condition.unsqueeze(1).repeat(1, trajectory_length, 1)
        
        # Concatenate trajectory and condition
        lstm_input = torch.cat([trajectory, condition_expanded], dim=2)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(lstm_input)
        
        # Use last hidden state for classification
        output = self.fc(lstm_out[:, -1, :])
        
        return output


class TrajectoryGAN:
    """Conditional GAN for generating realistic cell trajectories"""
    
    def __init__(self, noise_dim: int = 100, condition_dim: int = 10,
                 trajectory_length: int = 50, state_dim: int = 3):
        self.generator = TrajectoryGenerator(noise_dim, condition_dim,
                                             trajectory_length, state_dim)
        self.discriminator = TrajectoryDiscriminator(trajectory_length,
                                                     state_dim, condition_dim)
        
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.trajectory_length = trajectory_length
        self.state_dim = state_dim
    
    def train_step(self, real_trajectories: torch.Tensor,
                   conditions: torch.Tensor,
                   optimizer_g, optimizer_d) -> Tuple[float, float]:
        """Single training step"""
        batch_size = real_trajectories.size(0)
        
        # Train discriminator
        optimizer_d.zero_grad()
        
        # Real trajectories
        real_labels = torch.ones(batch_size, 1)
        real_output = self.discriminator(real_trajectories, conditions)
        d_loss_real = F.binary_cross_entropy(real_output, real_labels)
        
        # Fake trajectories
        noise = torch.randn(batch_size, self.trajectory_length, self.noise_dim)
        fake_trajectories = self.generator(noise, conditions)
        fake_labels = torch.zeros(batch_size, 1)
        fake_output = self.discriminator(fake_trajectories.detach(), conditions)
        d_loss_fake = F.binary_cross_entropy(fake_output, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()
        
        # Train generator
        optimizer_g.zero_grad()
        
        noise = torch.randn(batch_size, self.trajectory_length, self.noise_dim)
        fake_trajectories = self.generator(noise, conditions)
        fake_output = self.discriminator(fake_trajectories, conditions)
        g_loss = F.binary_cross_entropy(fake_output, real_labels)
        
        g_loss.backward()
        optimizer_g.step()
        
        return g_loss.item(), d_loss.item()
    
    def generate_trajectories(self, conditions: torch.Tensor,
                             n_samples: int = 1) -> np.ndarray:
        """Generate synthetic trajectories"""
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(n_samples, self.trajectory_length, self.noise_dim)
            if conditions.dim() == 1:
                conditions = conditions.unsqueeze(0).repeat(n_samples, 1)
            
            trajectories = self.generator(noise, conditions)
        
        return trajectories.numpy()


class DifferentiationPredictor(nn.Module):
    """Predict differentiation outcomes from early time points"""
    
    def __init__(self, input_length: int = 10, state_dim: int = 3,
                 n_outcomes: int = 3):
        super(DifferentiationPredictor, self).__init__()
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_outcomes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, trajectory):
        """
        Predict differentiation outcome
        
        Args:
            trajectory: Early trajectory (batch_size, input_length, state_dim)
        
        Returns:
            Outcome probabilities (batch_size, n_outcomes)
        """
        # Encode trajectory
        lstm_out, _ = self.lstm(trajectory)
        
        # Use last hidden state
        encoded = lstm_out[:, -1, :]
        
        # Predict outcome
        outcome_probs = self.classifier(encoded)
        
        return outcome_probs


class SyntheticDataGenerator:
    """Generate augmented training data"""
    
    def __init__(self, vae: MorphologyVAE = None, gan: TrajectoryGAN = None):
        self.vae = vae
        self.gan = gan
    
    def generate_morphology_variations(self, base_morphology: np.ndarray,
                                      n_variations: int = 10) -> np.ndarray:
        """Generate morphological variations of a cell"""
        if self.vae is None:
            raise ValueError("VAE model not provided")
        
        self.vae.eval()
        
        # Encode base morphology
        base_tensor = torch.FloatTensor(base_morphology).unsqueeze(0)
        mu, logvar = self.vae.encode(base_tensor)
        
        # Generate variations by sampling around the mean
        variations = []
        for _ in range(n_variations):
            z = self.vae.reparameterize(mu, logvar)
            variation = self.vae.decode(z)
            variations.append(variation.detach().numpy())
        
        return np.array(variations)
    
    def augment_trajectories(self, trajectories: List[np.ndarray],
                            noise_level: float = 0.1) -> List[np.ndarray]:
        """Augment trajectories with noise and transformations"""
        augmented = []
        
        for traj in trajectories:
            # Add Gaussian noise
            noise = np.random.normal(0, noise_level, traj.shape)
            noisy_traj = traj + noise
            augmented.append(noisy_traj)
            
            # Time warping
            if len(traj) > 2:
                warp_factor = np.random.uniform(0.8, 1.2)
                t_old = np.arange(len(traj))
                t_new = np.linspace(0, len(traj) - 1, int(len(traj) * warp_factor))
                
                warped = np.zeros((len(t_new), traj.shape[1]))
                for dim in range(traj.shape[1]):
                    warped[:, dim] = np.interp(t_new, t_old, traj[:, dim])
                augmented.append(warped)
        
        return augmented
