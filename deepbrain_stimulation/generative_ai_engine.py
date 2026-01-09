"""
Generative AI Engine for DBS Parameter Optimization
Implements VAE, GAN, and Reinforcement Learning for stimulation protocol generation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json


@dataclass
class StimulationParameters:
    """DBS stimulation parameters"""
    amplitude_ma: float
    frequency_hz: float
    pulse_width_us: float
    duty_cycle: float
    waveform_shape: str  # 'biphasic', 'monophasic'
    target_region: str  # 'amygdala', 'vmPFC', 'hippocampus'


class StimulationDataset(Dataset):
    """Synthetic dataset of stimulation parameters and outcomes"""
    
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
        self.data = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic stimulation data"""
        data = []
        
        for _ in range(self.num_samples):
            # Generate realistic parameter ranges
            amplitude = np.random.uniform(0.5, 8.0)
            frequency = np.random.choice([20, 60, 130, 185])  # Common DBS frequencies
            pulse_width = np.random.uniform(60, 210)
            duty_cycle = np.random.uniform(0.1, 0.9)
            
            # Simulate efficacy (higher for certain parameter combinations)
            efficacy = self._simulate_efficacy(amplitude, frequency, pulse_width, duty_cycle)
            
            # Simulate side effects
            side_effects = self._simulate_side_effects(amplitude, frequency, pulse_width)
            
            data.append({
                'parameters': np.array([amplitude, frequency, pulse_width, duty_cycle], dtype=np.float32),
                'efficacy': efficacy,
                'side_effects': side_effects
            })
        
        return data
    
    def _simulate_efficacy(self, amp, freq, pw, duty):
        """Simulate treatment efficacy (0-1)"""
        # Optimal around 130 Hz, 3 mA, 90 us based on literature
        freq_factor = np.exp(-((freq - 130) ** 2) / (2 * 50 ** 2))
        amp_factor = np.exp(-((amp - 3.0) ** 2) / (2 * 2.0 ** 2))
        pw_factor = np.exp(-((pw - 90) ** 2) / (2 * 40 ** 2))
        
        efficacy = freq_factor * amp_factor * pw_factor * duty
        return np.clip(efficacy + np.random.normal(0, 0.1), 0, 1)
    
    def _simulate_side_effects(self, amp, freq, pw):
        """Simulate side effects (0-1)"""
        # Higher amplitude and pulse width increase side effects
        side_effects = (amp / 8.0) * 0.5 + (pw / 210) * 0.3 + np.random.normal(0, 0.1)
        return np.clip(side_effects, 0, 1)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


# ==================== Variational Autoencoder ====================

class VAEEncoder(nn.Module):
    """VAE Encoder network"""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, latent_dim: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """VAE Decoder network"""
    
    def __init__(self, latent_dim: int = 8, hidden_dim: int = 64, output_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))  # Normalize to [0, 1]


class StimulationVAE(nn.Module):
    """Variational Autoencoder for stimulation pattern generation"""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, latent_dim: int = 8):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def generate(self, num_samples: int = 1):
        """Generate new stimulation parameters"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            samples = self.decoder(z)
        return samples
    
    def loss_function(self, recon_x, x, mu, logvar):
        """VAE loss: reconstruction + KL divergence"""
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss


# ==================== Generative Adversarial Network ====================

class Generator(nn.Module):
    """GAN Generator for waveform optimization"""
    
    def __init__(self, noise_dim: int = 16, hidden_dim: int = 128, output_dim: int = 4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Normalize to [0, 1]
        )
    
    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    """GAN Discriminator for waveform quality assessment"""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


class StimulationGAN:
    """GAN for optimized stimulation waveform generation"""
    
    def __init__(self, noise_dim: int = 16, hidden_dim: int = 128, output_dim: int = 4):
        self.generator = Generator(noise_dim, hidden_dim, output_dim)
        self.discriminator = Discriminator(output_dim, hidden_dim)
        self.noise_dim = noise_dim
        
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()
    
    def train_step(self, real_data):
        """Single training step"""
        batch_size = real_data.size(0)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real data
        real_labels = torch.ones(batch_size, 1)
        real_output = self.discriminator(real_data)
        d_loss_real = self.criterion(real_output, real_labels)
        
        # Fake data
        noise = torch.randn(batch_size, self.noise_dim)
        fake_data = self.generator(noise)
        fake_labels = torch.zeros(batch_size, 1)
        fake_output = self.discriminator(fake_data.detach())
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        noise = torch.randn(batch_size, self.noise_dim)
        fake_data = self.generator(noise)
        fake_output = self.discriminator(fake_data)
        g_loss = self.criterion(fake_output, real_labels)  # Want discriminator to think it's real
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return {'d_loss': d_loss.item(), 'g_loss': g_loss.item()}
    
    def generate(self, num_samples: int = 1):
        """Generate optimized stimulation parameters"""
        with torch.no_grad():
            noise = torch.randn(num_samples, self.noise_dim)
            samples = self.generator(noise)
        return samples


# ==================== Reinforcement Learning Agent ====================

class DQN(nn.Module):
    """Deep Q-Network for adaptive stimulation"""
    
    def __init__(self, state_dim: int = 8, action_dim: int = 12, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample random batch"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class AdaptiveStimulationAgent:
    """RL agent for adaptive DBS parameter optimization"""
    
    def __init__(self, state_dim: int = 8, action_dim: int = 12, hidden_dim: int = 128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.policy_net = DQN(state_dim, action_dim, hidden_dim)
        self.target_net = DQN(state_dim, action_dim, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99  # Discount factor
        
        # Action space: discretized parameter adjustments
        self.actions = self._define_action_space()
    
    def _define_action_space(self):
        """Define discrete actions for parameter adjustment"""
        return [
            {'amplitude_delta': 0.0, 'frequency_delta': 0, 'pulse_width_delta': 0},  # No change
            {'amplitude_delta': 0.5, 'frequency_delta': 0, 'pulse_width_delta': 0},  # Increase amplitude
            {'amplitude_delta': -0.5, 'frequency_delta': 0, 'pulse_width_delta': 0},  # Decrease amplitude
            {'amplitude_delta': 0, 'frequency_delta': 10, 'pulse_width_delta': 0},  # Increase frequency
            {'amplitude_delta': 0, 'frequency_delta': -10, 'pulse_width_delta': 0},  # Decrease frequency
            {'amplitude_delta': 0, 'frequency_delta': 0, 'pulse_width_delta': 10},  # Increase pulse width
            {'amplitude_delta': 0, 'frequency_delta': 0, 'pulse_width_delta': -10},  # Decrease pulse width
            {'amplitude_delta': 0.5, 'frequency_delta': 10, 'pulse_width_delta': 0},  # Combined increase
            {'amplitude_delta': -0.5, 'frequency_delta': -10, 'pulse_width_delta': 0},  # Combined decrease
            {'amplitude_delta': 0.5, 'frequency_delta': 0, 'pulse_width_delta': 10},
            {'amplitude_delta': -0.5, 'frequency_delta': 0, 'pulse_width_delta': -10},
            {'amplitude_delta': 0, 'frequency_delta': 10, 'pulse_width_delta': 10},
        ]
    
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    
    def train_step(self, batch_size: int = 32): # Reduced batch size for smaller datasets
        """Single training step"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        start_dim_check = self.replay_buffer.buffer[0][0].shape
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ==================== Unified Generative AI Engine ====================

class GenerativeAIEngine:
    """Unified engine for all generative AI models"""
    
    def __init__(self):
        self.vae = StimulationVAE(input_dim=4, hidden_dim=64, latent_dim=8)
        self.gan = StimulationGAN(noise_dim=16, hidden_dim=128, output_dim=4)
        self.rl_agent = AdaptiveStimulationAgent(state_dim=8, action_dim=12, hidden_dim=128)
        
        self.dataset = StimulationDataset(num_samples=1000)
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
    
    def train_vae(self, epochs: int = 100):
        """Train VAE model"""
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.001)
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in self.dataloader:
                params = batch['parameters']
                
                optimizer.zero_grad()
                recon, mu, logvar = self.vae(params)
                loss = self.vae.loss_function(recon, params, mu, logvar)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.dataloader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0 or epochs < 10:
                print(f"VAE Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def train_gan(self, epochs: int = 100):
        """Train GAN model"""
        losses = {'d_loss': [], 'g_loss': []}
        
        for epoch in range(epochs):
            epoch_d_loss = 0
            epoch_g_loss = 0
            
            for batch in self.dataloader:
                params = batch['parameters']
                step_losses = self.gan.train_step(params)
                epoch_d_loss += step_losses['d_loss']
                epoch_g_loss += step_losses['g_loss']
            
            avg_d_loss = epoch_d_loss / len(self.dataloader)
            avg_g_loss = epoch_g_loss / len(self.dataloader)
            losses['d_loss'].append(avg_d_loss)
            losses['g_loss'].append(avg_g_loss)
            
            if (epoch + 1) % 10 == 0 or epochs < 10:
                print(f"GAN Epoch {epoch+1}/{epochs}, D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
        
        return losses
    
    def generate_vae_parameters(self, num_samples: int = 10):
        """Generate parameters using VAE"""
        samples = self.vae.generate(num_samples)
        return self._denormalize_parameters(samples.numpy())
    
    def generate_gan_parameters(self, num_samples: int = 10):
        """Generate parameters using GAN"""
        samples = self.gan.generate(num_samples)
        return self._denormalize_parameters(samples.numpy())
    
    def optimize_with_rl(self, initial_state, num_steps: int = 100):
        """Optimize parameters using RL agent"""
        state = np.array(initial_state, dtype=np.float32)
        trajectory = [state.copy()]
        
        # Pre-fill buffer slightly to allow training to start
        if len(self.rl_agent.replay_buffer) < 32:
             for _ in range(32):
                 self.rl_agent.replay_buffer.push(state, 0, 0, state, False)

        for step in range(num_steps):
            action = self.rl_agent.select_action(state)
            # In real application, this would interact with environment
            # Here we simulate the next state
            next_state = self._simulate_next_state(state, action)
            reward = self._calculate_reward(next_state)
            done = step == num_steps - 1
            
            self.rl_agent.replay_buffer.push(state, action, reward, next_state, done)
            self.rl_agent.train_step(batch_size=32)
            
            state = next_state
            trajectory.append(state.copy())
            
            if (step + 1) % 10 == 0:
                self.rl_agent.update_target_network()
        
        return trajectory
    
    def _denormalize_parameters(self, normalized_params):
        """Convert normalized [0,1] to actual parameter ranges"""
        params_list = []
        for norm_param in normalized_params:
            params = {
                'amplitude_ma': float(norm_param[0] * 8.0),  # 0-8 mA
                'frequency_hz': float(norm_param[1] * 230 + 20),  # 20-250 Hz
                'pulse_width_us': float(norm_param[2] * 390 + 60),  # 60-450 us
                'duty_cycle': float(norm_param[3])  # 0-1
            }
            params_list.append(params)
        return params_list
    
    def _simulate_next_state(self, state, action):
        """Simulate next state placeholder"""
        # Ensure numpy array
        state = np.array(state, dtype=np.float32)
        # Add noise and slight drift towards stability (lower values) 
        # to simulate optimization success
        noise = np.random.normal(0, 0.02, size=state.shape)
        drift = -0.01 * state # Natural decay of symptoms
        
        # Action influence (mock): specific actions reduce specific dims
        action_effect = np.zeros_like(state)
        # Example: Action 0 reduces dim 0
        if action < 8:
            action_effect[action] = -0.05
            
        next_s = np.clip(state + noise + drift + action_effect, 0, 1)
        return next_s.astype(np.float32)
    
    def _calculate_reward(self, state):
        """Calculate reward based on state (placeholder)"""
        # In real application, this would be based on symptom reduction
        # Reward is negative of sum of severity metrics (indices 0-3 for symptoms usually)
        return float(-np.sum(state[:4]))
    
    def export_models(self, output_dir: str = "."):
        """Export trained models"""
        import os
        
        torch.save(self.vae.state_dict(), os.path.join(output_dir, "vae_model.pth"))
        torch.save(self.gan.generator.state_dict(), os.path.join(output_dir, "gan_generator.pth"))
        torch.save(self.gan.discriminator.state_dict(), os.path.join(output_dir, "gan_discriminator.pth"))
        torch.save(self.rl_agent.policy_net.state_dict(), os.path.join(output_dir, "rl_policy.pth"))
        
        print(f"Models exported to {output_dir}")


if __name__ == "__main__":
    # Example usage
    engine = GenerativeAIEngine()
    
    print("Training VAE...")
    vae_losses = engine.train_vae(epochs=50)
    
    print("\nTraining GAN...")
    gan_losses = engine.train_gan(epochs=50)
    
    print("\nGenerating parameters with VAE:")
    vae_params = engine.generate_vae_parameters(num_samples=5)
    for i, params in enumerate(vae_params):
        print(f"  Sample {i+1}: {params}")
    
    print("\nGenerating parameters with GAN:")
    gan_params = engine.generate_gan_parameters(num_samples=5)
    for i, params in enumerate(gan_params):
        print(f"  Sample {i+1}: {params}")
    
    # Export models
    engine.export_models()
