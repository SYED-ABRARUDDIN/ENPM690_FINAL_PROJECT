# Import necessary libraries
import os
import csv
import gym
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace  # NES environment for Super Mario
import gym_super_mario_bros  # Super Mario environment
import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical  # For stochastic action selection
from torchvision import transforms  # Image transformations
from gym.spaces import Box  # For defining observation spaces

# ---------------------
# Checkpoint & Logging
# ---------------------

def persist_weights(net, episode_idx, base_dir):
    """
    Save the neural network weights to a checkpoint file.
    
    Args:
        net: The neural network to save
        episode_idx: Current episode number
        base_dir: Base directory to save checkpoints
    """
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    filename = os.path.join(ckpt_dir, f"checkpoint_{episode_idx}.pth")
    torch.save(net.state_dict(), filename)
    print(f"✔ Saved weights to {filename}")

def load_weights(base_dir, trainer, target_episode=0):
    """
    Load network weights from a checkpoint file.
    
    Args:
        base_dir: Base directory containing checkpoints
        trainer: The trainer object to load weights into
        target_episode: Specific episode to load (0 means latest)
    """
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        print("⚠️  No checkpoints folder")
        return
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    if not files:
        print("⚠️  No .pth files found")
        return

    # Pick latest checkpoint or exact episode checkpoint
    if target_episode == 0:
        chosen = max(files, key=lambda fn: int(fn.split('_')[1].split('.')[0]))
    else:
        name = f"checkpoint_{target_episode}.pth"
        chosen = name if name in files else max(files, key=lambda fn: int(fn.split('_')[1].split('.')[0]))

    path = os.path.join(ckpt_dir, chosen)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(path, map_location=device)

    # Handle legacy model format - remap old policy_net/value_net → actor/critic if present
    state = {}
    for k,v in raw.items():
        if k.startswith("policy_net."):
            state["actor." + k[len("policy_net."):]] = v
        elif k.startswith("value_net."):
            state["critic." + k[len("value_net."):]] = v
        else:
            state[k] = v

    # Load weights into both networks
    trainer.net.load_state_dict(state)
    trainer.net_old.load_state_dict(state)
    trainer.net.to(device)
    trainer.net_old.to(device)
    trainer.current_episode = int(chosen.split('_')[1].split('.')[0])
    print(f"✅ Loaded {chosen} (episode {trainer.current_episode}) on {device}")

def append_log(log_dir, fname, episode, metric):
    """
    Append training metrics to a CSV log file.
    
    Args:
        log_dir: Directory to store logs
        fname: Log filename
        episode: Current episode number
        metric: Value to log (typically average reward)
    """
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, fname)
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([episode, metric])

# ---------------------
# Observation Wrappers
# ---------------------

class FrameReducer(gym.Wrapper):
    """
    Wrapper that repeats the same action for multiple frames and accumulates rewards.
    Helps with temporal consistency and speeds up training.
    """
    def __init__(self, env, repeat):
        super().__init__(env)
        self.repeat = repeat  # Number of frames to repeat each action

    def step(self, action):
        total_r, done, trunc, info = 0.0, False, False, {}
        # Repeat the same action for multiple frames
        for _ in range(self.repeat):
            obs, r, d, t, info = self.env.step(action)
            total_r += r  # Accumulate rewards
            trunc = trunc or t
            if d:  # Stop if the episode is done
                done = True
                break
        return obs, total_r, done, trunc, info

class GrayscaleProcessor(gym.ObservationWrapper):
    """
    Converts RGB observations to grayscale to reduce dimensionality.
    """
    def __init__(self, env):
        super().__init__(env)
        h, w, _ = env.observation_space.shape
        self.observation_space = Box(0, 255, (h, w), np.uint8)

    def observation(self, obs):
        arr = np.ascontiguousarray(obs)
        img = torch.from_numpy(arr).permute(2,0,1).float()  # Convert to torch tensor (C,H,W)
        return transforms.Grayscale()(img)  # Convert to grayscale

class Resizer(gym.ObservationWrapper):
    """
    Resizes observations to a fixed size and normalizes pixel values.
    """
    def __init__(self, env, size):
        super().__init__(env)
        self.size = (size, size)
        self.observation_space = Box(0,1.0,(1,size,size),np.float32)

    def observation(self, obs):
        pipeline = transforms.Compose([
            transforms.Resize(self.size),          # Resize image
            transforms.Normalize((0.0,), (255.0,)) # Normalize to [0,1]
        ])
        return pipeline(obs)

# ---------------------
# Actor-Critic Network
# ---------------------

class PolicyValueNetwork(nn.Module):
    """
    Combined policy (actor) and value (critic) network for PPO algorithm.
    Uses a CNN to process image observations.
    """
    def __init__(self, env):
        super().__init__()
        # Actor network outputs action probabilities
        self.actor = nn.Sequential(
            nn.Conv2d(4,32,8,4), nn.ReLU(),      # First convolutional layer
            nn.Conv2d(32,64,4,2), nn.ReLU(),     # Second convolutional layer
            nn.Conv2d(64,64,3,1), nn.ReLU(),     # Third convolutional layer
            nn.Flatten(),                         # Flatten for fully connected layers
            nn.Linear(3136,512), nn.ReLU(),      # Hidden layer
            nn.Linear(512, env.action_space.n)    # Output layer (action logits)
        )
        # Critic network estimates state value function
        self.critic = nn.Sequential(
            nn.Conv2d(4,32,8,4), nn.ReLU(),      # Same architecture as actor network
            nn.Conv2d(32,64,4,2), nn.ReLU(),     # but with separate parameters
            nn.Conv2d(64,64,3,1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136,512), nn.ReLU(),
            nn.Linear(512,1)                      # Output single value estimate
        )

    def forward(self, x):
        """
        Forward pass through both networks.
        
        Args:
            x: Input state (batch of observations)
            
        Returns:
            dist: Categorical distribution over actions
            val: Value function estimate
        """
        dist = Categorical(logits=self.actor(x))  # Convert logits to distribution
        val  = self.critic(x).view(-1)            # Get state value estimates
        return dist, val

# ---------------------
# PPO Trainer
# ---------------------

DEVICE = torch.device("cpu")  # Default device (can be changed to cuda)

class Trainer:
    """
    Proximal Policy Optimization (PPO) trainer implementation.
    Manages the training process including experience collection and policy updates.
    """
    def __init__(self, env, save_dir, gamma, lam, eps_clip,
                 epochs, mb_factor, batch_sz, lr_actor, lr_critic,
                 log_intv, render, verbose):
        """
        Initialize the PPO trainer.
        
        Args:
            env: The game environment
            save_dir: Directory to save model checkpoints
            gamma: Discount factor for future rewards
            lam: GAE lambda parameter
            eps_clip: PPO clipping parameter
            epochs: Number of epochs to train on each batch
            mb_factor: Mini-batch size divisor
            batch_sz: Total batch size for experience collection
            lr_actor: Learning rate for actor network
            lr_critic: Learning rate for critic network
            log_intv: Episode interval for logging
            render: Whether to render the environment
            verbose: Whether to print detailed logs
        """
        self.env = env
        self.save_dir = save_dir
        self.gamma = gamma
        self.lam = lam
        self.eps = eps_clip
        self.epochs = epochs
        self.batch_sz = batch_sz
        self.mb_size = batch_sz // mb_factor  # Mini-batch size
        self.log_intv = log_intv
        self.render = render
        self.verbose = verbose
        self.current_episode = 0
        self.ep_rewards = []       # Rewards for current episode
        self.ep_totals = []        # Total rewards for completed episodes

        # Initialize environment
        obs0 = env.reset()
        if isinstance(obs0, tuple): obs0 = obs0[0]  # Handle different gym versions
        self.state = np.squeeze(np.array(obs0), axis=1)

        # Initialize networks
        self.net     = PolicyValueNetwork(env).to(DEVICE)  # Current network
        self.net_old = PolicyValueNetwork(env).to(DEVICE)  # Target network
        self.net_old.load_state_dict(self.net.state_dict())
        
        # Separate learning rates for actor and critic
        self.opt = torch.optim.Adam([
            {'params': self.net.actor.parameters(),  'lr': lr_actor},
            {'params': self.net.critic.parameters(), 'lr': lr_critic}
        ], eps=1e-4)
        self.mse = nn.MSELoss()  # Loss function for critic

    def gather_experience(self):
        """
        Collect a batch of experiences by interacting with the environment.
        
        Returns:
            Dictionary containing states, actions, log probs, values, returns, and advantages
        """
        S, A, LP, V, R, D = [], [], [], [], [], []  # Initialize lists for experience storage
        obs_shape = self.state.shape
        batch_S = np.zeros((self.batch_sz,*obs_shape),np.float32)  # States
        batch_A = np.zeros(self.batch_sz, np.int64)                # Actions
        batch_LP= np.zeros(self.batch_sz, np.float32)              # Log probabilities
        batch_V = np.zeros(self.batch_sz, np.float32)              # Values
        batch_R = np.zeros(self.batch_sz, np.float32)              # Rewards
        batch_D = np.zeros(self.batch_sz, bool)                    # Done flags

        # Collect batch_sz steps of experience
        for t in range(self.batch_sz):
            with torch.no_grad():
                batch_S[t] = self.state
                # Get action and value from target network
                dist, val = self.net_old(torch.tensor(self.state,device=DEVICE).unsqueeze(0))
                batch_V[t] = val.item()
                act = dist.sample()  # Sample action from distribution
                batch_A[t]  = act.item()
                batch_LP[t] = dist.log_prob(act).item()  # Log probability for sampled action

            # Take action in environment
            nxt, rew, done, trunc, _ = self.env.step(batch_A[t])
            if isinstance(nxt, tuple): nxt = nxt[0]
            self.state = np.squeeze(np.array(nxt), axis=1)

            # Record reward and done flag
            self.ep_rewards.append(rew)
            batch_R[t] = rew
            batch_D[t] = done or trunc

            # Handle episode completion
            if done:
                self.current_episode += 1
                total = sum(self.ep_rewards)
                self.ep_totals.append(total)
                self.ep_rewards = []
                self.env.reset()
                
                # Log progress periodically
                if self.current_episode % self.log_intv == 0:
                    avg = np.mean(self.ep_totals[-10:])
                    print(f"[Episode {self.current_episode}] avg reward: {avg:.2f}")
                    append_log(self.save_dir, "metrics.csv", self.current_episode, avg)
                    persist_weights(self.net_old, self.current_episode, self.save_dir)

        # Compute advantages and returns for PPO update
        returns, advs = self._compute_advantages(batch_D, batch_R, batch_V)
        return {
            'states':    torch.tensor(batch_S, dtype=torch.float32, device=DEVICE),
            'actions':   torch.tensor(batch_A, device=DEVICE),
            'old_logp':  torch.tensor(batch_LP, device=DEVICE),
            'values':    torch.tensor(batch_V, device=DEVICE),
            'returns':   torch.tensor(returns, dtype=torch.float32, device=DEVICE),
            'advantages':torch.tensor(advs, dtype=torch.float32, device=DEVICE)
        }

    def _compute_advantages(self, done, rewards, values):
        """
        Compute generalized advantage estimates (GAE) and returns.
        
        Args:
            done: Array of done flags
            rewards: Array of rewards
            values: Array of value estimates
            
        Returns:
            returns: Discounted returns
            adv: Advantage estimates
        """
        # Get next state value for bootstrap
        with torch.no_grad():
            _, next_val = self.net_old(torch.tensor(self.state,device=DEVICE).unsqueeze(0))
        vals = np.append(values, next_val.item())
        
        # Calculate GAE (generalized advantage estimation)
        gae  = 0.0
        ret  = []
        for i in reversed(range(len(rewards))):
            mask = 1.0 - float(done[i])  # Zero out advantage if episode is done
            delta = rewards[i] + self.gamma * vals[i+1] * mask - vals[i]  # TD error
            gae   = delta + self.gamma * self.lam * mask * gae  # Recursive GAE calculation
            ret.insert(0, gae + vals[i])  # Return = GAE + value
            
        # Normalize advantages
        adv = np.array(ret) - vals[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # Normalize for stability
        return ret, adv

    def update_policy(self, batch):
        """
        Update the policy network using the PPO algorithm.
        
        Args:
            batch: Dictionary of collected experience
        """
        # Randomize batch order
        idxs = torch.randperm(self.batch_sz)
        
        # Process mini-batches
        for start in range(0, self.batch_sz, self.mb_size):
            mb = {k: v[idxs[start:start+self.mb_size]] for k,v in batch.items()}
            # Perform multiple epochs on each mini-batch
            for _ in range(self.epochs):
                loss = self._ppo_loss(mb)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            # Copy updated weights to target network after each mini-batch
            self.net_old.load_state_dict(self.net.state_dict())

    def _ppo_loss(self, mb):
        """
        Calculate PPO loss function.
        
        Args:
            mb: Mini-batch of experience
            
        Returns:
            Combined loss value
        """
        # Get current action probabilities and values
        dist, val = self.net(mb['states'])
        
        # Calculate probability ratio for PPO clipping
        ratio = torch.exp(dist.log_prob(mb['actions']) - mb['old_logp'])
        clipped = ratio.clamp(1-self.eps, 1+self.eps)
        
        # PPO objective: minimum of surrogates
        pol_obj = torch.min(ratio * mb['advantages'], clipped * mb['advantages'])
        
        # Entropy bonus for exploration
        ent = dist.entropy()
        
        # Value loss
        v_loss = self.mse(val, mb['returns'])
        
        # Combined loss (note the negative signs as we're minimizing)
        return (-pol_obj + 0.5*v_loss - 0.01*ent).mean()

# ---------------------
# Entrypoint
# ---------------------

if __name__ == "__main__":
    # Configuration parameters
    base_dir      = "./model"           # Directory to save/load models
    gamma, lam    = 0.95, 0.95          # Discount factors
    eps_clip      = 0.2                 # PPO clipping parameter
    epochs        = 30                  # Epochs per mini-batch
    mb_factor     = 4                   # Mini-batch size divisor
    batch_sz      = 4096                # Experience batch size
    lr_actor      = 2.5e-4              # Actor learning rate
    lr_critic     = 1e-3                # Critic learning rate
    log_interval  = 100                 # Episodes between logging
    render_flag   = True                # Whether to render game
    start_ep      = 6100                # Starting episode (for loading)
    max_episodes  = 10000               # Max training episodes

    # Setup environment with wrappers
    env = gym.make(
        "SuperMarioBros-1-1-v0",
        apply_api_compatibility=True,
        render_mode="human" if render_flag else None
    )
    env = JoypadSpace(env, [["right"], ["right","A"]])  # Simplify action space (right and jump)
    env = FrameReducer(env, repeat=4)                   # Reduce frame rate
    env = GrayscaleProcessor(env)                       # Convert to grayscale
    env = Resizer(env, size=84)                         # Resize to 84x84
    env = FrameStack(env, num_stack=4)                  # Stack 4 frames for temporal information

    # Initialize trainer
    trainer = Trainer(
        env, base_dir, gamma, lam, eps_clip,
        epochs, mb_factor, batch_sz,
        lr_actor, lr_critic,
        log_interval, render_flag, verbose=False
    )

    # Load existing weights if available
    load_weights(base_dir, trainer, start_ep)

    # Choose between training and evaluation
    TRAIN = False
    if TRAIN:
        # Training loop
        for ep in range(start_ep+1, max_episodes+1):
            batch = trainer.gather_experience()    # Collect experience
            trainer.update_policy(batch)           # Update policy
        print(f"🏁 Finished up to episode {max_episodes}")
    else:
        # Evaluation/demo loop (run forever)
        while True:
            print("▶ Running policy")
            trainer.gather_experience()  # Just execute policy without updates
