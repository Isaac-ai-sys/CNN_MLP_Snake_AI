# CNN_MLP_Snake_AI
Convolutional neural network designed and trained to solve the game of snake

## Architecture

![Neural network architecture](architecture.svg)

The network uses a shared CNN feature extractor feeding into two separate heads:

- **Feature extractor** — two convolutional layers (depth 8 then 16, kernel 3×3) separated by a 2×2 max pool, then flattened and concatenated with auxiliary inputs (direction, length, food delta) before a dense layer outputs a 64-dim feature vector.
- **Actor head** — three dense layers (64→32→16→4) with a softmax output producing a probability distribution over the four actions (up, right, down, left).
- **Critic head** — two dense layers (64→32→1) with a linear output estimating the state value V(s).

Both heads are trained jointly using **Proximal Policy Optimisation (PPO)** with GAE advantage estimation.
