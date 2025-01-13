import pygame
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy

# Configurar dispositivo: usar siempre la CPU
device = torch.device("cpu")

# Definición de la red neuronal para el agente
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Memoria de repeticiones
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Hiperparámetros
gamma = 0.99
batch_size = 64
learning_rate = 0.001
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 500
memory_capacity = 10000
generations = 100
parallel_games = 5

# Configuración del juego
frame_size_x = 720
frame_size_y = 480
state_dim = 11
action_dim = 4
difficulty = 15  # Velocidad del juego (FPS)

# Utilidades para guardar y cargar redes neuronales
def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))

# Entrenamiento basado en refuerzo
class SnakeAI:
    def __init__(self):
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.memory = ReplayMemory(memory_capacity)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.steps_done = 0

    def select_action(self, state):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                  np.exp(-1. * self.steps_done / epsilon_decay)
        self.steps_done += 1
        if random.random() > epsilon:
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1).view(1, 1)
        else:
            return torch.tensor([[random.randrange(action_dim)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)

        state_batch = torch.cat(state_batch)
        action_batch = torch.cat(action_batch)
        reward_batch = torch.cat(reward_batch)
        next_state_batch = torch.cat(next_state_batch)
        done_batch = torch.tensor(done_batch, device=device, dtype=torch.float32)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

        loss = nn.functional.smooth_l1_loss(q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Clase para gestionar el entorno del juego
class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.food_pos = [random.randrange(1, (frame_size_x // 10)) * 10,
                         random.randrange(1, (frame_size_y // 10)) * 10]
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0

    def step(self, action):
        change_to = self.direction
        if action == 0 and self.direction != 'DOWN':
            change_to = 'UP'
        elif action == 1 and self.direction != 'UP':
            change_to = 'DOWN'
        elif action == 2 and self.direction != 'RIGHT':
            change_to = 'LEFT'
        elif action == 3 and self.direction != 'LEFT':
            change_to = 'RIGHT'
        self.direction = change_to

        if self.direction == 'UP':
            self.snake_pos[1] -= 10
        if self.direction == 'DOWN':
            self.snake_pos[1] += 10
        if self.direction == 'LEFT':
            self.snake_pos[0] -= 10
        if self.direction == 'RIGHT':
            self.snake_pos[0] += 10

        self.snake_body.insert(0, list(self.snake_pos))
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.food_spawn = False
            self.score += 1
        else:
            self.snake_body.pop()

        if not self.food_spawn:
            self.food_pos = [random.randrange(1, (frame_size_x // 10)) * 10,
                             random.randrange(1, (frame_size_y // 10)) * 10]
        self.food_spawn = True

        done = False
        if (self.snake_pos[0] < 0 or self.snake_pos[0] >= frame_size_x or
                self.snake_pos[1] < 0 or self.snake_pos[1] >= frame_size_y or
                self.snake_pos in self.snake_body[1:]):
            done = True

        reward = 1.0 if self.snake_pos == self.food_pos else -0.1
        return self.get_state(), reward, done

    def get_state(self):
        state = [
            int(self.snake_pos[0] > self.food_pos[0]),
            int(self.snake_pos[0] < self.food_pos[0]),
            int(self.snake_pos[1] > self.food_pos[1]),
            int(self.snake_pos[1] < self.food_pos[1]),
            int(self.snake_pos in self.snake_body[1:]),
            int(self.snake_pos[0] < 0 or self.snake_pos[0] >= frame_size_x),
            int(self.snake_pos[1] < 0 or self.snake_pos[1] >= frame_size_y),
            int(self.direction == 'UP'),
            int(self.direction == 'DOWN'),
            int(self.direction == 'LEFT'),
            int(self.direction == 'RIGHT')
        ]
        return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    def render(self, screen):
        screen.fill((0, 0, 0))
        for pos in self.snake_body:
            pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(pos[0], pos[1], 10, 10))
        pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))
        pygame.display.flip()

# Modo de entrenamiento
def train():
    pygame.init()
    for generation in range(generations):
        agents = [SnakeAI() for _ in range(parallel_games)]
        games = [SnakeGame() for _ in range(parallel_games)]
        screens = [pygame.display.set_mode((frame_size_x, frame_size_y)) for _ in range(parallel_games)]
        scores = []

        for agent, game, screen in zip(agents, games, screens):
            done = False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                game.render(screen)
                state = game.get_state()
                action = agent.select_action(state).item()
                next_state, reward, done = game.step(action)
                agent.memory.push((state, torch.tensor([[action]], device=device),
                                   torch.tensor([reward], device=device), next_state, done))
                agent.optimize_model()

                pygame.time.wait(difficulty)  # Control de velocidad

            scores.append(game.score)

        # Identificar las dos mejores redes neuronales
        best_indices = np.argsort(scores)[-2:]  # Índices de los dos mejores
        print(f"Generation {generation}, Top Scores: {scores[best_indices[0]]}, {scores[best_indices[1]]}")

        # Guardar los modelos de las dos mejores redes
        for i, idx in enumerate(best_indices):
            save_model(agents[idx].policy_net, f"best_agent_gen{generation}_{i}.pth")

        # Crear nuevos agentes basados en los mejores
        new_agents = [copy.deepcopy(agents[i]) for i in best_indices]
        for i in range(parallel_games):
            if i not in best_indices:
                agents[i] = copy.deepcopy(new_agents[random.randint(0, 1)])

    pygame.quit()

# Modo de ejecución
def play(model_name):
    pygame.init()
    screen = pygame.display.set_mode((frame_size_x, frame_size_y))
    game = SnakeGame()
    agent = SnakeAI()
    model_path = os.path.join(os.getcwd(), model_name)  # Construye la ruta desde el directorio actual
    load_model(agent.policy_net, model_path)
    done = False

    while not done:
        game.render(screen)
        state = game.get_state()
        action = agent.policy_net(state).argmax(dim=1).item()
        _, _, done = game.step(action)
        pygame.time.wait(difficulty)

    pygame.quit()

# Elegir modo
mode = input("Elige modo (train/play): ").strip().lower()
if mode == "train":
    train()
elif mode == "play":
    model_name = input("Nombre del archivo del modelo (incluyendo extensión): ").strip()
    play(model_name)
else:
    print("Modo no válido")

