import pygame
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
from tqdm import tqdm


# Configurar dispositivo: usar siempre la CPU
device = torch.device("cpu")

# Variables de configuración
num_generations = 400  # Número de generaciones (-1 para infinito)
agents_per_generation = 100  # Número de agentes por generación
num_best_agents = 50  # Número de agentes que se conservan entre generaciones
show_render = False  # Mostrar el render durante el entrenamiento

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
    # Ahora usamos weights_only=True para evitar posibles riesgos de seguridad
    model.load_state_dict(torch.load(filename, map_location=device))

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
    best_agents = []  # Almacena los mejores agentes en general (no solo por generación)

    generation = 0
    while num_generations == -1 or generation < num_generations:
        # Crear la nueva generación a partir de los mejores agentes
        if best_agents:
            # Crear 100 agentes hijos a partir de los 30 mejores agentes
            agents = []
            for _ in range(agents_per_generation):
                parent = random.choice(best_agents)[0]  # Seleccionar un agente padre
                child = copy.deepcopy(parent)  # Clonar al agente
                # Mutación de los pesos de la red neuronal del agente
                for param in child.policy_net.parameters():
                    if random.random() < 0.1:  # 10% de probabilidad de mutación
                        noise = torch.normal(0, 0.1, size=param.data.size()).to(device)
                        param.data += noise
                agents.append(child)
        else:
            # Primera generación: crear agentes desde cero
            agents = [SnakeAI() for _ in range(agents_per_generation)]

        # Preparar los juegos para todos los agentes
        games = [SnakeGame() for _ in range(agents_per_generation)]
        screens = [pygame.display.set_mode((frame_size_x, frame_size_y)) if show_render else None
                   for _ in range(agents_per_generation)]
        scores = []

        # Barra de progreso para la simulación de los agentes
        for agent, game, screen in zip(tqdm(agents, desc=f"Generación {generation}", ncols=100), games, screens):
            done = False
            while not done:
                if show_render and screen:
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

                if show_render:
                    pygame.time.wait(difficulty)  # Control de velocidad

            scores.append(game.score)

        # Combinar los mejores agentes históricos con los de la generación actual
        all_agents = best_agents + list(zip(agents, scores))
        all_agents = sorted(all_agents, key=lambda x: x[1], reverse=True)[:num_best_agents]
        best_agents = all_agents

        # Obtener el agente con la mejor puntuación
        best_agent_index = scores.index(max(scores))  # Índice del agente con la mejor puntuación
        best_score = max(scores)  # Mejor puntuación

        # Guardar los mejores agentes, sobrescribiendo archivos existentes
        os.makedirs("agentes", exist_ok=True)
        for i, (agent, _) in enumerate(best_agents):
            save_model(agent.policy_net, f"agentes/best_agent_{i}.pth")

        # Imprimir detalles de la generación
        print(f"Generación {generation} completada. Mejor puntaje: {best_score} de Agente #{best_agent_index}")

        generation += 1

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
