"""DQN Agent -> Entrenamiento en Acrobot-v1 de Gymnasium."""
import random
from collections import deque

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

#  Hiperparámetros
ENV_ID = "Acrobot-v1"
SEED = 42
EPISODES = 500 # Cuántas partidas juega el agente para aprender.
GAMMA = 0.99 # Cuánto valora el futuro frente al presente.
# En este caso nos sirve bastante este GAMMA alto porque el objetivo es llegar a la meta (pendulo arriba) y eso requiere varios pasos.
LR = 5e-4
BATCH_SIZE = 64 # Cuántas experiencias usa en cada actualización. 
# Esto quiere decir que, una vez el buffer tiene al menos 1 000 transiciones, en cada paso del entorno el agente hace un paso de aprendizaje (backpropagation) usando un minibatch de 64 experiencias.

BUFFER_SIZE = 50_000 # Memoria de experiencias pasadas.
# El agente guarda las experiencias pasadas (estado, acción, recompensa, siguiente estado) en un buffer de tamaño 50 000.


START_LEARNING = 1_000      # pasos de relleno del buffer antes de aprender. No aprende hasts tener 1000 experiencias.

# Esto se hace porque al principio el agente no tiene experiencias y si aprende con pocas experiencias, se sobreajusta a ellas y no aprende bien.

TARGET_UPDATE = 500         # cada cuántos pasos se copia la red objetivo
# El target update es un truco para estabilizar el aprendizaje. En DQN, se usan dos redes: la red Q (la que aprende) y la red objetivo (target). 
# La red objetivo se actualiza cada cierto número de pasos copiando los pesos de la red Q. Esto ayuda a que el aprendizaje sea más estable y 
# no se desestabilice por cambios rápidos en la red Q.

EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 15_000 # Exploración: empieza actuando 100 % al azar y baja hasta 5 % en 15 000 pasos
HIDDEN = 128


class QNetwork(nn.Module):
    """6 entradas u obervaciones -> 128 -> 128 -> 3 salidas"""

    def __init__(self, n_obs, n_actions, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def greedy_eval(env, q_net, n=5):
    """Juega 5 partidas sin exploración (siempre la mejor acción según la red) y devuelve la 
    recompensa media. Con esto sabemos que tan bueno es el modelo en ese momento y poder guardar 
    el mejor, porque DQN es inestable y el último modelo no siempre es el mejor."""

    totals = []
    for i in range(n):
        state, _ = env.reset(seed=10_000 + i)
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                a = q_net(torch.tensor(state, dtype=torch.float32)).argmax().item()
            state, r, term, trunc, _ = env.step(a)
            total, done = total + r, term or trunc
        totals.append(total)
    return float(np.mean(totals))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make(ENV_ID)
    n_obs = env.observation_space.shape[0]   # La cantidad de observaciones es 6 (posición y velocidad de los dos links).
    n_actions = env.action_space.n           # 3, la cantidad de acciones posibles (torque en sentido horario, antihorario o ninguno).

# Creamos la red Q y la red objetivo, el optimizador y la memoria de experiencias.

#q_net (la que aprende) y target_net (copia congelada que sirve de referencia estable).

    q_net = QNetwork(n_obs, n_actions)
    target_net = QNetwork(n_obs, n_actions)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)
    buffer = deque(maxlen=BUFFER_SIZE)

    eval_env = gym.make(ENV_ID)
    step, rewards_log, best_eval = 0, [], -float("inf")

# Entrenamiento

# El agente juega EPISODES partidas, y en cada paso hace epsilon-greedy con decaimiento lineal.

# Se calcula eps (baja linealmente con los pasos). 
# Con probabilidad eps se elige una acción aleatoria (explorar); 
# si no, la que tiene mayor Q (explotar).

    for ep in range(EPISODES):
        state, _ = env.reset(seed=SEED + ep)
        total, done = 0.0, False

        while not done:
            # Epsilon-greedy con decaimiento lineal
            eps = max(EPS_END, EPS_START - step * (EPS_START - EPS_END) / EPS_DECAY_STEPS)
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = q_net(torch.tensor(state, dtype=torch.float32)).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action) # invocamos a step() para avanzar un paso en el entorno con la acción elegida.

            # La recompensa es -1 en cada paso hasta que el episodio termina (pendulo arriba) o se agotan los pasos.

            done = terminated or truncated
            # Solo 'terminated' corta el bootstrap; truncated no es un estado final real
            buffer.append((state, action, reward, next_state, float(terminated)))
            state, total, step = next_state, total + reward, step + 1

            if len(buffer) >= START_LEARNING:
                batch = random.sample(buffer, BATCH_SIZE)
                s, a, r, s2, d = map(np.array, zip(*batch))
                s = torch.tensor(s, dtype=torch.float32) # Estado actual
                a = torch.tensor(a, dtype=torch.int64).unsqueeze(1) # Acción tomada
                r = torch.tensor(r, dtype=torch.float32) # Recompensa recibida
                s2 = torch.tensor(s2, dtype=torch.float32) # Siguiente estado
                d = torch.tensor(d, dtype=torch.float32) # Indicador de si el episodio terminó (1 si terminó, 0 si no)

                q = q_net(s).gather(1, a).squeeze(1) # Q(s,a) para las acciones tomadas
                with torch.no_grad():
                    target = r + GAMMA * target_net(s2).max(1).values * (1 - d) # Q-learning target, sino solo r
                loss = nn.functional.smooth_l1_loss(q, target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()

            if step % TARGET_UPDATE == 0:
                target_net.load_state_dict(q_net.state_dict())

        rewards_log.append(total) # Guardamos la recompensa total de este episodio para graficar la curva de aprendizaje.
        if (ep + 1) % 10 == 0:
            score = greedy_eval(eval_env, q_net)
            if score > best_eval:       # guardamos el mejor modelo, no el último
                best_eval = score
                torch.save(q_net.state_dict(), "sabana_dqn_rl.pth")
            print(f"Episodio {ep+1:4d} | media entrenamiento (últ. 10): "
                  f"{np.mean(rewards_log[-10:]):7.1f} | eval greedy: {score:7.1f} "
                  f"| mejor: {best_eval:7.1f} | epsilon: {eps:.2f}")

# Guardamos recompensas.

    np.savetxt("rewards.csv", rewards_log, delimiter=",", header="reward")

    # Gráfica de aprendizaje
    plt.plot(rewards_log, alpha=0.4, label="Recompensa por episodio")
    ma = np.convolve(rewards_log, np.ones(20) / 20, mode="valid")
    plt.plot(range(19, len(rewards_log)), ma, label="Media móvil (20)")
    plt.xlabel("Episodio"); plt.ylabel("Recompensa total")
    plt.title("DQN en Acrobot-v1"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig("training_curve.png", dpi=150, bbox_inches="tight")
    print("Listo: sabana_dqn_rl.pth, rewards.csv y training_curve.png guardados.")


if __name__ == "__main__":
    main()
