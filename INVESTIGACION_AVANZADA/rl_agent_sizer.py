import numpy as np
import pandas as pd
from pathlib import Path

class RLSizerAgent:
    """Prototipo de Agente de Aprendizaje por Refuerzo para Dimensionamiento de Posición"""
    def __init__(self, n_states=10, n_actions=5):
        # Q-Table simplificada: 
        # Estados: Combinación de Confianza IA (0-9) y Volatilidad (Baja/Alta)
        # Acciones: Multiplicador de posición (0.2x, 0.5x, 1.0x, 1.5x, 2.0x)
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 0.1
        self.actions = [0.2, 0.5, 1.0, 1.5, 2.0]

    def get_state(self, prob, vol):
        # Discretizamos la probabilidad (0.90 a 1.0) en 5 niveles
        p_bin = int(np.clip((prob - 0.90) / 0.02, 0, 4))
        # Discretizamos la volatilidad en 2 niveles
        v_bin = 1 if vol > 0.02 else 0
        return p_bin + (v_bin * 5)

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, len(self.actions))
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (target - predict)

def simulate_rl_training():
    print("Simulando entrenamiento de Agente de Refuerzo (RL)...")
    agent = RLSizerAgent()
    
    # Simulamos 1000 operaciones para que el agente aprenda
    for i in range(1000):
        # Generamos un escenario aleatorio
        prob = np.random.uniform(0.90, 1.0)
        vol = np.random.uniform(0.01, 0.05)
        state = agent.get_state(prob, vol)
        
        action_idx = agent.choose_action(state)
        multiplier = agent.actions[action_idx]
        
        # Lógica de recompensa:
        # Si la probabilidad es alta (>0.96) y la volatilidad es baja, suele ganar
        prob_exito = 0.4 + (prob - 0.90) * 5
        if vol > 0.03: prob_exito -= 0.2
        
        win = np.random.uniform(0, 1) < prob_exito
        gain = 0.02 if win else -0.015
        
        # Recompensa = Ganancia neta * Multiplicador
        reward = gain * multiplier * 100
        
        # En este prototipo, el siguiente estado es independiente (mercado eficiente)
        next_state = agent.get_state(np.random.uniform(0.90, 1.0), np.random.uniform(0.01, 0.05))
        
        agent.learn(state, action_idx, reward, next_state)
        
        if i % 200 == 0:
            print(f"  Op {i}: Multiplicador {multiplier}x -> Recompensa: {reward:.2f}")

    print("\nAPRENDIZAJE COMPLETADO. Mapa de Decisiones del Agente:")
    print("Estado (Confianza/Vol) | Mejor Multiplicador")
    for s in range(10):
        best_act = agent.actions[np.argmax(agent.q_table[s])]
        tipo_vol = "ALTA" if s >= 5 else "BAJA"
        nivel_conf = (s % 5) + 1
        print(f" Confianza {nivel_conf}/5 | Vol {tipo_vol} -> Apuesta: {best_act}x")

if __name__ == "__main__":
    simulate_rl_training()
