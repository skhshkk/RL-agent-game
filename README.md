# RL Agent Food Fight Game 🎮🤖

## 📌 Overview
The **RL Agent Food Fight Game** is an interactive 2D game built using **Pygame**, where a human player competes against an **AI agent trained using Reinforcement Learning (Q-learning)**.  
The AI learns strategic behaviors such as grabbing food, dodging incoming attacks, repositioning, and throwing with adaptive difficulty settings.

---

## 🚀 Features
- Reinforcement Learning–based AI agent (Q-learning)
- Dynamic difficulty levels: Easy, Medium, Hard
- Real-time AI training mode
- Strategic decision-making (dodge, reposition, attack)
- Physics-based projectile motion
- Health system, particles, screen shake, and animations
- Human vs AI gameplay

---

## 🧠 Reinforcement Learning Details
- **Algorithm:** Q-Learning  
- **State Space:**  
  - Distance to player  
  - Relative player position  
  - AI health bucket  
  - Food availability  
  - Imminent threat detection  
- **Action Space:**  
  - Get food  
  - Throw  
  - Dodge  
  - Reposition  
- **Reward Strategy:**  
  - Positive reward for successful hits and grabs  
  - Negative reward for getting hit or missing throws  

The Q-table is saved and reused across sessions for continuous learning.

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Game Engine:** Pygame  
- **AI & Math:** NumPy, Reinforcement Learning (Q-learning)  
- **Concurrency:** Threading  
- **Graphics & Effects:** Particle systems, screen shake, animations  

---

## 🎮 Controls
- **← / →** : Move player  
- **Space (Hold)** : Charge throw  
- **Space (Release)** : Throw food  
- **D** : Dodge  
- **R** : Restart round  
- **T** : Train AI in background  
- **1 / 2 / 3** : Change AI difficulty (Easy / Medium / Hard)

---

## ▶️ How to Run

bash
# Clone the repository
git clone https://github.com/your-username/rl-agent-food-fight.git

# Navigate to the project folder
cd rl-agent-food-fight

# Install dependencies
pip install pygame numpy

# Run the game
python main.py


📊 Training Mode
Press T during gameplay to start AI training in the background.
A progress bar indicates training status, and the trained Q-table is saved for future runs.


👤 Author
Sudershan Kharwade
Machine Learning | Reinforcement Learning | Generative AI Enthusiast


