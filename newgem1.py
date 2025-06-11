import pygame
import sys
import random
import math
import numpy as np
import threading
import os
from collections import defaultdict

# --- Constants & Initialization ---
pygame.init()
pygame.mixer.init()

# Screen and UI
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
FPS = 60
PLAYER_SIZE = 40
FOOD_SIZE = 25
FLOOR_Y = SCREEN_HEIGHT - 60
HEALTH_BAR_WIDTH, HEALTH_BAR_HEIGHT = 150, 20
BACKGROUND_COLOR = (210, 230, 255)
PLAYER_COLOR = (70, 130, 180)
AI_COLOR = (220, 60, 50)
HEALTH_COLOR = (50, 205, 50)
TEXT_COLOR = (25, 25, 112)
FOOD_COLORS = [(255, 99, 71), (255, 165, 0), (0, 128, 0), (75, 0, 130), (255, 20, 147)]
SPLAT_COLORS = [(178, 34, 34), (205, 133, 63), (106, 90, 205)]
GRAVITY = 0.35

# ### --- STRATEGY/SPEED V3 --- ###
# - Capping AI's max speed directly with `max_speed`.
# - Re-balancing `aim_error` and `dodge_chance` to make it a better shot but less mobile.
AI_DIFFICULTY_SETTINGS = {
    "EASY":   {"aim_error": 120, "dodge_chance": 0.35, "min_throw_power": 30, "post_throw_cooldown": 50, "max_speed": 3},
    "MEDIUM": {"aim_error": 60,  "dodge_chance": 0.60, "min_throw_power": 50, "post_throw_cooldown": 25, "max_speed": 5},
    "HARD":   {"aim_error": 10,  "dodge_chance": 0.95, "min_throw_power": 80, "post_throw_cooldown": 5,  "max_speed": 8} # Player's theoretical max is ~10
}
CURRENT_DIFFICULTY = "EASY"

# Create screen and clock
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Food Fight Frenzy! (Strategic AI)")
clock = pygame.time.Clock()

# Font setup
try:
    font = pygame.font.SysFont("Comic Sans MS", 30)
    small_font = pygame.font.SysFont("Comic Sans MS", 20)
    title_font = pygame.font.SysFont("Comic Sans MS", 40, bold=True)
    damage_font = pygame.font.SysFont("Comic Sans MS", 26, bold=True)
except:
    font, small_font, title_font, damage_font = [pygame.font.Font(None, s) for s in [36, 24, 48, 32]]

# --- Utility Classes ---
# ... [No changes needed in ScreenShaker, ParticleSystem, FloatingText, Food] ...
class ScreenShaker:
    def __init__(self):
        self.offset = pygame.Vector2(0, 0)
        self.shake_magnitude = 0
        self.shake_duration = 0

    def start(self, magnitude, duration):
        self.shake_magnitude = magnitude
        self.shake_duration = duration

    def update(self):
        if self.shake_duration > 0:
            self.shake_duration -= 1
            self.offset.x = random.randint(-self.shake_magnitude, self.shake_magnitude)
            self.offset.y = random.randint(-self.shake_magnitude, self.shake_magnitude)
        else:
            self.offset.x, self.offset.y = 0, 0

shaker = ScreenShaker()

class ParticleSystem:
    def __init__(self):
        self.particles = []

    def emit(self, pos, type):
        if type == 'hit':
            for _ in range(12):
                angle = random.uniform(0, 2*math.pi)
                speed = random.uniform(2, 5)
                self.particles.append({'pos': pos.copy(), 'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed, 'radius': random.randint(4, 8), 'color': (255, 215, 0), 'life': 15})
        elif type == 'grab':
             for _ in range(8):
                angle = random.uniform(0, 2*math.pi)
                speed = random.uniform(1, 3)
                self.particles.append({'pos': pos.copy(), 'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed, 'radius': random.randint(3, 6), 'color': (200,200,200), 'life': 10})

    def update(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0: self.particles.remove(p)

    def draw(self, surface, offset):
        for p in self.particles:
            s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            alpha = int(255 * (p['life'] / 15))
            pygame.draw.circle(s, (*p['color'], alpha), (p['radius'], p['radius']), p['radius'])
            surface.blit(s, p['pos'] - pygame.Vector2(p['radius'], p['radius']) + offset)

particles = ParticleSystem()

class FloatingText:
    def __init__(self, text, pos, color):
        self.text, self.pos, self.color = text, pygame.Vector2(pos), color
        self.vel_y, self.lifetime = -2, 40
    def update(self):
        self.pos.y += self.vel_y; self.vel_y += 0.1; self.lifetime -= 1; return self.lifetime > 0
    def draw(self, surface, offset):
        text_surf = damage_font.render(self.text, True, self.color)
        text_surf.set_alpha(int(255 * (self.lifetime / 40)))
        surface.blit(text_surf, self.pos + offset)

# --- Game Classes ---
class Food:
    def __init__(self):
        self.pos = pygame.Vector2(random.randint(50, SCREEN_WIDTH - 50), -FOOD_SIZE)
        self.vel = pygame.Vector2(random.uniform(-1, 1), 0)
        self.size = FOOD_SIZE; self.color = random.choice(FOOD_COLORS)
        self.state = "falling"; self.splat_particles = []
        self.rotation, self.rotation_speed = 0, random.uniform(-7, 7)
        self.owner = None

    def throw(self, start_pos, power, target_pos, owner):
        self.state = "flying"; self.pos = start_pos
        self.owner = owner
        time_to_target = 60 - power * 0.3
        self.vel.x = (target_pos.x - self.pos.x) / time_to_target
        self.vel.y = (target_pos.y - self.pos.y) / time_to_target - 0.5 * GRAVITY * time_to_target
        self.vel.y = max(-12, self.vel.y)

    def update(self):
        if self.state in ["falling", "flying"]:
            self.vel.y += GRAVITY; self.pos += self.vel
            self.rotation = (self.rotation + self.rotation_speed) % 360
            if self.pos.y >= FLOOR_Y - self.size and self.state == "falling":
                self.pos.y = FLOOR_Y - self.size; self.vel.y *= -0.5
                if abs(self.vel.y) < 1: self.state = "landed"; self.vel = pygame.Vector2(0,0)
            if not (0 < self.pos.x < SCREEN_WIDTH): self.create_splat(); return "miss"
        
        for p in self.splat_particles[:]:
            p['pos'] += p['vel']; p['vel'].y += 0.2; p['life'] -= 1
            if p['life'] <= 0: self.splat_particles.remove(p)
        if self.state == "splatted" and not self.splat_particles: return "done"
        return None

    def create_splat(self):
        self.state = "splatted"; splat_color = random.choice(SPLAT_COLORS)
        for _ in range(15):
            angle = random.uniform(0, 2*math.pi)
            speed = random.uniform(1, 4)
            self.splat_particles.append({'pos': self.pos.copy(), 'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed, 'color': splat_color, 'radius': random.randint(3,8), 'life': 20})

    def draw(self, surface, offset):
        if self.state in ["falling", "flying", "landed"]:
            if self.pos.y < FLOOR_Y - 5:
                shadow_size = int(self.size * 0.8 * (1 - self.pos.y / FLOOR_Y))
                shadow_surf = pygame.Surface((shadow_size*2, shadow_size*2), pygame.SRCALPHA)
                pygame.draw.circle(shadow_surf, (0,0,0, 50), (shadow_size, shadow_size), shadow_size)
                surface.blit(shadow_surf, (self.pos.x - shadow_size + offset.x, FLOOR_Y - shadow_size + offset.y))
            
            food_surface = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(food_surface, self.color, (self.size, self.size), self.size)
            rotated = pygame.transform.rotate(food_surface, self.rotation)
            surface.blit(rotated, self.pos - pygame.Vector2(rotated.get_width()//2, rotated.get_height()//2) + offset)
        
        for p in self.splat_particles: pygame.draw.circle(surface, p['color'], p['pos'] + offset, int(p['radius'] * (p['life']/20)))

class Player:
    def __init__(self, x, color, is_ai=False):
        self.pos = pygame.Vector2(x, FLOOR_Y); self.vel_x = 0
        self.color, self.size, self.health = color, PLAYER_SIZE, 100
        self.holding_food = None; self.is_ai = is_ai
        self.timers = {'dodge': 0, 'hit_stun': 0, 'throw': 0, 'grab': 0, 'ai_cooldown': 0}
        self.scale = pygame.Vector2(1.0, 1.0)
        self.face_dir = -1 if is_ai else 1
        self.throw_power = 0
        # ### --- STRATEGY/SPEED V3 --- ###
        # AI now has a max_speed property, set by the agent.
        self.max_speed = 10 

    def move(self, direction):
        if any(self.timers[k] > 0 for k in ['dodge', 'hit_stun', 'grab', 'ai_cooldown']):
            return
        self.vel_x += direction * 0.6
        if direction != 0: self.face_dir = direction

    def start_grab(self, foods):
        if self.holding_food is None and all(t == 0 for t in self.timers.values()):
            self.timers['grab'] = 15
            for food in foods:
                if food.state == "landed" and self.pos.distance_to(food.pos) < self.size + 15:
                    self.holding_food = food; foods.remove(food); particles.emit(self.pos, 'grab'); return True
            return False

    def charge_throw(self):
        if self.holding_food: self.throw_power = min(100, self.throw_power + 2)

    def release_throw(self, target_pos):
        if self.holding_food and self.throw_power > 10:
            self.timers['throw'] = 15
            thrown_food = self.holding_food; self.holding_food = None
            start_pos = self.pos + pygame.Vector2(self.face_dir * 45, -self.size/2)
            thrown_food.throw(start_pos, self.throw_power, target_pos, self)
            self.throw_power = 0
            return thrown_food
        self.throw_power = 0
        return None

    def dodge(self):
        if all(t == 0 for t in self.timers.values()):
            self.timers['dodge'] = 35
            self.vel_x = 0
            return True
        return False

    def take_damage(self, amount):
        if self.timers['dodge'] > 10: return False, None
        self.health = max(0, self.health - amount)
        self.timers['hit_stun'] = 25; shaker.start(6, 10)
        self.vel_x += -self.face_dir * 7
        self.throw_power = 0; self.holding_food = None
        particles.emit(self.pos, 'hit')
        return True, FloatingText(f"-{amount}", self.pos + pygame.Vector2(0, -self.size*2), (220,20,60))

    def update(self):
        for timer in self.timers:
            if self.timers[timer] > 0: self.timers[timer] -= 1
        
        # ### --- STRATEGY/SPEED V3 --- ###
        # Clamp the horizontal velocity to the max_speed.
        self.vel_x = np.clip(self.vel_x, -self.max_speed, self.max_speed)

        if self.timers['dodge'] > 0: 
            squish = (self.timers['dodge'] / 35)
            self.scale.x, self.scale.y = 1.0 + 0.4 * (1-squish), 1.0 - 0.4 * (1-squish)
        elif self.timers['grab'] > 0: self.scale.x, self.scale.y = 0.8, 1.2
        elif self.timers['throw'] > 0: self.scale.x, self.scale.y = 1.2, 0.8
        elif self.throw_power > 0:
            wobble = math.sin(pygame.time.get_ticks() * 0.5) * 0.1
            self.scale.x, self.scale.y = 1.0 - wobble, 1.0 + wobble
        else:
            self.scale.x += (1.0 - self.scale.x) * 0.2; self.scale.y += (1.0 - self.scale.y) * 0.2

        if self.timers['dodge'] > 0:
            dodge_speed = 12 * math.sin(math.pi * self.timers['dodge'] / 35)
            self.pos.x += self.face_dir * dodge_speed
        else:
            self.pos.x += self.vel_x
            
        self.pos.x = np.clip(self.pos.x, self.size, SCREEN_WIDTH - self.size)
        self.vel_x *= 0.85
        
    def draw(self, surface, offset):
        body_color = (255, 255, 150) if self.timers['hit_stun'] > 15 else self.color
        if self.timers['dodge'] > 10 : body_color = (200, 220, 255)
        
        draw_width, draw_height = self.size * self.scale.x * 2, self.size * self.scale.y * 2
        body_rect = pygame.Rect(0, 0, draw_width, draw_height)
        body_rect.center = self.pos + pygame.Vector2(0, self.size - draw_height / 2) + offset
        if self.timers['throw'] > 7: body_rect.x += -10 * self.face_dir
        pygame.draw.ellipse(surface, body_color, body_rect); pygame.draw.ellipse(surface, (0,0,0), body_rect, 3)
        self.draw_face(surface, body_rect.center)
        if self.throw_power > 10:
            bar_w, bar_h = 50, 8; bar_x, bar_y = self.pos.x - bar_w/2, self.pos.y - self.size - 20
            pygame.draw.rect(surface, (50,50,50), (bar_x, bar_y, bar_w, bar_h), border_radius=2)
            pygame.draw.rect(surface, (255,255,0), (bar_x, bar_y, bar_w * (self.throw_power/100), bar_h), border_radius=2)
        bar_x = 20 if not self.is_ai else SCREEN_WIDTH - HEALTH_BAR_WIDTH - 20
        pygame.draw.rect(surface, (50,50,50), (bar_x, 20, HEALTH_BAR_WIDTH, HEALTH_BAR_HEIGHT), border_radius=3)
        pygame.draw.rect(surface, HEALTH_COLOR, (bar_x, 20, HEALTH_BAR_WIDTH * (self.health / 100), HEALTH_BAR_HEIGHT), border_radius=3)

    def draw_face(self, surface, center):
        eye_x = center[0] + 12 * self.face_dir; eye_y = center[1] - 8
        if self.timers['hit_stun'] > 0:
            pygame.draw.line(surface, (0,0,0), (eye_x-4, eye_y-4), (eye_x+4, eye_y+4), 3)
            pygame.draw.line(surface, (0,0,0), (eye_x-4, eye_y+4), (eye_x+4, eye_y-4), 3)
        elif self.is_ai and self.timers['ai_cooldown'] > 0:
            pygame.draw.circle(surface, (0,0,0), (eye_x, eye_y), 5)
            pygame.draw.line(surface, (0,0,0), (center[0]-8, center[1]+8), (center[0]+8, center[1]+8), 2)
        else:
            pygame.draw.circle(surface, (0,0,0), (eye_x, eye_y), 5)
            if self.throw_power > 10 or self.timers['throw'] > 0: pygame.draw.circle(surface, (0,0,0), (center[0], center[1]+5), 8)
            else: pygame.draw.arc(surface, (0,0,0), (center[0]-10, center[1], 20, 15), math.pi, 2*math.pi, 2)


# --- AI & Main Game ---
class AIAgent:
    def __init__(self, ai_player, player, difficulty="EASY"):
        self.ai_player, self.player = ai_player, player
        # ### --- STRATEGY/SPEED V3 --- ### Removed move_left/right, they are handled by get_food/reposition
        self.actions = ['get_food', 'throw', 'dodge', 'reposition'] 
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        self.alpha, self.gamma, self.epsilon = 0.1, 0.99, 0.9
        self.epsilon_decay, self.min_epsilon = 0.9995, 0.1
        # A new Q-table is needed because the action space changed.
        self.q_table_path = "food_fight_q_table_v5.npy"
        self.load_q_table()
        self.prediction_factor = 20
        self.set_difficulty(difficulty)

    def set_difficulty(self, difficulty_level):
        print(f"AI difficulty set to: {difficulty_level}")
        self.difficulty = difficulty_level
        settings = AI_DIFFICULTY_SETTINGS[difficulty_level]
        self.aim_error = settings["aim_error"]
        self.dodge_chance = settings["dodge_chance"]
        self.min_throw_power = settings["min_throw_power"]
        self.post_throw_cooldown = settings["post_throw_cooldown"]
        # ### --- STRATEGY/SPEED V3 --- ### Set the max_speed on the AI's Player object
        if self.ai_player:
            self.ai_player.max_speed = settings["max_speed"]

    def load_q_table(self):
        # Adjust for the different action space size when loading old tables
        if os.path.exists(self.q_table_path):
            try:
                loaded_q_table = np.load(self.q_table_path, allow_pickle=True).item()
                # Basic check to see if shapes match
                first_val = next(iter(loaded_q_table.values()))
                if len(first_val) == len(self.actions):
                    self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), loaded_q_table)
                    print(f"Loaded compatible Q-table: {self.q_table_path}")
                else:
                    print(f"Q-table {self.q_table_path} has incompatible action space. Starting fresh.")
            except Exception as e:
                 print(f"Could not load Q-table: {e}. Starting fresh.")
        self.epsilon = self.min_epsilon if len(self.q_table) > 10 else 0.9


    def save_q_table(self): np.save(self.q_table_path, dict(self.q_table))

    def get_discretized_state(self, all_foods):
        # No changes to state representation needed
        dx = self.player.pos.x - self.ai_player.pos.x
        dist_to_player_bucket = int(abs(dx) / 150)
        player_is_left = dx < 0
        ai_health_bucket = int(self.ai_player.health / 25)
        ai_has_food = self.ai_player.holding_food is not None
        landed_foods = [f for f in all_foods if f.state == "landed"]
        closest_food_dist = 9999
        food_is_close = False
        if landed_foods:
            closest_food = min(landed_foods, key=lambda f: self.ai_player.pos.distance_to(f.pos))
            closest_food_dist = self.ai_player.pos.distance_to(closest_food.pos)
            if closest_food_dist < 200: food_is_close = True
        imminent_threat = False
        for f in all_foods:
            if f.state == 'flying' and f.owner == self.player:
                if (f.vel.x > 0 and self.ai_player.pos.x > f.pos.x) or \
                   (f.vel.x < 0 and self.ai_player.pos.x < f.pos.x):
                    time_to_impact = abs((self.ai_player.pos.x - f.pos.x) / f.vel.x) if f.vel.x != 0 else 999
                    if 0 < time_to_impact < 40:
                        imminent_threat = True
                        break
        return (dist_to_player_bucket, player_is_left, ai_health_bucket, ai_has_food, food_is_close, imminent_threat)
    
    # ### --- STRATEGY/SPEED V3 --- ### New action logic
    def get_action(self, state, all_foods):
        # Priority 1: Cooldown
        if self.ai_player.timers['ai_cooldown'] > 0:
            return None

        # Priority 2: Dodge
        imminent_threat = state[-1]
        if imminent_threat and self.ai_player.timers['dodge'] == 0 and random.random() < self.dodge_chance:
            return 'dodge'

        # Priority 3: If no food, GET food. This is the new core logic.
        ai_has_food = state[3]
        if not ai_has_food:
            return 'get_food' # Always prioritize getting food if unarmed.
        
        # Priority 4: If we have food, use RL to decide what to do.
        # This prevents the AI from choosing 'get_food' when it's already holding something.
        if self.ai_player.holding_food:
            self.ai_player.charge_throw()

        # Let RL decide between 'throw' and 'reposition'
        if random.random() < self.epsilon:
            return random.choice(['throw', 'reposition'])
        else:
            # Get Q-values only for the valid actions when holding food
            q_values = self.q_table[state]
            action_map = {'throw': self.actions.index('throw'), 'reposition': self.actions.index('reposition')}
            
            # Decide between throw and reposition based on which has a higher Q-value
            if q_values[action_map['throw']] > q_values[action_map['reposition']]:
                return 'throw'
            else:
                return 'reposition'

    def update_q_table(self, state, action, reward, next_state):
        if action is None: return
        action_idx = self.actions.index(action)
        old_value = self.q_table[state][action_idx]
        next_max = np.max(self.q_table[next_state])
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action_idx] = new_value
        if self.epsilon > self.min_epsilon: self.epsilon *= self.epsilon_decay

class FoodFightGame:
    def __init__(self):
        self.ai_agent = AIAgent(None, None, difficulty=CURRENT_DIFFICULTY)
        self.training_thread, self.is_training = None, False
        self.training_progress, self.total_training_rounds = 0, 0
        self.floating_texts = []
        self.reset_game()
        
    def reset_game(self, new_round=True):
        self.player, self.ai = Player(100, PLAYER_COLOR), Player(SCREEN_WIDTH - 100, AI_COLOR, True)
        self.ai_agent.player, self.ai_agent.ai_player = self.player, self.ai
        self.ai_agent.set_difficulty(self.ai_agent.difficulty) # Re-apply difficulty settings to new player object
        self.all_foods = [Food() for _ in range(3)]
        self.game_over, self.winner = False, None
        if new_round: self.round = getattr(self, 'round', 0) + 1
        
    def spawn_food(self):
        if len([f for f in self.all_foods if f.state != 'splatted']) < 10: self.all_foods.append(Food())

    def update(self, player_action, ai_action):
        if self.game_over: return
        self.execute_player_action(player_action)
        reward = self.execute_ai_action(ai_action)
        if random.random() < 0.03: self.spawn_food()
        self.player.update(); self.ai.update()
        for food in self.all_foods[:]:
            result = food.update()
            if result == "miss": 
                if food.owner == self.ai: reward -= 0.2
                food.create_splat()
            if food.state == "splatted" and not food.splat_particles: self.all_foods.remove(food)
            if food.state == "flying":
                if food.owner != self.player:
                    hit_player, text_p = self.check_hit(food, self.player)
                    if hit_player: reward -= 1.0; food.create_splat(); self.floating_texts.append(text_p)
                if food.owner != self.ai:
                    hit_ai, text_ai = self.check_hit(food, self.ai)
                    if hit_ai: reward += 1.0; food.create_splat(); self.floating_texts.append(text_ai)
        
        particles.update(); shaker.update()
        for text in self.floating_texts[:]:
            if not text.update(): self.floating_texts.remove(text)
        if self.player.health <= 0: self.end_game("AI")
        elif self.ai.health <= 0: self.end_game("Player")
        return reward

    def check_hit(self, food, player):
        if player.pos.distance_to(food.pos) < player.size: return player.take_damage(15)
        return False, None
        
    def execute_player_action(self, action):
        self.player.move(action['move'])
        if action['grab']: self.player.start_grab([f for f in self.all_foods if f.state == "landed"])
        if action['charge_throw']: self.player.charge_throw()
        if action['release_throw']:
            thrown = self.player.release_throw(self.ai.pos)
            if thrown: self.all_foods.append(thrown)
        if action['dodge']: self.player.dodge()

    def execute_ai_action(self, action):
        if action is None:
            return -0.01

        reward = -0.01
        landed = [f for f in self.all_foods if f.state == "landed"]
        if self.player.pos.x < self.ai.pos.x: self.ai.face_dir = -1
        else: self.ai.face_dir = 1

        if action == 'get_food':
            if landed:
                closest_food = min(landed, key=lambda f: self.ai.pos.distance_to(f.pos))
                move_direction = np.sign(closest_food.pos.x - self.ai.pos.x)
                self.ai.move(move_direction)
                # Attempt to grab while moving
                if self.ai.start_grab(landed): reward += 0.3
            else: # No food on the ground, move to a defensive spot
                target_x = SCREEN_WIDTH * 0.75
                move_direction = np.sign(target_x - self.ai.pos.x)
                self.ai.move(move_direction)
        elif action == 'reposition':
            # Move away from the player to create distance
            move_direction = -np.sign(self.player.pos.x - self.ai.pos.x)
            self.ai.move(move_direction)
        elif action == 'throw':
            if self.ai.throw_power > self.ai_agent.min_throw_power:
                predicted_player_pos = self.player.pos + pygame.Vector2(self.player.vel_x * self.ai_agent.prediction_factor, 0)
                error = self.ai_agent.aim_error
                predicted_player_pos.x += random.uniform(-error, error)
                
                thrown = self.ai.release_throw(predicted_player_pos)
                if thrown:
                    self.ai.timers['ai_cooldown'] = self.ai_agent.post_throw_cooldown
                    self.all_foods.append(thrown); reward += 0.5
        elif action == 'dodge':
            if self.ai.dodge(): reward += 0.1

        return reward
    
    # ... [end_game, draw methods, training methods, and main loop are the same] ...
    def end_game(self, winner): self.game_over, self.winner = True, winner
    
    def draw(self, surface):
        offset = shaker.offset
        surface.fill(BACKGROUND_COLOR)
        pygame.draw.rect(surface, (160, 120, 80), (0+offset.x, FLOOR_Y+offset.y, SCREEN_WIDTH, SCREEN_HEIGHT - FLOOR_Y))
        for food in self.all_foods: food.draw(surface, offset)
        self.player.draw(surface, offset); self.ai.draw(surface, offset)
        particles.draw(surface, offset)
        for text in self.floating_texts: text.draw(surface, offset)
        
        diff_text = small_font.render(f"AI: {self.ai_agent.difficulty} (Keys 1, 2, 3)", True, TEXT_COLOR)
        surface.blit(diff_text, (SCREEN_WIDTH - diff_text.get_width() - 10, SCREEN_HEIGHT - 30))
        
        round_text = font.render(f"Round: {self.round}", True, TEXT_COLOR); surface.blit(round_text, round_text.get_rect(centerx=SCREEN_WIDTH/2, y=10))
        if self.is_training: self.draw_training_ui(surface)
        if self.game_over: self.draw_game_over_ui(surface)

    def draw_training_ui(self, surface):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 180)); surface.blit(overlay, (0, 0))
        title = title_font.render("AI IS TRAINING...", True, (255, 100, 100)); surface.blit(title, title.get_rect(centerx=SCREEN_WIDTH/2, y=SCREEN_HEIGHT/2 - 100))
        if self.total_training_rounds > 0:
            progress = self.training_progress / self.total_training_rounds
            bar_w, bar_h = 400, 30; bar_x, bar_y = SCREEN_WIDTH/2 - bar_w/2, SCREEN_HEIGHT/2
            pygame.draw.rect(surface, (50,50,50), (bar_x, bar_y, bar_w, bar_h), border_radius=5)
            pygame.draw.rect(surface, AI_COLOR, (bar_x, bar_y, bar_w * progress, bar_h), border_radius=5)
            prog_text = small_font.render(f"{self.training_progress}/{self.total_training_rounds}", True, TEXT_COLOR); surface.blit(prog_text, prog_text.get_rect(center=(SCREEN_WIDTH/2, bar_y + 15)))

    def draw_game_over_ui(self, surface):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 180)); surface.blit(overlay, (0, 0))
        win_text = title_font.render(f"{self.winner} Wins!", True, (255, 215, 0)); surface.blit(win_text, win_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2 - 50)))
        restart_text = font.render("Press 'R' for the next round!", True, (255, 255, 255)); surface.blit(restart_text, restart_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2 + 20)))

    def train_in_background(self, rounds=2000):
        self.is_training, self.total_training_rounds = True, rounds
        self.ai_agent.set_difficulty("HARD")
        for i in range(rounds):
            self.training_progress = i + 1; self.reset_game(new_round=False)
            self.ai_agent.set_difficulty("HARD") # Ensure HARD settings are used for training round
            for step in range(500):
                state = self.ai_agent.get_discretized_state(self.all_foods)
                ai_action = self.ai_agent.get_action(state, self.all_foods)
                # Simplified random player for training
                player_action = {'move': 0, 'grab': random.random()<0.05, 'charge_throw': self.player.holding_food, 'release_throw': self.player.throw_power > 50 and random.random()<0.1, 'dodge': random.random()<0.02}
                if self.player.holding_food: player_action['move'] = 0
                else: player_action['move'] = random.choice([-1,1,0])
                reward = self.update(player_action, ai_action)
                next_state = self.ai_agent.get_discretized_state(self.all_foods)
                self.ai_agent.update_q_table(state, ai_action, reward, next_state)
                if self.game_over: break
        self.ai_agent.save_q_table(); print("Training complete.")
        self.ai_agent.set_difficulty(CURRENT_DIFFICULTY)
        self.is_training = False; self.reset_game()

    def start_training_thread(self):
        if self.is_training: return
        self.training_thread = threading.Thread(target=self.train_in_background, daemon=True)
        self.training_thread.start()


def main():
    game = FoodFightGame()
    while True:
        keys = pygame.key.get_pressed()
        player_action = {'move': 0, 'grab': False, 'charge_throw': False, 'release_throw': False, 'dodge': False}
        if keys[pygame.K_LEFT]: player_action['move'] = -1
        if keys[pygame.K_RIGHT]: player_action['move'] = 1
        if keys[pygame.K_SPACE]: player_action['charge_throw'] = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and (game.game_over or not game.is_training): game.reset_game()
                if event.key == pygame.K_t: game.start_training_thread()
                if event.key == pygame.K_d: player_action['dodge'] = True
                if event.key == pygame.K_SPACE: player_action['grab'] = True
                
                if event.key == pygame.K_1:
                    game.ai_agent.set_difficulty("EASY")
                if event.key == pygame.K_2:
                    game.ai_agent.set_difficulty("MEDIUM")
                if event.key == pygame.K_3:
                    game.ai_agent.set_difficulty("HARD")

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: player_action['release_throw'] = True

        if not game.is_training and not game.game_over:
            ai_state = game.ai_agent.get_discretized_state(game.all_foods)
            ai_action = game.ai_agent.get_action(ai_state, game.all_foods)
            game.update(player_action, ai_action)

        game.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()