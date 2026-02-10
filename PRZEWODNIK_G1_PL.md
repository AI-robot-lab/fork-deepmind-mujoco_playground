# Przewodnik: Robot Humanoidalny Unitree G1 EDU-U6 w MuJoCo Playground

## Spis treści
1. [Wstęp](#wstęp)
2. [Czym jest MuJoCo Playground](#czym-jest-mujoco-playground)
3. [Robot Unitree G1 - Specyfikacja](#robot-unitree-g1---specyfikacja)
4. [Instalacja i konfiguracja](#instalacja-i-konfiguracja)
5. [Pierwsze kroki - Symulacja G1](#pierwsze-kroki---symulacja-g1)
6. [Trening polityki sterowania](#trening-polityki-sterowania)
7. [Praktyczne przykłady](#praktyczne-przykłady)
8. [Analiza i debugowanie](#analiza-i-debugowanie)
9. [Transfer sim-to-real](#transfer-sim-to-real)
10. [Często zadawane pytania](#często-zadawane-pytania)

---

## Wstęp

Ten przewodnik został przygotowany specjalnie dla studentów Politechniki Rzeszowskiej pracujących nad projektem z robotem humanoidalnym **Unitree G1 EDU-U6**. Celem jest umożliwienie szybkiego opanowania narzędzi do symulacji i uczenia robotów przed przejściem do pracy z fizycznym sprzętem.

### Dlaczego symulacja?

- **Bezpieczeństwo**: Możesz eksperymentować bez ryzyka uszkodzenia drogiego sprzętu
- **Szybkość**: Symulacje na GPU są tysiące razy szybsze niż czas rzeczywisty
- **Powtarzalność**: Możesz łatwo powtarzać eksperymenty z identycznymi warunkami
- **Koszt**: Nie potrzebujesz fizycznego robota do nauki i eksperymentów

---

## Czym jest MuJoCo Playground

MuJoCo Playground to platforma do:
- **Symulacji robotów** z wykorzystaniem silnika fizyki MuJoCo
- **Uczenia ze wzmocnieniem (Reinforcement Learning)** z akceleracją GPU
- **Trenowania polityk sterowania** w równoległych środowiskach
- **Transferu sim-to-real** - przenoszenia polityk z symulacji do rzeczywistości

### Kluczowe komponenty:

1. **MuJoCo** - silnik fizyki symulujący dynamikę robotów
2. **MJX (MuJoCo JAX)** - wersja MuJoCo zoptymalizowana dla GPU
3. **JAX** - framework do obliczeń numerycznych z automatycznym różniczkowaniem
4. **PPO** - algorytm uczenia ze wzmocnieniem (Proximal Policy Optimization)

---

## Robot Unitree G1 - Specyfikacja

### Charakterystyka robota G1:

- **Typ**: Humanoid (robot dwunożny, dwuręczny)
- **Wysokość**: ~130 cm
- **Waga**: ~35 kg
- **Stopnie swobody**: 23 DOF (Degrees of Freedom)
  - Nogi: 12 DOF (po 6 na każdą nogę)
  - Tułów: 3 DOF
  - Ręce: 8 DOF (po 4 na każdą rękę)

### Dostępne środowiska dla G1:

```python
from mujoco_playground import registry

# Lista wszystkich środowisk G1
g1_envs = [env for env in registry.ALL_ENVS if 'G1' in env]
print(g1_envs)
# ['G1JoystickFlatTerrain', 'G1InplaceGaitTracking', ...]
```

Najważniejsze środowiska:
- **G1JoystickFlatTerrain**: Chodzenie po płaskim terenie z kontrolą joysticka
- **G1InplaceGaitTracking**: Śledzenie wzorców chodu w miejscu
- **G1FlatTerrain**: Podstawowe chodzenie do przodu

---

## Instalacja i konfiguracja

### Wymagania systemowe:

- **System**: Linux (Ubuntu 20.04+ zalecany) lub macOS
- **GPU**: NVIDIA z CUDA 12.x (zalecane dla szybkiego treningu)
- **RAM**: Minimum 16 GB (32 GB zalecane dla dużych symulacji)
- **Python**: 3.10 lub nowszy

### Kroki instalacji:

```bash
# 1. Sklonuj repozytorium
git clone git@github.com:google-deepmind/mujoco_playground.git
cd mujoco_playground

# 2. Zainstaluj uv (szybka alternatywa dla pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Stwórz środowisko wirtualne
uv venv --python 3.12
source .venv/bin/activate

# 4. Zainstaluj JAX z obsługą CUDA
uv pip install -U "jax[cuda12]" --index-url https://pypi.org/simple

# 5. Sprawdź czy GPU jest wykrywane
python -c "import jax; print(f'Backend: {jax.default_backend()}')"
# Powinno wyświetlić: Backend: gpu

# 6. Zainstaluj playground ze wszystkimi dodatkami
uv --no-config sync --all-extras

# 7. Zweryfikuj instalację
uv --no-config run python -c "import mujoco_playground; print('Sukces!')"

# 8. Pobierz modele robotów (włącznie z G1)
uv --no-config run python -c "from mujoco_playground import locomotion; locomotion.load('G1JoystickFlatTerrain')"
```

### Możliwe problemy i rozwiązania:

**Problem**: `jax.default_backend()` zwraca 'cpu' zamiast 'gpu'
```bash
# Rozwiązanie: usuń konfliktujące zmienne środowiskowe
unset LD_LIBRARY_PATH
python -c "import jax; print(jax.default_backend())"
```

**Problem**: Brak CUDA 12
```bash
# Sprawdź wersję CUDA
nvidia-smi
# Jeśli masz CUDA 11, zainstaluj JAX dla CUDA 11
uv pip install -U "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

## Pierwsze kroki - Symulacja G1

### Przykład 1: Podstawowa symulacja

Stwórz plik `test_g1_basic.py`:

```python
"""
Prosty skrypt do załadowania i wizualizacji robota G1.
Ten przykład pokazuje jak:
1. Załadować środowisko G1
2. Uruchomić symulację
3. Zastosować losowe akcje
"""

import jax
from mujoco_playground import locomotion

# Załaduj środowisko G1 na płaskim terenie z kontrolą joysticka
env = locomotion.load('G1JoystickFlatTerrain')

print(f"Wymiary przestrzeni obserwacji: {env.observation_size}")
print(f"Wymiary przestrzeni akcji: {env.action_size}")

# Zainicjalizuj stan początkowy
rng = jax.random.PRNGKey(0)
state = jax.jit(env.reset)(rng)

print(f"Początkowa pozycja robota: {state.data.qpos[:3]}")  # x, y, z

# Wykonaj 100 kroków z losowymi akcjami
for i in range(100):
    rng, action_key = jax.random.split(rng)
    # Losowe akcje w zakresie [-1, 1]
    action = jax.random.uniform(action_key, (env.action_size,), minval=-0.1, maxval=0.1)
    state = env.step(state, action)
    
    if i % 20 == 0:
        print(f"Krok {i}: nagroda = {state.reward:.3f}")

print("Symulacja zakończona!")
```

Uruchom:
```bash
python test_g1_basic.py
```

### Przykład 2: Wizualizacja trajektorii

```python
"""
Wizualizacja trajektorii robota G1.
Zapisuje wideo z symulacji.
"""

import jax
import jax.numpy as jp
import mediapy as media
import mujoco
from mujoco_playground import locomotion

env = locomotion.load('G1JoystickFlatTerrain')
rng = jax.random.PRNGKey(42)

# Reset środowiska
state = jax.jit(env.reset)(rng)

# Lista stanów do wizualizacji
states = [state]

# Symulacja 200 kroków z małymi losowymi akcjami
for _ in range(200):
    rng, key = jax.random.split(rng)
    action = jax.random.uniform(key, (env.action_size,), minval=-0.05, maxval=0.05)
    state = env.step(state, action)
    states.append(state)

# Renderowanie wideo
print("Renderowanie wideo...")
frames = env.render(states, height=480, width=640)
media.write_video('g1_simulation.mp4', frames, fps=50)
print("Wideo zapisane jako g1_simulation.mp4")
```

---

## Trening polityki sterowania

### Podstawowy trening

Najprostszy sposób na rozpoczęcie treningu:

```bash
# Trening na 2 miliony kroków (ok. 1-2 godziny na dobrej GPU)
python learning/train_jax_ppo.py \
    --env_name G1JoystickFlatTerrain \
    --num_timesteps 2000000 \
    --num_envs 2048
```

### Parametry treningu - co oznaczają?

```bash
# Przykład z wyjaśnieniem każdego parametru
# --env_name: Nazwa środowiska
# --num_timesteps: Całkowita liczba kroków treningu
# --num_envs: Liczba równoległych symulacji
# --num_evals: Ewaluacja co N aktualizacji
# --learning_rate: Szybkość uczenia
# --entropy_cost: Koszt entropii (eksploracja)
# --batch_size: Rozmiar batcha
# --unroll_length: Długość sekwencji
# --use_tb: Używaj TensorBoard
# --domain_randomization: WAŻNE dla sim-to-real!

python learning/train_jax_ppo.py \
    --env_name G1JoystickFlatTerrain \
    --num_timesteps 5000000 \
    --num_envs 4096 \
    --num_evals 10 \
    --learning_rate 3e-4 \
    --entropy_cost 1e-2 \
    --batch_size 512 \
    --unroll_length 20 \
    --use_tb \
    --domain_randomization
```

### Jak dobrać parametry?

**Zbyt wolny trening?**
- Zwiększ `--num_envs` (wymaga więcej GPU RAM)
- Zwiększ `--batch_size`
- Zmniejsz `--num_evals`

**Niestabilny trening?**
- Zmniejsz `--learning_rate` (spróbuj 1e-4)
- Zmniejsz `--batch_size`
- Zwiększ `--max_grad_norm`

**Robot uczy się za wolno?**
- Zwiększ `--num_timesteps`
- Dostosuj funkcję nagrody w konfiguracji środowiska
- Sprawdź czy domainrandomization nie jest zbyt agresywna

### Kontynuacja treningu

```bash
# Wznów trening z checkpointu
python learning/train_jax_ppo.py \
    --env_name G1JoystickFlatTerrain \
    --load_checkpoint_path logs/G1JoystickFlatTerrain-20250210-120000/checkpoints
```

---

## Praktyczne przykłady

### Przykład 3: Trening z custom rewards

Możesz modyfikować funkcję nagrody, aby dostosować zachowanie robota:

```python
"""
Przykład modyfikacji konfiguracji nagród dla G1.
"""

import json
from mujoco_playground import registry, locomotion

# Załaduj domyślną konfigurację
config = registry.get_default_config('G1JoystickFlatTerrain')

# Modyfikuj wagi nagród
config.reward_config.scales.tracking_lin_vel = 2.0  # Zwiększ za podążanie
config.reward_config.scales.feet_air_time = 3.0     # Nagradzaj dłuższy krok
config.reward_config.scales.orientation = -5.0       # Mocniej karz za przechyły

# Zapisz do JSON
overrides = {
    'reward_config': {
        'scales': {
            'tracking_lin_vel': 2.0,
            'feet_air_time': 3.0,
            'orientation': -5.0,
        }
    }
}

# Użyj w treningu:
# python learning/train_jax_ppo.py \
#   --env_name G1JoystickFlatTerrain \
#   --playground_config_overrides '{"reward_config": {"scales": {"tracking_lin_vel": 2.0}}}'
```

### Przykład 4: Analiza wytrenowanej polityki

```python
"""
Załaduj wytrenowany model i przeanalizuj jego zachowanie.
"""

import jax
import jax.numpy as jp
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import locomotion
import pickle

# Załaduj środowisko
env = locomotion.load('G1JoystickFlatTerrain')

# Załaduj checkpoint
checkpoint_path = 'logs/G1JoystickFlatTerrain-20250210-120000/checkpoints/1000000'
with open(f'{checkpoint_path}/params', 'rb') as f:
    params = pickle.load(f)

# Stwórz sieć polityki
network = ppo_networks.make_ppo_networks(
    env.observation_size,
    env.action_size,
    preprocess_observations_fn=lambda x: x,
)

# Funkcja inferencji
inference_fn = ppo_networks.make_inference_fn(network)(params, deterministic=True)

# Testuj politykę
rng = jax.random.PRNGKey(0)
state = env.reset(rng)

rewards = []
for i in range(1000):
    action, _ = inference_fn(state.obs, rng)
    state = env.step(state, action)
    rewards.append(state.reward)

print(f"Średnia nagroda: {jp.mean(jp.array(rewards)):.3f}")
print(f"Całkowita nagroda: {jp.sum(jp.array(rewards)):.3f}")
```

---

## Analiza i debugowanie

### Monitorowanie treningu z TensorBoard

```bash
# Uruchom trening z TensorBoard
python learning/train_jax_ppo.py \
    --env_name G1JoystickFlatTerrain \
    --use_tb

# W osobnym terminalu:
tensorboard --logdir logs/
# Otwórz http://localhost:6006 w przeglądarce
```

Kluczowe metryki do obserwacji:
- **eval/episode_reward**: Nagroda podczas ewaluacji (cel: powinna rosnąć)
- **losses/policy_loss**: Strata polityki
- **losses/value_loss**: Strata funkcji wartości
- **losses/total_loss**: Całkowita strata

### Debugowanie problemów

**Robot się przewraca:**
1. Sprawdź skalę nagród dla `orientation` i `base_height`
2. Zwiększ nagrodę za `feet_air_time` (zachęca do chodzenia)
3. Zmniejsz `action_scale` w konfiguracji (mniejsze, płynniejsze ruchy)

**Robot nie idzie do przodu:**
1. Zwiększ `tracking_lin_vel` w nagrodach
2. Sprawdź czy komenda prędkości jest różna od zera
3. Zmniejsz `stand_still` penalty

**Trening nie konwerguje:**
1. Zmniejsz `learning_rate`
2. Zwiększ `num_envs` dla lepszej statystyki
3. Sprawdź czy `normalize_observations=True`

---

## Transfer sim-to-real

Transfer sim-to-real to proces przenoszenia polityki wytrenowanej w symulacji do rzeczywistego robota.

### Kluczowe techniki:

#### 1. Domain Randomization

**ZAWSZE używaj podczas treningu dla rzeczywistego robota!**

```bash
python learning/train_jax_ppo.py \
    --env_name G1JoystickFlatTerrain \
    --domain_randomization
```

Co jest randomizowane:
- Masa segmentów robota (±20%)
- Współczynniki tarcia (±50%)
- Opóźnienia aktuatorów
- Siły zakłóceń (wiatr, pchnięcia)

#### 2. Dodaj szum do obserwacji

Już skonfigurowane w środowisku G1:

```python
# W konfiguracji G1
noise_config = {
    'level': 1.0,  # 0.0 = brak szumu, 1.0 = pełny szum
    'scales': {
        'joint_pos': 0.03,    # Szum w odczytach pozycji stawów
        'joint_vel': 1.5,      # Szum w prędkościach
        'gravity': 0.05,       # Szum w odczytach orientacji
        'linvel': 0.1,         # Szum w prędkości liniowej
        'gyro': 0.2,           # Szum w żyroskopie
    }
}
```

#### 3. Ograniczenia fizyczne

```python
# Ogranicz zakres stawów do bezpiecznych wartości
config.restricted_joint_range = True
# Ogranicz maksymalną prędkość
config.action_scale = 0.3  # Zmniejsz dla bezpieczeństwa
```

### Procedura transferu:

1. **Trening w symulacji** (2-5M kroków z domain randomization)
2. **Walidacja w symulacji** (sprawdź odporność na zakłócenia)
3. **Test w środowisku kontrolowanym** (robot na podwieszeniu/asekuracji)
4. **Stopniowe zwiększanie swobody** (najpierw małe ruchy, potem pełny chód)
5. **Fine-tuning** (opcjonalnie dotrening na rzeczywistym robocie)

### Checklist przed testem na robocie:

- [ ] Model wytrenowany z `--domain_randomization`
- [ ] Działanie sprawdzone w symulacji z różnymi zaburzeniami
- [ ] Ograniczenia zakresu stawów włączone
- [ ] Action scale ustawiony na bezpieczną wartość (≤0.5)
- [ ] System awaryjnego zatrzymania przygotowany
- [ ] Przestrzeń testowa zabezpieczona (materace, asekuracja)

---

## Często zadawane pytania

### Q: Jak długo trwa trening?

**A**: Zależy od środowiska i sprzętu:
- Proste środowisko (CartPole): 1-5 minut
- G1 na płaskim terenie: 1-3 godziny
- Złożone manipulacje: 5-10 godzin

*Czasy dla GPU NVIDIA A100 z 4096 równoległymi środowiskami*

### Q: Ile pamięci GPU potrzebuję?

**A**: Orientacyjne wymagania:
- 8 GB: 1024 środowiska
- 16 GB: 2048-4096 środowiska
- 24 GB: 4096-8192 środowiska
- 40+ GB: >8192 środowiska

### Q: Czy mogę trenować bez GPU?

**A**: Tak, ale będzie BARDZO wolno (100-1000x wolniej). JAX może działać na CPU:

```bash
# Zainstaluj JAX bez CUDA
pip install jax

# Zmniejsz liczbę środowisk
python learning/train_jax_ppo.py \
    --env_name G1JoystickFlatTerrain \
    --num_envs 64 \  # Zamiast 4096
    --num_timesteps 100000  # Krótszy trening
```

### Q: Jak wybrać najlepszy checkpoint?

**A**: Nie zawsze ostatni checkpoint jest najlepszy! Sprawdź:

```bash
# Odtwórz różne checkpointy i porównaj
python learning/train_jax_ppo.py \
    --env_name G1JoystickFlatTerrain \
    --play_only \
    --load_checkpoint_path logs/.../checkpoints/1000000 \
    --num_videos 5
```

Wybierz checkpoint z:
- Najwyższą średnią nagrodą podczas ewaluacji
- Najbardziej stabilnym zachowaniem
- Najlepszą odpornością na zakłócenia

### Q: Jak dostosować środowisko do własnych potrzeb?

**A**: Możesz modyfikować konfigurację:

```python
# Zobacz dostępne opcje
from mujoco_playground import registry
config = registry.get_default_config('G1JoystickFlatTerrain')
print(config)

# Modyfikuj i zapisz
config.ctrl_dt = 0.01  # Częstotliwość sterowania
config.episode_length = 2000  # Długość epizodu
# ... i użyj w treningu z --playground_config_overrides
```

### Q: Gdzie znaleźć więcej przykładów?

**A**: 
- Notebooki Jupyter w `learning/notebooks/`
- Przykładowe skrypty w `mujoco_playground/experimental/`
- Dokumentacja online: https://playground.mujoco.org/
- GitHub Issues: https://github.com/google-deepmind/mujoco_playground/issues

---

## Dodatkowe zasoby

### Polecane materiały do nauki:

1. **Uczenie ze wzmocnieniem**:
   - Spinning Up in Deep RL (OpenAI): https://spinningup.openai.com/
   - Sutton & Barto: "Reinforcement Learning: An Introduction"

2. **MuJoCo i symulacja**:
   - Dokumentacja MuJoCo: https://mujoco.readthedocs.io/
   - MJX Tutorial: https://mujoco.readthedocs.io/en/stable/mjx.html

3. **JAX**:
   - JAX Quickstart: https://jax.readthedocs.io/en/latest/quickstart.html
   - JAX Tutorial (Polski): [YouTube - JAX basics]

### Społeczność i wsparcie:

- **Discord**: [MuJoCo Community Discord]
- **GitHub Discussions**: https://github.com/google-deepmind/mujoco_playground/discussions
- **Forum**: https://github.com/google-deepmind/mujoco/discussions

---

## Podsumowanie

Ten przewodnik powinien zapewnić solidne podstawy do pracy z robotem Unitree G1 w MuJoCo Playground. Pamiętaj:

1. **Zacznij od prostych eksperymentów** - najpierw poznaj środowisko
2. **Eksperymentuj z parametrami** - ucz się jak wpływają na zachowanie
3. **Zapisuj wszystko** - dokumentuj swoje eksperymenty
4. **Testuj stopniowo** - od symulacji do rzeczywistości małymi krokami
5. **Bezpieczeństwo przede wszystkim** - szczególnie przy pracy z fizycznym robotem

**Powodzenia w pracy z robotem G1!** 🤖

---

*Dokument przygotowany dla studentów Politechniki Rzeszowskiej*
*Ostatnia aktualizacja: 2025-02-10*
