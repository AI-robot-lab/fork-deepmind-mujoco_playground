# Quick Start - Pierwsze Kroki dla Studentów

**Czas wykonania: 15-30 minut**

Ten przewodnik przeprowadzi Cię przez podstawową konfigurację i pierwsze eksperymenty z MuJoCo Playground.

---

## Krok 1: Sprawdź wymagania (2 minuty)

Przed rozpoczęciem upewnij się, że masz:

```bash
# Sprawdź Pythona (wymagane: 3.10+)
python --version
# Powinno wyświetlić: Python 3.10.x lub nowszy

# Sprawdź CUDA (opcjonalne, ale zalecane)
nvidia-smi
# Jeśli widzisz informacje o GPU, masz CUDA!
```

**Nie masz GPU?** Nie martw się - możesz trenować na CPU, ale będzie wolniej.

---

## Krok 2: Instalacja (5-10 minut)

```bash
# 1. Sklonuj repozytorium
cd ~
git clone https://github.com/AI-robot-lab/fork-deepmind-mujoco_playground.git
cd fork-deepmind-mujoco_playground

# 2. Zainstaluj uv (szybki menedżer pakietów)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Uruchom ponownie terminal lub:
source ~/.bashrc

# 3. Stwórz środowisko wirtualne
uv venv --python 3.12
source .venv/bin/activate

# 4. Zainstaluj JAX z GPU (jeśli masz CUDA 12)
uv pip install -U "jax[cuda12]" --index-url https://pypi.org/simple

# Jeśli masz CUDA 11:
# uv pip install -U "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Jeśli NIE masz GPU:
# uv pip install jax

# 5. Sprawdź czy GPU działa
python -c "import jax; print(f'Backend: {jax.default_backend()}')"
# Powinno wyświetlić: Backend: gpu (lub cpu jeśli nie masz GPU)

# 6. Zainstaluj playground
uv --no-config sync --all-extras

# 7. Pobierz modele robotów (włącznie z G1)
python -c "from mujoco_playground import locomotion; locomotion.load('G1JoystickFlatTerrain')"
```

**Problem?** Zobacz sekcję "Możliwe problemy" w `PRZEWODNIK_G1_PL.md`

---

## Krok 3: Pierwszy test - prosty przykład (3 minuty)

Stwórz plik `test_podstawowy.py`:

```python
"""Pierwszy test - czy wszystko działa?"""
import jax
from mujoco_playground import locomotion

print("🤖 Ładowanie robota G1...")
env = locomotion.load('G1JoystickFlatTerrain')

print(f"✓ Załadowano!")
print(f"  Wymiar obserwacji: {env.observation_size}")
print(f"  Wymiar akcji: {env.action_size}")

# Reset środowiska
rng = jax.random.PRNGKey(0)
state = jax.jit(env.reset)(rng)

print(f"\n✓ Robot zainicjalizowany na pozycji: {state.data.qpos[:3]}")

# Wykonaj 50 kroków symulacji
print("\n🏃 Wykonuję 50 kroków symulacji...")
for i in range(50):
    action = jax.numpy.zeros(env.action_size)  # Zero akcji = próba stania
    state = env.step(state, action)
    
    if i % 10 == 0:
        print(f"  Krok {i}: nagroda = {state.reward:.3f}")

print("\n✅ Test zakończony pomyślnie!")
```

Uruchom:
```bash
python test_podstawowy.py
```

**Oczekiwany rezultat**: Skrypt powinien wyświetlić informacje o robocie i wykonać symulację bez błędów.

---

## Krok 4: Wideo z symulacji (5 minut)

Stwórz plik `test_wideo.py`:

```python
"""Stwórz wideo z symulacji robota G1"""
import jax
import jax.numpy as jp
import mediapy as media
from mujoco_playground import locomotion

print("🤖 Ładowanie robota G1...")
env = locomotion.load('G1JoystickFlatTerrain')

# Inicjalizacja
rng = jax.random.PRNGKey(42)
state = jax.jit(env.reset)(rng)

# Symulacja 200 kroków z małymi losowymi ruchami
print("🎬 Nagrywanie symulacji (200 kroków)...")
states = [state]

for i in range(200):
    if i % 50 == 0:
        print(f"  Postęp: {i}/200 kroków")
    
    rng, key = jax.random.split(rng)
    # Małe losowe akcje
    action = jax.random.uniform(key, (env.action_size,), minval=-0.05, maxval=0.05)
    state = env.step(state, action)
    states.append(state)

# Renderowanie
print("\n📹 Renderowanie wideo...")
frames = env.render(states, height=480, width=640)

# Zapis
output = 'moje_pierwsze_wideo_g1.mp4'
media.write_video(output, frames, fps=50)

print(f"\n✅ Wideo zapisane jako '{output}'")
print(f"   Możesz je teraz obejrzeć!")
```

Uruchom:
```bash
python test_wideo.py
# Następnie otwórz plik moje_pierwsze_wideo_g1.mp4 w odtwarzaczu
```

---

## Krok 5: Pierwszy trening (10-15 minut)

Teraz wytrenuj prostą politykę na prostym środowisku:

```bash
# Krótki trening na CartPole (2-3 minuty)
python learning/train_jax_ppo.py \
    --env_name CartpoleBalance \
    --num_timesteps 100000 \
    --num_envs 512 \
    --num_evals 2

# Sprawdź logi
ls -lh logs/
```

**Co się dzieje?**
- `--num_timesteps 100000`: Trenujesz przez 100k kroków (krótki test)
- `--num_envs 512`: Używasz 512 równoległych symulacji
- `--num_evals 2`: Ewaluacja polityki 2 razy w trakcie treningu

Po treningu zobaczysz katalog w `logs/` z wynikami i checkpointami.

**Chcesz zobaczyć wideo?** Skrypt automatycznie tworzy `rollout0.mp4` po treningu.

---

## Krok 6: Trening robota G1 (opcjonalnie, jeśli masz czas)

```bash
# UWAGA: Ten trening zajmie 30-60 minut (lub więcej bez GPU)
python learning/train_jax_ppo.py \
    --env_name G1JoystickFlatTerrain \
    --num_timesteps 500000 \
    --num_envs 2048 \
    --num_evals 3
```

Możesz przerwać w każdej chwili (Ctrl+C) - postęp jest zapisywany.

---

## Co dalej?

Gratulacje! 🎉 Masz działające środowisko. Teraz możesz:

### 1. Przejrzeć przykłady
```bash
# Uruchom wszystkie przykłady interaktywnie
python przyklady_g1.py

# Lub konkretny przykład (np. przykład 2)
python przyklady_g1.py 2
```

### 2. Przeczytać pełny przewodnik
```bash
# Otwórz w edytorze lub przeglądarce markdown
cat PRZEWODNIK_G1_PL.md
```

### 3. Eksperymentować z parametrami

Spróbuj zmienić wagi nagród w treningu:

```bash
python learning/train_jax_ppo.py \
    --env_name G1JoystickFlatTerrain \
    --num_timesteps 200000 \
    --playground_config_overrides '{"reward_config": {"scales": {"tracking_lin_vel": 2.0}}}'
```

### 4. Dołączyć do społeczności

- GitHub Issues: https://github.com/google-deepmind/mujoco_playground/issues
- GitHub Discussions: https://github.com/google-deepmind/mujoco_playground/discussions
- MuJoCo Forum: https://github.com/google-deepmind/mujoco/discussions

---

## Często zadawane pytania (FAQ)

### Q: Import error: "No module named 'mujoco_playground'"

**A**: Upewnij się, że:
1. Aktywowałeś środowisko wirtualne: `source .venv/bin/activate`
2. Zainstalowałeś playground: `uv --no-config sync --all-extras`

### Q: "Backend: cpu" zamiast "gpu"

**A**: Spróbuj:
```bash
unset LD_LIBRARY_PATH
python -c "import jax; print(jax.default_backend())"
```

Jeśli nadal CPU, sprawdź czy masz zainstalowane CUDA i odpowiednią wersję JAX.

### Q: Trening jest bardzo wolny

**A**: Możliwe przyczyny:
1. Brak GPU - trening na CPU jest 100-1000x wolniejszy
2. Za dużo środowisk - zmniejsz `--num_envs` (np. do 256)
3. Za często ewaluacja - zmniejsz `--num_evals`

### Q: Gdzie są checkpointy?

**A**: W katalogu `logs/<nazwa_eksperymentu>/checkpoints/`

```bash
ls -lh logs/*/checkpoints/
```

### Q: Jak kontynuować przerwany trening?

**A**:
```bash
python learning/train_jax_ppo.py \
    --env_name G1JoystickFlatTerrain \
    --load_checkpoint_path logs/G1JoystickFlatTerrain-20250210-120000/checkpoints
```

### Q: Robot się przewraca/nie chodzi prawidłowo

**A**: To normalne na początku treningu! Robot uczy się od zera. Po 1-2M kroków powinien nauczyć się chodzić. Jeśli nie:
1. Sprawdź wagi nagród w konfiguracji
2. Zwiększ liczbę kroków treningu
3. Zobacz przykłady w `przyklady_g1.py`

---

## Checklist sukcesu

Zaznacz co już zrobiłeś:

- [ ] Python 3.10+ zainstalowany
- [ ] Repozytorium sklonowane
- [ ] Środowisko wirtualne utworzone i aktywowane
- [ ] JAX zainstalowany (z GPU jeśli możliwe)
- [ ] MuJoCo Playground zainstalowany
- [ ] Test podstawowy przeszedł pomyślnie
- [ ] Wideo z symulacji wygenerowane
- [ ] Pierwszy trening (CartPole) zakończony
- [ ] Obejrzałem `przyklady_g1.py`
- [ ] Przeczytałem `PRZEWODNIK_G1_PL.md`

**Wszystko zaznaczone?** Świetnie! Jesteś gotowy do pracy z robotem G1! 🚀

---

## Pomoc

Jeśli masz problemy:

1. **Sprawdź przewodnik**: `PRZEWODNIK_G1_PL.md` - sekcja "Często zadawane pytania"
2. **Uruchom przykłady**: `python przyklady_g1.py` - mogą pomóc zidentyfikować problem
3. **Sprawdź logi**: Komunikaty błędów często wskazują przyczynę
4. **Poproś kolegów**: Inni studenci mogą mieć podobne problemy
5. **Zapytaj prowadzącego**: Wykładowca lub asystent pomoże rozwiązać problem

**Pamiętaj**: Każdy ekspert kiedyś był początkującym! 💪

---

*Dokument przygotowany dla studentów Politechniki Rzeszowskiej*
*Ostatnia aktualizacja: 2025-02-10*
