# Learning RL Agents

## English

In this directory, we demonstrate learning RL agents from MuJoCo Playground environments using [Brax](https://github.com/google/brax) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl). We provide two entrypoints from the command line: `python train_jax_ppo.py` and `python train_rsl_rl.py`.

## Polski (Polish)

W tym katalogu pokazujemy jak trenować agentów RL (Reinforcement Learning - Uczenie ze Wzmocnieniem) w środowiskach MuJoCo Playground, używając frameworków [Brax](https://github.com/google/brax) oraz [RSL-RL](https://github.com/leggedrobotics/rsl_rl). Udostępniamy dwa główne skrypty: `python train_jax_ppo.py` oraz `python train_rsl_rl.py`.

**Cel**: Te skrypty pozwalają wytrenować polityki sterowania robotami (np. Unitree G1, Panda, Leap Hand) bez pisania algorytmów od podstaw. Wystarczy wybrać środowisko i uruchomić trening!

For more detailed tutorials on using MuJoCo Playground for RL, see:

1. Intro. to the Playground with DM Control Suite [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/dm_control_suite.ipynb)
2. Locomotion Environments [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/locomotion.ipynb)
3. Manipulation Environments [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/manipulation.ipynb)
4. Training CartPole from Vision [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_1.ipynb)
5. Robotic Manipulation from Vision [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_2.ipynb)

## Training with brax PPO

### English

To train with brax PPO, you can use the `train_jax_ppo.py` script. This script uses the brax PPO algorithm to train an agent on a given environment.

```bash
python train_jax_ppo.py --env_name=CartpoleBalance
```

To train a vision-based policy using pixel observations:
```bash
python train_jax_ppo.py --env_name=CartpoleBalance --vision
```

Use `python train_jax_ppo.py --help` to see possible options and usage. Logs and checkpoints are saved in `logs` directory.

### Polski

Aby trenować z algorytmem PPO z biblioteki Brax, użyj skryptu `train_jax_ppo.py`. Ten skrypt trenuje agenta na wybranym środowisku.

**Podstawowe użycie**:
```bash
python train_jax_ppo.py --env_name=CartpoleBalance
```

**Dlaczego PPO?** PPO (Proximal Policy Optimization) to jeden z najpopularniejszych algorytmów uczenia ze wzmocnieniem. Jest stabilny, efektywny i dobrze działa na wielu problemach robotyki.

**Trening z wejściem wizyjnym** (kamera zamiast stanów):
```bash
python train_jax_ppo.py --env_name=CartpoleBalance --vision
```

**Przydatne komendy**:
```bash
# Zobacz wszystkie opcje
python train_jax_ppo.py --help

# Trening robota G1 (humanoid)
python train_jax_ppo.py --env_name=G1JoystickFlatTerrain --num_timesteps=2000000

# Trening z domain randomization (WAŻNE dla sim-to-real!)
python train_jax_ppo.py --env_name=G1JoystickFlatTerrain --domain_randomization

# Kontynuacja treningu z checkpointu
python train_jax_ppo.py --env_name=G1JoystickFlatTerrain \
    --load_checkpoint_path logs/G1JoystickFlatTerrain-20250210-120000/checkpoints
```

**Gdzie znajdują się wyniki?**
- Logi i checkpointy są zapisywane w katalogu `logs/`
- Każdy eksperyment ma własny podkatalog z timestamp

## Training with RSL-RL

### English

To train with RSL-RL, you can use the `train_rsl_rl.py` script. This script uses the RSL-RL algorithm to train an agent on a given environment.

```bash
python train_rsl_rl.py --env_name=LeapCubeReorient
```

To render the behaviour from the resulting policy:
```bash
python learning/train_rsl_rl.py --env_name LeapCubeReorient --play_only --load_run_name <run_name>
```

where `run_name` is the name of the run you want to load (will be printed in the console when the training run is started).

Logs and checkpoints are saved in `logs` directory.

### Polski

Aby trenować z RSL-RL, użyj skryptu `train_rsl_rl.py`. Ten skrypt używa algorytmu RSL-RL do trenowania agenta.

**Podstawowe użycie**:
```bash
python train_rsl_rl.py --env_name=LeapCubeReorient
```

**Czym różni się od PPO?** RSL-RL to framework stworzony przez Robotic Systems Lab w ETH Zurich, specjalnie zoptymalizowany pod kątem robotów nożnych (quadrupeds, bipeds). Często daje lepsze wyniki dla lokomocji niż standardowe PPO.

**Odtwarzanie wytrenowanej polityki**:
```bash
python learning/train_rsl_rl.py --env_name LeapCubeReorient --play_only --load_run_name <run_name>
```

gdzie `<run_name>` to nazwa eksperymentu (wyświetlana w konsoli na początku treningu).

**Przykład dla robota G1**:
```bash
# Trening
python train_rsl_rl.py --env_name=G1JoystickFlatTerrain

# Po treningu - odtwórz wideo
python train_rsl_rl.py --env_name=G1JoystickFlatTerrain \
    --play_only --load_run_name G1JoystickFlatTerrain_2025-02-10_12-00-00
```

Logi i checkpointy są zapisywane w katalogu `logs/`.

## Porównanie: PPO vs RSL-RL

| Aspekt | Brax PPO | RSL-RL |
|--------|----------|---------|
| **Szybkość** | Bardzo szybki (GPU) | Szybki (GPU/CPU) |
| **Środowiska** | Wszystkie | Głównie lokomocja |
| **Łatwość użycia** | Prosta | Średnia |
| **Gdy użyć?** | Uniwersalne zadania | Roboty nożne/lokomocja |

**Rekomendacja dla studentów**: Zacznij od Brax PPO - jest prostszy i bardziej uniwersalny. RSL-RL wypróbuj gdy pracujesz specyficznie nad lokomocją.
