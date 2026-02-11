# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a PPO agent using JAX on the specified environment.

PL (Polski): Trenowanie agenta PPO przy użyciu JAX dla wybranego środowiska.

OPIS DLA STUDENTÓW:
==================
Ten skrypt jest głównym narzędziem do trenowania polityk sterowania robotem w symulacji.
Używa algorytmu PPO (Proximal Policy Optimization) - jednego z najpopularniejszych
algorytmów głębokiego uczenia ze wzmocnieniem.

Co robi ten skrypt:
1. Wczytuje środowisko symulacyjne (np. robot G1, ramię Panda, itp.)
2. Konfiguruje parametry treningu PPO
3. Uruchamia równoległe symulacje na GPU dla szybkiego uczenia
4. Zapisuje checkpointy modelu w regularnych odstępach
5. Generuje wideo z zachowaniem wytrenowanego agenta

Podstawowe użycie:
  python train_jax_ppo.py --env_name G1JoystickFlatTerrain --num_timesteps 2000000

Przydatne parametry:
  --env_name: nazwa środowiska (np. G1JoystickFlatTerrain dla robota G1)
  --num_timesteps: ile kroków symulacji (więcej = dłuższy trening)
  --num_envs: ile równoległych symulacji (więcej = szybszy trening, ale więcej RAM)
  --load_checkpoint_path: ścieżka do checkpointu do kontynuacji treningu
  --domain_randomization: włącza losowość parametrów fizyki (ważne dla sim-to-real)
"""

import datetime
import functools
import json
import os
import time
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params
import tensorboardX

try:
  import wandb
except ImportError:
  wandb = None


# Konfiguracja środowiska dla obliczeń GPU (PL)
# ===============================================
# XLA_FLAGS: optymalizacje kompilatora XLA dla szybszych obliczeń na GPU
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"  # Używaj Triton dla mnożenia macierzy
os.environ["XLA_FLAGS"] = xla_flags
# Nie alokuj całej pamięci GPU z góry - pozwala na równoległe uruchamianie wielu procesów
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Użyj EGL do renderowania bez wyświetlacza (ważne na serwerach bez GUI)
os.environ["MUJOCO_GL"] = "egl"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Wyciszenie ostrzeżeń (PL)
# ==========================
# Podczas treningu JAX może generować wiele ostrzeżeń technicznych, które
# nie wpływają na działanie programu. Wyciszamy je dla czytelności.

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


# Parametry uruchamiania skryptu (PL: Flagi linii poleceń)
# ==========================================================
# Poniższe flagi pozwalają konfigurować trening bez edycji kodu.
# Przykład użycia: python train_jax_ppo.py --env_name G1JoystickFlatTerrain --num_envs 2048

_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "LeapCubeReorient",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
    # PL: Nazwa środowiska do treningu (np. G1JoystickFlatTerrain dla humanoidalnego robota)
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
# PL: Backend symulacji - 'jax' (MJX) lub 'warp' (MuJoCo Warp)
_PLAYGROUND_CONFIG_OVERRIDES = flags.DEFINE_string(
    "playground_config_overrides",
    None,
    "Overrides for the playground env config.",
    # PL: Nadpisania konfiguracji środowiska w formacie JSON
)
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
# PL: Czy używać wizji (kamera) jako wejścia do polityki
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
    # PL: Ścieżka do checkpointu do kontynuacji treningu
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
# PL: Dodatkowy sufiks dla nazwy eksperymentu (ułatwia organizację)
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
    # PL: Tylko odtwarzaj model bez treningu (do testowania wytrenowanej polityki)
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode)",
    # PL: Czy używać Weights & Biases do logowania metryk
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
    # PL: Czy używać TensorBoard do logowania (lokalna alternatywa dla W&B)
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
    # PL: Losowość domenowa - zmienia parametry fizyki w każdym epizodzie
    # (WAŻNE dla transferu sim-to-real - robot lepiej się adaptuje do rzeczywistości!)
)
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
# PL: Ziarno losowości - ustaw to samo dla powtarzalnych wyników
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 1_000_000, "Number of timesteps"
    # PL: Całkowita liczba kroków symulacji (więcej = dłuższy trening, lepsza polityka)
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 1, "Number of videos to record after training."
    # PL: Ile wideo nagrać po treningu (do oceny jakości polityki)
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
# PL: Ile razy oceniać politykę w trakcie treningu
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
# PL: Skalowanie nagród - ważne dla stabilności uczenia
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
# PL: Długość epizodu w krokach (1000 kroków * 0.02s = 20 sekund dla ctrl_dt=0.02)
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
    # PL: Normalizacja obserwacji - poprawia stabilność uczenia
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
# PL: Ile razy powtórzyć tę samą akcję (zmniejsza częstotliwość decyzji)
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
# PL: Długość sekwencji do obliczania gradientów
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 8, "Number of minibatches"
    # PL: Na ile części podzielić batch do aktualizacji (wpływa na stabilność)
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "Number of updates per batch"
    # PL: Ile razy zaktualizować sieć na jednym batchu danych
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
# PL: Współczynnik dyskontowania (gamma) - jak ważne są przyszłe nagrody (0.97 = dość ważne)
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
# PL: Szybkość uczenia - za duża = niestabilne, za mała = wolne uczenie
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
# PL: Koszt entropii - zachęca do eksploracji (wyższy = więcej eksploracji)
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
# PL: Liczba równoległych symulacji - więcej = szybszy trening (ale wymaga więcej GPU RAM)
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
    # PL: Liczba środowisk do ewaluacji polityki
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
# PL: Rozmiar batcha do uczenia
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
# PL: Maksymalna norma gradientu - przycinanie zapobiega eksplozji gradientów
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.2, "Clipping epsilon for PPO"
    # PL: Epsilon dla PPO clipping - zapobiega zbyt dużym aktualizacjom polityki
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "Policy hidden layer sizes",
    # PL: Rozmiary ukrytych warstw sieci polityki [64,64,64] = 3 warstwy po 64 neurony
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "Value hidden layer sizes",
    # PL: Rozmiary warstw sieci wartości (critic) - ocenia jak dobre są stany
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Policy obs key"
    # PL: Klucz obserwacji dla polityki (może być 'state' lub 'pixels' dla wizji)
)
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")
# PL: Klucz obserwacji dla funkcji wartości
_RSCOPE_ENVS = flags.DEFINE_integer(
    "rscope_envs",
    None,
    "Number of parallel environment rollouts to save for the rscope viewer",
    # PL: Liczba środowisk do wizualizacji w rscope (narzędzie do interaktywnej wizualizacji)
)
_DETERMINISTIC_RSCOPE = flags.DEFINE_boolean(
    "deterministic_rscope",
    True,
    "Run deterministic rollouts for the rscope viewer",
    # PL: Czy używać deterministycznej polityki dla rscope (bez losowości)
)
_RUN_EVALS = flags.DEFINE_boolean(
    "run_evals",
    True,
    "Run evaluation rollouts between policy updates.",
    # PL: Czy uruchamiać ewaluacje między aktualizacjami (spowalnia, ale daje feedback)
)
_LOG_TRAINING_METRICS = flags.DEFINE_boolean(
    "log_training_metrics",
    False,
    "Whether to log training metrics and callback to progress_fn. Significantly"
    " slows down training if too frequent.",
    # PL: Czy logować metryki treningowe (może spowolnić trening jeśli zbyt często)
)
_TRAINING_METRICS_STEPS = flags.DEFINE_integer(
    "training_metrics_steps",
    1_000_000,
    "Number of steps between logging training metrics. Increase if training"
    " experiences slowdown.",
    # PL: Co ile kroków logować metryki treningowe
)


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  """Selects a PPO configuration for the environment category.

  PL: Dobiera konfigurację PPO odpowiednią dla typu środowiska.
  """
  if env_name in mujoco_playground.manipulation._envs:
    if _VISION.value:
      return manipulation_params.brax_vision_ppo_config(env_name, _IMPL.value)
    return manipulation_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.locomotion._envs:
    return locomotion_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.dm_control_suite._envs:
    if _VISION.value:
      return dm_control_suite_params.brax_vision_ppo_config(
          env_name, _IMPL.value
      )
    return dm_control_suite_params.brax_ppo_config(env_name, _IMPL.value)

  raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def rscope_fn(full_states, obs, rew, done):
  """
  All arrays are of shape (unroll_length, rscope_envs, ...)
  full_states: dict with keys 'qpos', 'qvel', 'time', 'metrics'
  obs: nd.array or dict obs based on env configuration
  rew: nd.array rewards
  done: nd.array done flags

  PL: Funkcja oblicza skumulowane nagrody epizodów z rolloutów, aby można było
  raportować metryki w dalszych krokach.
  """
  # Calculate cumulative rewards per episode, stopping at first done flag
  done_mask = jp.cumsum(done, axis=0)
  valid_rewards = rew * (done_mask == 0)
  episode_rewards = jp.sum(valid_rewards, axis=0)
  print(
      "Collected rscope rollouts with reward"
      f" {episode_rewards.mean():.3f} +- {episode_rewards.std():.3f}"
  )


def main(argv):
  """Run training and evaluation for the specified environment.

  PL: Główna ścieżka programu prowadząca od konfiguracji po zapis wideo.
  """

  del argv

  # Step 1: Load the environment configuration and choose the impl backend.
  # PL: Krok 1: Wczytaj konfigurację środowiska i wybierz backend w kluczu impl
  # (np. MJX/JAX lub MuJoCo Warp).
  env_cfg = registry.get_default_config(_ENV_NAME.value)
  env_cfg["impl"] = _IMPL.value

  # Step 2: Fetch the default PPO parameters for the selected environment.
  # PL: Krok 2: Pobierz domyślne parametry PPO dla danego środowiska.
  ppo_params = get_rl_config(_ENV_NAME.value)

  # Step 3: Override PPO parameters with CLI flags for quick experiments.
  # PL: Krok 3: Nadpisz parametry PPO flagami CLI, aby łatwo robić
  # eksperymenty.
  if _NUM_TIMESTEPS.present:
    ppo_params.num_timesteps = _NUM_TIMESTEPS.value
  if _PLAY_ONLY.present:
    ppo_params.num_timesteps = 0
  if _NUM_EVALS.present:
    ppo_params.num_evals = _NUM_EVALS.value
  if _REWARD_SCALING.present:
    ppo_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present:
    ppo_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    ppo_params.action_repeat = _ACTION_REPEAT.value
  if _UNROLL_LENGTH.present:
    ppo_params.unroll_length = _UNROLL_LENGTH.value
  if _NUM_MINIBATCHES.present:
    ppo_params.num_minibatches = _NUM_MINIBATCHES.value
  if _NUM_UPDATES_PER_BATCH.present:
    ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
  if _DISCOUNTING.present:
    ppo_params.discounting = _DISCOUNTING.value
  if _LEARNING_RATE.present:
    ppo_params.learning_rate = _LEARNING_RATE.value
  if _ENTROPY_COST.present:
    ppo_params.entropy_cost = _ENTROPY_COST.value
  if _NUM_ENVS.present:
    ppo_params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present:
    ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present:
    ppo_params.batch_size = _BATCH_SIZE.value
  if _MAX_GRAD_NORM.present:
    ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
  if _CLIPPING_EPSILON.present:
    ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value
  if _POLICY_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.policy_hidden_layer_sizes = list(
        map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
    )
  if _VALUE_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.value_hidden_layer_sizes = list(
        map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
    )
  if _POLICY_OBS_KEY.present:
    ppo_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
  if _VALUE_OBS_KEY.present:
    ppo_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value
  if _VISION.value:
    env_cfg.vision = True
    env_cfg.vision_config.render_batch_size = ppo_params.num_envs
  # Step 4: Apply additional environment overrides from JSON.
  # PL: Krok 4: Dodatkowe nadpisania konfiguracji środowiska z JSON.
  env_cfg_overrides = {}
  if _PLAYGROUND_CONFIG_OVERRIDES.value is not None:
    env_cfg_overrides = json.loads(_PLAYGROUND_CONFIG_OVERRIDES.value)
  # Step 5: Load the environment from the registry with the prepared config.
  # PL: Krok 5: Załaduj środowisko z rejestru i gotową konfiguracją.
  env = registry.load(
      _ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides
  )
  if _RUN_EVALS.present:
    ppo_params.run_evals = _RUN_EVALS.value
  if _LOG_TRAINING_METRICS.present:
    ppo_params.log_training_metrics = _LOG_TRAINING_METRICS.value
  if _TRAINING_METRICS_STEPS.present:
    ppo_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value

  print(f"Environment Config:\n{env_cfg}")
  if env_cfg_overrides:
    print(f"Environment Config Overrides:\n{env_cfg_overrides}\n")
  print(f"PPO Training Parameters:\n{ppo_params}")

  # Step 6: Generate a unique experiment name for logs and checkpoints.
  # PL: Krok 6: Wygeneruj unikalną nazwę eksperymentu do logów i
  # checkpointów.
  now = datetime.datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{_ENV_NAME.value}-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  print(f"Experiment name: {exp_name}")

  # Step 7: Prepare a logging directory for runs and checkpoints.
  # PL: Krok 7: Przygotuj katalog na logi i pliki kontrolne.
  logdir = epath.Path("logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Step 8: Optionally initialize Weights & Biases for metric tracking.
  # PL: Krok 8: Opcjonalnie inicjalizuj Weights & Biases dla śledzenia
  # metryk.
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    if wandb is None:
      raise ImportError(
          "wandb is required for --use_wandb. "
          "Install via: pip install wandb"
      )
    wandb.init(project="mjxrl", name=exp_name)
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({"env_name": _ENV_NAME.value})

  # Step 9: Optionally initialize TensorBoard for training charts.
  # PL: Krok 9: Opcjonalnie inicjalizuj TensorBoard dla wykresów.
  if _USE_TB.value and not _PLAY_ONLY.value:
    writer = tensorboardX.SummaryWriter(logdir)

  # Step 10: Restore training state from a checkpoint when provided.
  # PL: Krok 10: Jeśli podano checkpoint, przywróć stan treningu.
  if _LOAD_CHECKPOINT_PATH.value is not None:
    # Convert to absolute path
    ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path.is_dir():
      latest_ckpts = list(ckpt_path.glob("*"))
      latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
      latest_ckpts.sort(key=lambda x: int(x.name))
      latest_ckpt = latest_ckpts[-1]
      restore_checkpoint_path = latest_ckpt
      print(f"Restoring from: {restore_checkpoint_path}")
    else:
      restore_checkpoint_path = ckpt_path
      print(f"Restoring from checkpoint: {restore_checkpoint_path}")
  else:
    print("No checkpoint path provided, not restoring from checkpoint")
    restore_checkpoint_path = None

  # Step 11: Create a directory for new checkpoints.
  # PL: Krok 11: Utwórz katalog na nowe checkpointy.
  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")

  # Step 12: Save the environment configuration for reproducibility.
  # PL: Krok 12: Zapisz konfigurację środowiska, by była odtwarzalna.
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  training_params = dict(ppo_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]

  # Step 13: Choose the network factory (vision-based or classic).
  # PL: Krok 13: Wybierz fabrykę sieci (oparta na wizji lub klasyczna).
  network_fn = (
      ppo_networks_vision.make_ppo_networks_vision
      if _VISION.value
      else ppo_networks.make_ppo_networks
  )
  if hasattr(ppo_params, "network_factory"):
    network_factory = functools.partial(
        network_fn, **ppo_params.network_factory
    )
  else:
    network_factory = network_fn

  # Step 14: Add domain randomization to vary physics parameters for sim-to-real.
  # PL: Krok 14: Dodaj losowość domenową, która zmienia parametry fizyki i
  # ułatwia transfer sim-to-real.
  if _DOMAIN_RANDOMIZATION.value:
    training_params["randomization_fn"] = registry.get_domain_randomizer(
        _ENV_NAME.value
    )

  # Step 15: Wrap the environment for PPO training (optionally vision-based).
  # PL: Krok 15: Dopasuj środowisko do treningu PPO (opcjonalnie z wizją).
  if _VISION.value:
    env = wrapper.wrap_for_brax_training(
        env,
        vision=True,
        num_vision_envs=env_cfg.vision_config.render_batch_size,
        episode_length=ppo_params.episode_length,
        action_repeat=ppo_params.action_repeat,
        randomization_fn=training_params.get("randomization_fn"),
    )

  num_eval_envs = (
      ppo_params.num_envs
      if _VISION.value
      else ppo_params.get("num_eval_envs", 128)
  )

  if "num_eval_envs" in training_params:
    del training_params["num_eval_envs"]

  # Step 16: Prepare the PPO training function with fixed parameters.
  # PL: Krok 16: Przygotuj funkcję treningu PPO z ustalonymi parametrami.
  train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory=network_factory,
      seed=_SEED.value,
      restore_checkpoint_path=restore_checkpoint_path,
      save_checkpoint_path=ckpt_path,
      wrap_env_fn=None if _VISION.value else wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
  )

  times = [time.monotonic()]

  # Step 17: Progress callback collects metrics and prints a summary.
  # PL: Krok 17: Funkcja postępu zbiera metryki i wypisuje podsumowanie.
  def progress(num_steps, metrics):
    times.append(time.monotonic())

    # Log to Weights & Biases
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.log(metrics, step=num_steps)

    # Log to TensorBoard
    if _USE_TB.value and not _PLAY_ONLY.value:
      for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
      writer.flush()
    if _RUN_EVALS.value:
      print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")
    if _LOG_TRAINING_METRICS.value:
      if "episode/sum_reward" in metrics:
        print(
            f"{num_steps}: mean episode"
            f" reward={metrics['episode/sum_reward']:.3f}"
        )

  # Step 18: Prepare the evaluation environment separate from training.
  # PL: Krok 18: Przygotuj środowisko ewaluacyjne (osobne od treningowego).
  eval_env = None
  if not _VISION.value:
    eval_env = registry.load(
        _ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides
    )
  num_envs = 1
  if _VISION.value:
    num_envs = env_cfg.vision_config.render_batch_size

  policy_params_fn = lambda *args: None
  if _RSCOPE_ENVS.value:
    # Interactive visualisation of policy checkpoints
    from rscope import brax as rscope_utils

    if not _VISION.value:
      rscope_env = registry.load(
          _ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides
      )
      rscope_env = wrapper.wrap_for_brax_training(
          rscope_env,
          episode_length=ppo_params.episode_length,
          action_repeat=ppo_params.action_repeat,
          randomization_fn=training_params.get("randomization_fn"),
      )
    else:
      rscope_env = env

    rscope_handle = rscope_utils.BraxRolloutSaver(
        rscope_env,
        ppo_params,
        _VISION.value,
        _RSCOPE_ENVS.value,
        _DETERMINISTIC_RSCOPE.value,
        jax.random.PRNGKey(_SEED.value),
        rscope_fn,
    )

    def policy_params_fn(current_step, make_policy, params):  # pylint: disable=unused-argument
      rscope_handle.set_make_policy(make_policy)
      rscope_handle.dump_rollout(params)

  # Step 19: Run training or restore a model from a checkpoint.
  # PL: Krok 19: Uruchom trening lub odtwórz model z checkpointu.
  make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
      environment=env,
      progress_fn=progress,
      policy_params_fn=policy_params_fn,
      eval_env=eval_env,
  )

  print("Done training.")
  if len(times) > 1:
    print(f"Time to JIT compile: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

  print("Starting inference...")

  # Step 20: Prepare the inference function for policy evaluation.
  # PL: Krok 20: Przygotuj funkcję inferencji do oceny polityki.
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)

  # Step 21: Run evaluation rollouts and store trajectories.
  # PL: Krok 21: Uruchom rollouty ewaluacyjne i zapisz trajektorie.
  def do_rollout(rng, state):
    empty_data = state.data.__class__(
        **{k: None for k in state.data.__annotations__}
    )  # pytype: disable=attribute-error
    empty_traj = state.__class__(**{k: None for k in state.__annotations__})  # pytype: disable=attribute-error
    empty_traj = empty_traj.replace(data=empty_data)

    def step(carry, _):
      state, rng = carry
      rng, act_key = jax.random.split(rng)
      act = jit_inference_fn(state.obs, act_key)[0]
      state = eval_env.step(state, act)
      traj_data = empty_traj.tree_replace({
          "data.qpos": state.data.qpos,
          "data.qvel": state.data.qvel,
          "data.time": state.data.time,
          "data.ctrl": state.data.ctrl,
          "data.mocap_pos": state.data.mocap_pos,
          "data.mocap_quat": state.data.mocap_quat,
          "data.xfrc_applied": state.data.xfrc_applied,
      })
      if _VISION.value:
        traj_data = jax.tree_util.tree_map(lambda x: x[0], traj_data)
      return (state, rng), traj_data

    _, traj = jax.lax.scan(
        step, (state, rng), None, length=_EPISODE_LENGTH.value
    )
    return traj

  rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
  reset_states = jax.jit(jax.vmap(eval_env.reset))(rng)
  if _VISION.value:
    reset_states = jax.tree_util.tree_map(lambda x: x[0], reset_states)
  traj_stacked = jax.jit(jax.vmap(do_rollout))(rng, reset_states)
  trajectories = [None] * _NUM_VIDEOS.value
  for i in range(_NUM_VIDEOS.value):
    t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
    trajectories[i] = [
        jax.tree.map(lambda x, j=j: x[j], t)
        for j in range(_EPISODE_LENGTH.value)
    ]

  # Step 22: Render trajectories to video for analysis.
  # PL: Krok 22: Wyrenderuj trajektorie do wideo, aby je przeanalizować.
  render_every = 2
  fps = 1.0 / eval_env.dt / render_every
  print(f"FPS for rendering: {fps}")
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
  for i, rollout in enumerate(trajectories):
    traj = rollout[::render_every]
    frames = eval_env.render(
        traj, height=480, width=640, scene_option=scene_option
    )
    media.write_video(f"rollout{i}.mp4", frames, fps=fps)
    print(f"Rollout video saved as 'rollout{i}.mp4'.")


def run():
  """Entry point for uv/pip script."""
  app.run(main)


if __name__ == "__main__":
  run()
