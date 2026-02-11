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
"""Base classes for G1.

PL (Polski): Klasy bazowe dla robota Unitree G1.

OPIS DLA STUDENTÓW:
===================
Ten moduł definiuje podstawową klasę G1Env, która jest bazą dla wszystkich
środowisk z robotem Unitree G1. Klasa ta:

1. Ładuje model URDF robota G1 z plików XML
2. Konfiguruje parametry symulacji (krok czasowy, zakres stawów)
3. Udostępnia metody do odczytu sensorów (IMU, akcelerometr, żyroskop)
4. Zarządza komunikacją między MuJoCo a MJX (wersja GPU)

WAŻNE KONCEPTY:
- MjModel: Model robota w MuJoCo (CPU)
- MjxModel: Model robota w MJX (GPU) - używany do treningu
- Sensory: Symulowane czujniki robota (jak na prawdziwym G1)
"""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import g1_constants as consts


def get_assets() -> Dict[str, bytes]:
  """Load all asset files needed for G1 robot model.
  
  PL: Ładuje wszystkie pliki zasobów potrzebne do modelu robota G1.
  
  Funkcja ta zbiera:
  - Pliki XML z definicją robota (geometria, masa, stawy)
  - Tekstury i meshe (wygląd robota)
  - Pliki z Menagerie (oficjalne modele Unitree)
  
  Returns:
    Słownik {nazwa_pliku: zawartość_binarna}
  """
  assets = {}
  # Załaduj XML z lokalnego katalogu środowiska
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "assets")
  # Załaduj oficjalny model z Menagerie (baza modeli robotów)
  path = mjx_env.MENAGERIE_PATH / "unitree_g1"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  return assets


class G1Env(mjx_env.MjxEnv):
  """Base class for G1 environments.
  
  PL: Klasa bazowa dla wszystkich środowisk z robotem G1.
  
  Ta klasa jest dziedziczona przez konkretne środowiska takie jak:
  - G1JoystickFlatTerrain (chodzenie z kontrolą joysticka)
  - G1InplaceGaitTracking (śledzenie wzorców chodu)
  
  Zapewnia:
  - Ładowanie i konfigurację modelu robota
  - Metody dostępu do sensorów (IMU, akcelerometr, żyroskop)
  - Zarządzanie stanem symulacji
  """

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    """Initialize G1 environment.
    
    PL: Inicjalizuje środowisko G1.
    
    Args:
      xml_path: Ścieżka do pliku XML z modelem robota
      config: Konfiguracja środowiska (nagrody, parametry fizyki, etc.)
      config_overrides: Nadpisania domyślnej konfiguracji
    """
    super().__init__(config, config_overrides)

    # Krok 1: Załaduj wszystkie pliki zasobów (XML, tekstury, meshe)
    self._model_assets = get_assets()
    
    # Krok 2: Stwórz model MuJoCo z pliku XML
    # MjModel zawiera całą fizyczną definicję robota: masy, geometrie,
    # stawy, aktuatory, sensory, etc.
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=self._model_assets
    )
    # Ustaw krok czasowy symulacji (np. 0.002s = 500Hz)
    self._mj_model.opt.timestep = self.sim_dt

    # Krok 3: Opcjonalnie ogranicz zakres stawów dla bezpieczeństwa
    # Przydatne przy transferze sim-to-real - unikniesz uszkodzenia robota
    if self._config.restricted_joint_range:
      self._mj_model.jnt_range[1:] = consts.RESTRICTED_JOINT_RANGE
      self._mj_model.actuator_ctrlrange[:] = consts.RESTRICTED_JOINT_RANGE

    # Krok 4: Ustaw wysoką rozdzielczość renderowania (do wideo)
    self._mj_model.vis.global_.offwidth = 3840   # 4K szerokość
    self._mj_model.vis.global_.offheight = 2160  # 4K wysokość

    # Krok 5: Skonwertuj model do MJX (wersja GPU)
    # MJX pozwala na równoległą symulację tysięcy środowisk na GPU
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path

  # Sensor readings.
  # PL: Odczyty z sensorów robota
  # ==============================
  # Poniższe metody zwracają dane z symulowanych czujników.
  # W rzeczywistym robocie G1 te same dane pochodziłyby z IMU,
  # akcelerometru i żyroskopu.

  def get_gravity(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the gravity vector in the world frame.
    
    PL: Zwraca wektor grawitacji w układzie świata.
    
    Wektor grawitacji wskazuje "gdzie jest dół" - kluczowe dla orientacji robota.
    Symuluje czujnik IMU (Inertial Measurement Unit).
    
    Args:
      data: Stan symulacji MJX
      frame: Nazwa ramki (np. 'pelvis' dla miednicy robota)
    
    Returns:
      Wektor [gx, gy, gz] w układzie świata
    """
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GRAVITY_SENSOR}_{frame}"
    )

  def get_global_linvel(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the linear velocity of the robot in the world frame.
    
    PL: Zwraca prędkość liniową robota w układzie świata.
    
    Jak szybko robot się porusza w przestrzeni [vx, vy, vz].
    """
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GLOBAL_LINVEL_SENSOR}_{frame}"
    )

  def get_global_angvel(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the angular velocity of the robot in the world frame.
    
    PL: Zwraca prędkość kątową robota w układzie świata.
    
    Jak szybko robot się obraca [wx, wy, wz].
    """
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GLOBAL_ANGVEL_SENSOR}_{frame}"
    )

  def get_local_linvel(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the linear velocity of the robot in the local frame.
    
    PL: Zwraca prędkość liniową robota w układzie lokalnym.
    
    Prędkość względem orientacji robota (przód/tył, lewo/prawo, góra/dół).
    """
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.LOCAL_LINVEL_SENSOR}_{frame}"
    )

  def get_accelerometer(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the accelerometer readings in the local frame.
    
    PL: Zwraca odczyty z akcelerometru w układzie lokalnym.
    
    Mierzy przyspieszenie liniowe - jak mocno robot przyspiesza/hamuje.
    Symuluje rzeczywisty akcelerometr w IMU robota G1.
    """
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.ACCELEROMETER_SENSOR}_{frame}"
    )

  def get_gyro(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the gyroscope readings in the local frame.
    
    PL: Zwraca odczyty z żyroskopu w układzie lokalnym.
    
    Mierzy prędkość kątową - jak szybko robot się obraca wokół swoich osi.
    Symuluje rzeczywisty żyroskop w IMU robota G1.
    """
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GYRO_SENSOR}_{frame}"
    )

  # Accessors.

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
