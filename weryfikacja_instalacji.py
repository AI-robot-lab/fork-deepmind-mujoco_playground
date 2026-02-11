#!/usr/bin/env python3
"""
Skrypt weryfikacyjny dla studentów - sprawdza czy środowisko jest poprawnie skonfigurowane.

Uruchom: python weryfikacja_instalacji.py
"""

import sys
import subprocess

def print_header(text):
    """Wypisz nagłówek sekcji."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_python_version():
    """Sprawdź wersję Pythona."""
    print("\n🔍 Sprawdzanie wersji Pythona...")
    version = sys.version_info
    print(f"   Znaleziono: Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 10:
        print("   ✅ Python 3.10+ - OK!")
        return True
    else:
        print("   ❌ BŁĄD: Wymagany Python 3.10 lub nowszy")
        return False

def check_module(module_name, display_name=None):
    """Sprawdź czy moduł jest zainstalowany."""
    if display_name is None:
        display_name = module_name
    
    print(f"\n🔍 Sprawdzanie {display_name}...")
    try:
        __import__(module_name)
        print(f"   ✅ {display_name} zainstalowany!")
        return True
    except ImportError:
        print(f"   ❌ BŁĄD: {display_name} nie jest zainstalowany")
        return False

def check_jax_backend():
    """Sprawdź backend JAX (GPU/CPU)."""
    print("\n🔍 Sprawdzanie backendu JAX...")
    try:
        import jax
        backend = jax.default_backend()
        print(f"   Backend: {backend}")
        
        if backend == 'gpu':
            print("   ✅ JAX używa GPU - świetnie!")
            return True
        elif backend == 'cpu':
            print("   ⚠️  JAX używa CPU - trening będzie wolniejszy")
            print("   💡 Jeśli masz GPU, sprawdź instalację CUDA")
            return True
        else:
            print(f"   ⚠️  Nieznany backend: {backend}")
            return True
    except Exception as e:
        print(f"   ❌ BŁĄD: {e}")
        return False

def check_mujoco_playground():
    """Sprawdź instalację MuJoCo Playground."""
    print("\n🔍 Sprawdzanie MuJoCo Playground...")
    try:
        import mujoco_playground
        print("   ✅ MuJoCo Playground zainstalowany!")
        
        # Sprawdź czy można załadować środowiska
        print("   🔍 Sprawdzanie dostępnych środowisk...")
        from mujoco_playground import registry
        all_envs = registry.ALL_ENVS
        print(f"   ✅ Znaleziono {len(all_envs)} środowisk")
        
        # Sprawdź czy G1 jest dostępny
        g1_envs = [env for env in all_envs if 'G1' in env]
        if g1_envs:
            print(f"   ✅ Znaleziono {len(g1_envs)} środowisk G1: {g1_envs}")
            return True
        else:
            print("   ⚠️  Nie znaleziono środowisk G1")
            return True
    except Exception as e:
        print(f"   ❌ BŁĄD: {e}")
        return False

def test_basic_simulation():
    """Wykonaj prosty test symulacji."""
    print("\n🔍 Test podstawowej symulacji...")
    try:
        import jax
        from mujoco_playground import locomotion
        
        print("   Ładowanie środowiska G1JoystickFlatTerrain...")
        env = locomotion.load('G1JoystickFlatTerrain')
        
        print("   Inicjalizacja stanu...")
        rng = jax.random.PRNGKey(0)
        state = jax.jit(env.reset)(rng)
        
        print("   Wykonywanie 10 kroków symulacji...")
        for i in range(10):
            action = jax.numpy.zeros(env.action_size)
            state = env.step(state, action)
        
        print("   ✅ Test symulacji zakończony sukcesem!")
        print(f"   📊 Wymiar obserwacji: {env.observation_size}")
        print(f"   📊 Wymiar akcji: {env.action_size}")
        return True
    except Exception as e:
        print(f"   ❌ BŁĄD podczas testu symulacji: {e}")
        return False

def check_optional_tools():
    """Sprawdź opcjonalne narzędzia."""
    print("\n🔍 Sprawdzanie opcjonalnych narzędzi...")
    
    optional = [
        ('mediapy', 'MediaPy (do zapisywania wideo)'),
        ('tensorboardX', 'TensorBoard (do logowania)'),
        ('wandb', 'Weights & Biases (do logowania)'),
    ]
    
    for module, name in optional:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ⚠️  {name} - nieobecny (opcjonalny)")

def main():
    """Główna funkcja weryfikująca."""
    print_header("WERYFIKACJA INSTALACJI MUJOCO PLAYGROUND")
    print("\nTen skrypt sprawdzi czy Twoje środowisko jest poprawnie skonfigurowane.")
    print("Uruchom go po zakończeniu instalacji zgodnie z QUICK_START_PL.md\n")
    
    results = []
    
    # Wymagane sprawdzenia
    results.append(("Python 3.10+", check_python_version()))
    results.append(("JAX", check_module('jax')))
    results.append(("JAX Backend", check_jax_backend()))
    results.append(("MuJoCo", check_module('mujoco')))
    results.append(("MuJoCo Playground", check_mujoco_playground()))
    results.append(("Test symulacji", test_basic_simulation()))
    
    # Opcjonalne narzędzia
    check_optional_tools()
    
    # Podsumowanie
    print_header("PODSUMOWANIE")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nWynik: {passed}/{total} testów przeszło pomyślnie")
    print("\nSzczegóły:")
    for name, result in results:
        status = "✅ OK" if result else "❌ BŁĄD"
        print(f"  {status:10} - {name}")
    
    if passed == total:
        print("\n" + "="*70)
        print("  🎉 GRATULACJE! Wszystko działa poprawnie!")
        print("="*70)
        print("\nKolejne kroki:")
        print("  1. Uruchom przykłady: python przyklady_g1.py")
        print("  2. Przeczytaj PRZEWODNIK_G1_PL.md")
        print("  3. Rozpocznij trening!")
        return 0
    else:
        print("\n" + "="*70)
        print("  ⚠️  Niektóre testy nie przeszły")
        print("="*70)
        print("\nCo zrobić:")
        print("  1. Sprawdź komunikaty błędów powyżej")
        print("  2. Zobacz sekcję 'Możliwe problemy' w QUICK_START_PL.md")
        print("  3. Poproś kolegów lub prowadzącego o pomoc")
        return 1

if __name__ == "__main__":
    sys.exit(main())
