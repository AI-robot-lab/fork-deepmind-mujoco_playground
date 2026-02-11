"""
Przykłady użycia MuJoCo Playground z robotem Unitree G1.

Ten plik zawiera gotowe przykłady kodu dla studentów, które pokazują:
1. Podstawową symulację robota G1
2. Trening polityki sterowania
3. Ewaluację wytrenowanego modelu
4. Wizualizację i analizę wyników

Każdy przykład jest samodzielny i może być uruchomiony osobno.
"""

# ============================================================================
# PRZYKŁAD 1: Podstawowa symulacja z losowymi akcjami
# ============================================================================

def przyklad_1_podstawowa_symulacja():
    """
    Najbardziej podstawowy przykład - załaduj środowisko i wykonaj kilka kroków.
    
    Cel: Zrozumieć podstawową strukturę środowiska i pętlę symulacji.
    """
    import jax
    import jax.numpy as jp
    from mujoco_playground import locomotion
    
    print("\n" + "="*60)
    print("PRZYKŁAD 1: Podstawowa symulacja robota G1")
    print("="*60)
    
    # Krok 1: Załaduj środowisko
    # ---------------------------
    # G1JoystickFlatTerrain to środowisko gdzie robot G1 chodzi po płaskim
    # terenie i reaguje na komendy prędkości (jak z joysticka).
    print("\nŁadowanie środowiska G1JoystickFlatTerrain...")
    env = locomotion.load('G1JoystickFlatTerrain')
    
    print(f"✓ Środowisko załadowane!")
    print(f"  - Wymiar obserwacji: {env.observation_size}")
    print(f"  - Wymiar akcji: {env.action_size}")
    print(f"  - Krok czasu: {env.dt}s")
    
    # Krok 2: Zainicjalizuj stan początkowy
    # --------------------------------------
    # PRNGKey to generator liczb losowych w JAX - potrzebny do losowania
    # pozycji startowej i innych elementów losowych.
    rng = jax.random.PRNGKey(42)  # 42 to "seed" - dla powtarzalności
    state = jax.jit(env.reset)(rng)  # jit kompiluje funkcję dla GPU
    
    print(f"\n✓ Stan początkowy zainicjalizowany")
    print(f"  - Pozycja robota [x,y,z]: {state.data.qpos[:3]}")
    
    # Krok 3: Wykonaj symulację
    # -------------------------
    # W pętli: generuj akcje -> wykonaj krok -> obserwuj rezultat
    print("\nRozpoczynam symulację 100 kroków...")
    
    rewards_list = []
    for i in range(100):
        # Generuj losową akcję (małe wartości dla bezpieczeństwa)
        rng, action_key = jax.random.split(rng)
        action = jax.random.uniform(
            action_key, 
            (env.action_size,), 
            minval=-0.1,  # Małe akcje = delikatne ruchy
            maxval=0.1
        )
        
        # Wykonaj krok symulacji
        state = env.step(state, action)
        rewards_list.append(float(state.reward))
        
        # Co 20 kroków wypisz status
        if i % 20 == 0:
            print(f"  Krok {i:3d}: nagroda = {state.reward:7.3f}, "
                  f"pozycja z = {state.data.qpos[2]:5.3f}m")
    
    # Podsumowanie
    rewards = jp.array(rewards_list)
    print(f"\n✓ Symulacja zakończona!")
    print(f"  - Średnia nagroda: {jp.mean(rewards):.3f}")
    print(f"  - Suma nagród: {jp.sum(rewards):.3f}")
    print(f"  - Końcowa pozycja [x,y,z]: {state.data.qpos[:3]}")


# ============================================================================
# PRZYKŁAD 2: Wizualizacja trajektorii - zapis do wideo
# ============================================================================

def przyklad_2_wizualizacja():
    """
    Wykonaj symulację i zapisz do wideo.
    
    Cel: Nauczyć się jak wizualizować zachowanie robota.
    """
    import jax
    import jax.numpy as jp
    import mediapy as media
    from mujoco_playground import locomotion
    
    print("\n" + "="*60)
    print("PRZYKŁAD 2: Wizualizacja trajektorii robota")
    print("="*60)
    
    # Załaduj środowisko
    print("\nŁadowanie środowiska...")
    env = locomotion.load('G1JoystickFlatTerrain')
    
    # Inicjalizacja
    rng = jax.random.PRNGKey(123)
    state = jax.jit(env.reset)(rng)
    
    # Zbieraj stany do wizualizacji
    print("Wykonywanie symulacji 300 kroków...")
    states = [state]
    
    for i in range(300):
        rng, key = jax.random.split(rng)
        # Generuj akcje oscylujące (jak proste chodzenie)
        t = i * env.dt
        action = 0.1 * jp.sin(2 * jp.pi * 0.5 * t) * jp.ones(env.action_size)
        action = action + jax.random.uniform(key, action.shape, minval=-0.02, maxval=0.02)
        
        state = env.step(state, action)
        states.append(state)
        
        if i % 50 == 0:
            print(f"  Postęp: {i}/300 kroków")
    
    # Renderowanie wideo
    print("\nRenderowanie wideo (może chwilę potrwać)...")
    frames = env.render(states, height=480, width=640)
    
    output_file = 'g1_przyklad2_trajektoria.mp4'
    media.write_video(output_file, frames, fps=50)
    
    print(f"\n✓ Wideo zapisane jako '{output_file}'")
    print(f"  - Liczba klatek: {len(frames)}")
    print(f"  - Czas trwania: {len(frames)/50:.1f}s")


# ============================================================================
# PRZYKŁAD 3: Kontrola z komendami prędkości
# ============================================================================

def przyklad_3_kontrola_predkosci():
    """
    Sterowanie robotem przez zadawanie komend prędkości (jak joystick).
    
    Cel: Zrozumieć jak robot reaguje na komendy i jak są przetwarzane obserwacje.
    """
    import jax
    import jax.numpy as jp
    from mujoco_playground import locomotion
    
    print("\n" + "="*60)
    print("PRZYKŁAD 3: Kontrola z komendami prędkości")
    print("="*60)
    
    # Załaduj środowisko
    env = locomotion.load('G1JoystickFlatTerrain')
    rng = jax.random.PRNGKey(999)
    state = jax.jit(env.reset)(rng)
    
    # Sekwencja komend: stój -> idź do przodu -> skręć w lewo -> stój
    print("\nWykonywanie sekwencji komend:")
    
    # Komenda 1: Stój w miejscu (100 kroków)
    print("\n1. Komenda: Stój w miejscu")
    for i in range(100):
        action = jp.zeros(env.action_size)  # Zero akcji = próba utrzymania pozycji
        state = env.step(state, action)
    print(f"   Po 100 krokach: pozycja x = {state.data.qpos[0]:.3f}m")
    
    # Komenda 2: Idź do przodu (200 kroków)
    print("\n2. Komenda: Idź do przodu")
    start_x = state.data.qpos[0]
    for i in range(200):
        # Pozytywne akcje na przednich stawach = próba ruchu do przodu
        action = jp.zeros(env.action_size)
        action = action.at[0:6].set(0.3)  # Przednie stawy
        state = env.step(state, action)
    distance = state.data.qpos[0] - start_x
    print(f"   Przebyta odległość: {distance:.3f}m")
    
    # Komenda 3: Skręć w lewo (150 kroków)  
    print("\n3. Komenda: Skręć w lewo")
    start_yaw = jp.arctan2(2*state.data.qpos[5]*state.data.qpos[6], 
                            1-2*state.data.qpos[6]**2)
    for i in range(150):
        action = jp.zeros(env.action_size)
        action = action.at[0:3].set(0.2)   # Lewa strona
        action = action.at[3:6].set(-0.2)  # Prawa strona
        state = env.step(state, action)
    end_yaw = jp.arctan2(2*state.data.qpos[5]*state.data.qpos[6], 
                         1-2*state.data.qpos[6]**2)
    rotation = float(end_yaw - start_yaw) * 180 / jp.pi
    print(f"   Obrót: {rotation:.1f} stopni")
    
    print("\n✓ Sekwencja komend zakończona!")


# ============================================================================
# PRZYKŁAD 4: Analiza przestrzeni obserwacji
# ============================================================================

def przyklad_4_analiza_obserwacji():
    """
    Przeanalizuj co zawiera wektor obserwacji robota.
    
    Cel: Zrozumieć jakie informacje robot "widzi" i jak je interpretować.
    """
    import jax
    import jax.numpy as jp
    from mujoco_playground import locomotion
    
    print("\n" + "="*60)
    print("PRZYKŁAD 4: Analiza przestrzeni obserwacji")
    print("="*60)
    
    # Załaduj środowisko
    env = locomotion.load('G1JoystickFlatTerrain')
    rng = jax.random.PRNGKey(0)
    state = jax.jit(env.reset)(rng)
    
    print(f"\nWymiar obserwacji: {env.observation_size}")
    print(f"Typ obserwacji: {type(state.obs)}")
    
    # Wykonaj kilka kroków żeby zobaczyć jak się zmienia
    print("\nWartości obserwacji po kilku krokach:")
    for step in [0, 10, 50, 100]:
        if step > 0:
            for _ in range(10):
                action = jax.random.uniform(rng, (env.action_size,), minval=-0.05, maxval=0.05)
                state = env.step(state, action)
        
        obs = state.obs
        print(f"\n  Krok {step}:")
        print(f"    - Kształt: {obs.shape}")
        print(f"    - Min: {jp.min(obs):.3f}")
        print(f"    - Max: {jp.max(obs):.3f}")
        print(f"    - Średnia: {jp.mean(obs):.3f}")
        print(f"    - Pierwsze 5 wartości: {obs[:5]}")
    
    print("\n✓ Analiza zakończona!")
    print("\nINFO: Obserwacje zazwyczaj zawierają:")
    print("  - Pozycje stawów (joint positions)")
    print("  - Prędkości stawów (joint velocities)")
    print("  - Orientację robota (IMU/gravity vector)")
    print("  - Prędkość liniową i kątową")
    print("  - Historię poprzednich akcji")
    print("  - Komendy prędkości (target velocities)")


# ============================================================================
# PRZYKŁAD 5: Proste uczenie przez imitację
# ============================================================================

def przyklad_5_zbieranie_danych_demonstracyjnych():
    """
    Zbierz dane z heurystycznej polityki (np. PD controller).
    
    Cel: Pokazać jak można zbierać dane do uczenia nadzorowanego lub imitacji.
    """
    import jax
    import jax.numpy as jp
    from mujoco_playground import locomotion
    import pickle
    
    print("\n" + "="*60)
    print("PRZYKŁAD 5: Zbieranie danych demonstracyjnych")
    print("="*60)
    
    # Załaduj środowisko
    env = locomotion.load('G1JoystickFlatTerrain')
    rng = jax.random.PRNGKey(42)
    
    # Prosta heurystyczna polityka: sinusoidalne wzory dla chodzenia
    def heuristic_policy(t, phase_offset=0.0):
        """
        Prosta polityka oparta na sinusoidalnych wzorach.
        W rzeczywistości użylibyśmy bardziej zaawansowanego controllera.
        """
        freq = 1.0  # Hz
        amplitude = 0.3
        action = amplitude * jp.sin(2 * jp.pi * freq * t + phase_offset)
        return action * jp.ones(env.action_size)
    
    # Zbieraj trajektorie
    print("\nZbieranie 5 trajektorii demonstracyjnych...")
    trajectories = []
    
    for traj_idx in range(5):
        rng, reset_key = jax.random.split(rng)
        state = jax.jit(env.reset)(reset_key)
        
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
        }
        
        print(f"\n  Trajektoria {traj_idx + 1}/5:")
        for step in range(200):
            # Generuj akcję z heurystyki
            t = step * env.dt
            action = heuristic_policy(t, phase_offset=traj_idx * 0.5)
            
            # Zapisz dane
            trajectory['observations'].append(state.obs)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(state.reward)
            
            # Krok symulacji
            state = env.step(state, action)
        
        # Konwertuj do numpy arrays
        trajectory['observations'] = jp.stack(trajectory['observations'])
        trajectory['actions'] = jp.stack(trajectory['actions'])
        trajectory['rewards'] = jp.array(trajectory['rewards'])
        
        trajectories.append(trajectory)
        
        total_reward = jp.sum(trajectory['rewards'])
        print(f"    Suma nagród: {total_reward:.2f}")
    
    # Zapisz dane
    output_file = 'g1_demonstracje.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(trajectories, f)
    
    print(f"\n✓ Dane zapisane do '{output_file}'")
    print(f"  - Liczba trajektorii: {len(trajectories)}")
    print(f"  - Kroków na trajektorię: {len(trajectories[0]['observations'])}")
    print(f"\nTe dane mogą być użyte do:")
    print("  - Uczenia nadzorowanego (behavioral cloning)")
    print("  - Inicjalizacji polityki przed RL")
    print("  - Analizy optymalnych akcji")


# ============================================================================
# PRZYKŁAD 6: Testowanie stabilności robota
# ============================================================================

def przyklad_6_test_stabilnosci():
    """
    Przetestuj stabilność robota przy różnych zakłóceniach.
    
    Cel: Sprawdzić jak robot reaguje na zewnętrzne siły (ważne dla sim-to-real).
    """
    import jax
    import jax.numpy as jp
    from mujoco_playground import locomotion
    
    print("\n" + "="*60)
    print("PRZYKŁAD 6: Test stabilności robota")
    print("="*60)
    
    # Załaduj środowisko z włączonymi losowymi pchnięciami
    print("\nŁadowanie środowiska z domain randomization...")
    env = locomotion.load('G1JoystickFlatTerrain')
    
    # Test 1: Bez zakłóceń
    print("\nTest 1: Robot bez zakłóceń zewnętrznych")
    rng = jax.random.PRNGKey(0)
    state = jax.jit(env.reset)(rng)
    
    survived_steps = 0
    for i in range(500):
        action = jp.zeros(env.action_size)  # Próbuj tylko stać
        state = env.step(state, action)
        
        # Sprawdź czy robot się nie przewrócił (wysokość > 0.3m)
        if state.data.qpos[2] < 0.3:
            print(f"  Robot upadł po {i} krokach")
            break
        survived_steps = i + 1
    
    print(f"  Rezultat: przetrwał {survived_steps}/500 kroków")
    
    # Test 2: Z losowymi siłami
    print("\nTest 2: Robot z losowymi siłami zewnętrznymi")
    rng = jax.random.PRNGKey(1)
    state = jax.jit(env.reset)(rng)
    
    survived_steps = 0
    for i in range(500):
        action = jp.zeros(env.action_size)
        
        # Dodaj losową siłę co 50 kroków
        if i % 50 == 0 and i > 0:
            # W prawdziwej implementacji użylibyśmy xfrc_applied
            # Tu symulujemy to przez zakłócenie akcji
            rng, force_key = jax.random.split(rng)
            disturbance = jax.random.uniform(force_key, action.shape, minval=-2.0, maxval=2.0)
            action = action + disturbance
            print(f"  Krok {i}: Zastosowano siłę zakłócającą")
        
        state = env.step(state, action)
        
        if state.data.qpos[2] < 0.3:
            print(f"  Robot upadł po {i} krokach")
            break
        survived_steps = i + 1
    
    print(f"  Rezultat: przetrwał {survived_steps}/500 kroków")
    
    print("\n✓ Testy stabilności zakończone!")
    print("\nWNIOSKI:")
    print("  - Im dłużej robot przetrwa, tym bardziej stabilna jest polityka")
    print("  - Domain randomization podczas treningu poprawia odporność")
    print("  - Testy stabilności są kluczowe przed transferem do rzeczywistości")


# ============================================================================
# Główna funkcja uruchamiająca wszystkie przykłady
# ============================================================================

def uruchom_wszystkie_przyklady():
    """Uruchom wszystkie przykłady po kolei."""
    import sys
    
    print("\n" + "="*60)
    print("PRZYKŁADY UŻYCIA MUJOCO PLAYGROUND Z ROBOTEM G1")
    print("="*60)
    print("\nTen skrypt zawiera 6 przykładów pokazujących różne aspekty")
    print("pracy z robotem Unitree G1 w symulacji.")
    print("\nMożesz uruchomić:")
    print("  - Wszystkie przykłady: python przyklady_g1.py")
    print("  - Konkretny przykład: python przyklady_g1.py <numer>")
    print("    np: python przyklady_g1.py 2")
    
    # Sprawdź argumenty
    if len(sys.argv) > 1:
        try:
            example_num = int(sys.argv[1])
            if 1 <= example_num <= 6:
                print(f"\nUruchamiam tylko przykład {example_num}...")
                examples = [
                    przyklad_1_podstawowa_symulacja,
                    przyklad_2_wizualizacja,
                    przyklad_3_kontrola_predkosci,
                    przyklad_4_analiza_obserwacji,
                    przyklad_5_zbieranie_danych_demonstracyjnych,
                    przyklad_6_test_stabilnosci,
                ]
                examples[example_num - 1]()
                return
            else:
                print(f"\nBłąd: Numer przykładu musi być między 1 a 6, podano: {example_num}")
                return
        except ValueError:
            print(f"\nBłąd: '{sys.argv[1]}' nie jest poprawnym numerem")
            return
    
    # Uruchom wszystkie
    print("\nUruchamiam wszystkie przykłady (to może potrwać kilka minut)...")
    input("\nNaciśnij Enter aby kontynuować...")
    
    try:
        przyklad_1_podstawowa_symulacja()
        input("\nNaciśnij Enter aby przejść do następnego przykładu...")
        
        przyklad_2_wizualizacja()
        input("\nNaciśnij Enter aby przejść do następnego przykładu...")
        
        przyklad_3_kontrola_predkosci()
        input("\nNaciśnij Enter aby przejść do następnego przykładu...")
        
        przyklad_4_analiza_obserwacji()
        input("\nNaciśnij Enter aby przejść do następnego przykładu...")
        
        przyklad_5_zbieranie_danych_demonstracyjnych()
        input("\nNaciśnij Enter aby przejść do następnego przykładu...")
        
        przyklad_6_test_stabilnosci()
        
        print("\n" + "="*60)
        print("WSZYSTKIE PRZYKŁADY ZAKOŃCZONE!")
        print("="*60)
        print("\nGratulacje! Przeszedłeś przez wszystkie podstawowe przykłady.")
        print("Teraz możesz:")
        print("  1. Eksperymentować z parametrami w przykładach")
        print("  2. Przejść do treningu własnej polityki (train_jax_ppo.py)")
        print("  3. Zapoznać się z PRZEWODNIK_G1_PL.md dla więcej informacji")
        
    except KeyboardInterrupt:
        print("\n\nPrzerwano przez użytkownika (Ctrl+C)")
    except Exception as e:
        print(f"\n\nWystąpił błąd: {e}")
        print("Sprawdź czy wszystkie zależności są zainstalowane (jax, mujoco_playground, mediapy)")


if __name__ == "__main__":
    uruchom_wszystkie_przyklady()
