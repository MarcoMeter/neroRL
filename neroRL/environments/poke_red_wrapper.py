import uuid
from pathlib import Path
sess_id = str(uuid.uuid4())[:8]
sess_path = Path(f'session_{sess_id}')

ep_length = 2048 * 10
env_train_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': './neroRL/environments/poke_red/has_pokedex_nballs.state', 'max_steps': ep_length, 
                'print_rewards': False, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': './neroRL/environments/poke_red/PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
                'use_screen_explore': True, 'reward_scale': 4, 'extra_buttons': False,
                'explore_weight': 3 # 2.5
            }

env_enjoy_config = {
                'headless': False, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': './neroRL/environments/poke_red/has_pokedex_nballs.state', 'max_steps': 2**23, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': './neroRL/environments/poke_red/PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
                'use_screen_explore': True, 'reward_scale': 4, 'extra_buttons': True,
                'explore_weight': 3 # 2.5
            }