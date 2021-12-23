from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from neroRL.environments import *

# Extend object IDs
OBJECT_TO_IDX["arrow_right"] = 11
OBJECT_TO_IDX["arrow_down"] = 12
OBJECT_TO_IDX["arrow_left"] = 13
OBJECT_TO_IDX["arrow_up"] = 14

COMMANDS = {
    "right" : (1, 0),
    "down"  : (0, 1),
    "left"  : (-1, 0),
    "up"    : (0, -1),
    "stay"  : (0, 0)
}

class ArrowRight(WorldObj):
    def __init__(self, color):
        super(ArrowRight, self).__init__("arrow_right", color)

    def render(self, img):
        tri_fn = point_in_triangle((0.12, 0.19),(0.87, 0.50),(0.12, 0.81),)
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*0)
        fill_coords(img, tri_fn, COLORS[self.color])

class ArrowDown(WorldObj):
    def __init__(self, color):
        super(ArrowDown, self).__init__("arrow_down", color)

    def render(self, img):
        tri_fn = point_in_triangle((0.12, 0.19),(0.87, 0.50),(0.12, 0.81),)
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*1)
        fill_coords(img, tri_fn, COLORS[self.color])

class ArrowLeft(WorldObj):
    def __init__(self, color):
        super(ArrowLeft, self).__init__("arrow_left", color)

    def render(self, img):
        tri_fn = point_in_triangle((0.12, 0.19),(0.87, 0.50),(0.12, 0.81),)
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*2)
        fill_coords(img, tri_fn, COLORS[self.color])

class ArrowUp(WorldObj):
    def __init__(self, color):
        super(ArrowUp, self).__init__("arrow_up", color)

    def render(self, img):
        tri_fn = point_in_triangle((0.12, 0.19),(0.87, 0.50),(0.12, 0.81),)
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*3)
        fill_coords(img, tri_fn, COLORS[self.color])

class MortarEnv(MiniGridEnv):
    def __init__(self, seed=None):
        self.num_available_commands = 5 # right, down, left, up, stay
        self.num_commands = 5           # how many commands to execute
        self.command_duration = 5       # how much time the agent has to move to the commanded position
        max_steps = self.num_commands * self.command_duration + 1
        super().__init__(grid_size=9, max_steps=max_steps, seed=seed, see_through_walls=True)

    def _gen_grid(self, width, height):
        self.current_command = 0

        # Create an empty grid and its surrounding walls
        self.grid = Grid(width, height)
        self.grid.wall_rect(1, 1, width-2, height-2)

        # Sample a start position and a start rotation
        self.agent_pos = self._rand_pos(2, width - 2, 2, height - 2)
        self.agent_dir = self._rand_int(0, 3)

        # Sample commands
        self.commands = self._generate_commands(self.agent_pos)
        # Render (place) commands in the first row of tiles
        for i in range(len(self.commands)):
            if self.commands[i] == "right":
                obj = ArrowRight("green")
            elif self.commands[i] == "down":
                obj = ArrowDown("green")
            elif self.commands[i] == "left":
                obj = ArrowLeft("green")
            elif self.commands[i] == "up":
                obj = ArrowUp("green")
            elif self.commands[i] == "stay":
                obj = Ball("green")
            self.grid.set(i, 0, obj)

        # Initial target position
        self.target_pos = (self.agent_pos[0] + COMMANDS[self.commands[0]][0],
                            self.agent_pos[1] + COMMANDS[self.commands[0]][1])

        # Mission description
        self.mission = "execute the green commands correctly"

    def _generate_commands(self, start_pos):
        simulated_pos = start_pos
        commands = []
        for i in range(self.num_commands):
            # Retrieve valid commands (we cannot walk on to a wall)
            valid_commands = self._get_valid_commands(simulated_pos)            
            # Sample one command from the available ones
            sample = self._rand_elem(valid_commands)
            commands.append(sample)
            # Update the simulated position
            simulated_pos = (simulated_pos[0] + COMMANDS[sample][0], simulated_pos[1] + COMMANDS[sample][1])
        return commands

    def _get_valid_commands(self, pos):
        # Check whether each command can be executed or not
        valid_commands = []
        for key, value in COMMANDS.items():
            obj = self.grid.get(pos[0] + value[0], pos[1] + value[1])
            if obj is None:
                valid_commands.append(key)
        # Return the commands that can be executed
        return valid_commands

    def _toggle_lava_on(self, target_pos):
        # Place lava on each field execept for the target one
        for i in range(2, 7):
            for j in range(2, 7):
                if target_pos != (i , j):
                    self.grid.set(i, j, Lava())
    
    def _toggle_lave_off(self):
        # Clear the lava
        for i in range(2, 7):
            for j in range(2, 7):
                self.grid.set(i, j, None)

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        # Process the command execution logic
        # One command is alive for command_duration steps
        if (self.step_count) % (self.command_duration) == 0 and self.step_count > 0:
            # Check if to be executed commands are still remaining
            if self.current_command < self.num_commands:
                self.current_command += 1
                # Toggle lava on for the next step
                self._toggle_lava_on(self.target_pos)
                # Check if the agent is on the target position
                if tuple(self.agent_pos) == self.target_pos:
                    # Success!
                    if self.current_command < self.num_commands:
                        # Update target position
                        self.target_pos = (self.target_pos[0] + COMMANDS[self.commands[self.current_command]][0],
                                            self.target_pos[1] + COMMANDS[self.commands[self.current_command]][1])
                    reward = 0.1
                # If the agent is not on the target position, terminate the episode
                else:
                    # Failure!
                    done = True
                    reward = -0.1
            # Finish the episode once all commands are completed
            if self.current_command >= self.num_commands:
                # All commands completed!
                done = True
                print(self.step_count)
        else:
            # Turn off lava
            self._toggle_lave_off()

        # Make sure that the lava signals a negative reward
        if done and reward == 0:
            reward = -0.1

        return obs, reward, done, info

register(
    id="MiniGrid-Mortar-v0",
    entry_point="neroRL.environments:MortarEnv"
)