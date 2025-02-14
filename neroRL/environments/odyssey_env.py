from math import inf
import gymnasium as gym
import numpy as np
from typing import Optional
import subprocess
import os, os.path
import sys
import socket
import struct

class OdysseyEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array', 'human'], 'render_fps': 60}

    COMMAND_OUT_FRAME = bytes([1])
    COMMAND_OUT_RESET = bytes([2])
    COMMAND_OUT_RENDER = bytes([3])
    COMMAND_IN_ACK = bytes([1])
    COMMAND_IN_DATA = bytes([2])
    COMMAND_IN_RENDER = bytes([3])


    def __init__(self, stage: str, scenario: int, instance: str, romfs_path: str = "/scratch/odyssey/romfs", render_mode: str = None):
        self.render_mode = render_mode
        self.socket_file = "/tmp/odyssey-physics-"+instance+".sock"
        if os.path.exists(self.socket_file):
            os.remove(self.socket_file)
        
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.bind(self.socket_file)

        display = 2 if render_mode == "human" else 1 if render_mode == "rgb_array" else 0
        print(["/scratch/odyssey/build/OdysseyPhysics", stage, str(scenario), romfs_path, self.socket_file, str(display)])
        self.process = subprocess.Popen(["/scratch/odyssey/build/OdysseyPhysics", stage, str(scenario), romfs_path, self.socket_file, str(display)], stdout=sys.stdout, stderr=sys.stderr)

        self.socket.listen(1)
        self.conn, self.addr = self.socket.accept()

        self.observation_space = gym.spaces.Dict(
            {
                "playerPos": gym.spaces.Box(low=np.array([-6800, -100, 300]), high=np.array([1400, 1500, 2300]), shape=(3,), dtype=np.float32),
                "playerVel": gym.spaces.Box(low=np.array([-100, -100, -100]), high=np.array([100, 100, 100]), shape=(3,), dtype=np.float32),
                "playerQuat": gym.spaces.Box(low=np.array([-1, -1, -1, -1]), high=np.array([1, 1, 1, 1]), shape=(4,), dtype=np.float32),
                "states": gym.spaces.MultiBinary(96),
                "raycastResults": gym.spaces.Box(low=0, high=500_00, shape=(250,), dtype=np.float32),
                "counterQuickTurnJump": gym.spaces.Discrete(21, start=0),
                "counterContinuousJump": gym.spaces.Discrete(3, start=0),
                "isTouchingMoon": gym.spaces.Discrete(2),
                "isTouchingPoison": gym.spaces.Discrete(2),
            }
        )

        self.action_space = gym.spaces.Dict(
            {
                "buttons": gym.spaces.MultiBinary(3),  # B, Y, ZL
                "stickLeft": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "stickRight": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            }
        )


    def close(self):
        if self.socket is None:
            return
        
        self.process.kill()
        self.conn.close()
        self.socket.close()
        self.socket = None
        os.remove(self.socket_file)

    def readState(self):
        command = self.conn.recv(1)
        if(command != self.COMMAND_IN_DATA):
            raise Exception("OdysseyPhysics did not send COMMAND_IN_DATA! Instead: "+str(command)+", followed by "+str(self.conn.recv(300)))
        
        """
        struct __attribute__((packed)) DataPacket {
            char type = COMMAND_OUT_DATA;  // +1
            sead::Vector3f playerPos;  // 0x000
            sead::Vector3f playerVel;  // 0x00c
            sead::Quatf playerQuat;    // 0x018
            u128 stateBitMap;          // 0x028
            f32 raycastResults[250];   // 0x038
            u32 counterContinuousJump; // 0x420
            s32 counterQuickTurnJump;  // 0x424
            bool isTouchingMoon;       // 0x428
            bool isTouchingPoison;     // 0x429
        };
        """

        state_data = self.conn.recv(0x42a)
        playerPosX, playerPosY, playerPosZ = struct.unpack("=fff", state_data[0:0xc])
        playerVelX, playerVelY, playerVelZ = struct.unpack("=fff", state_data[0xc:0x18])
        playerQuatX, playerQuatY, playerQuatZ, playerQuatW = struct.unpack("=ffff", state_data[0x18:0x28])
        stateBitMap1, stateBitMap2 = struct.unpack("=QQ", state_data[0x28:0x38])
        raycastResults = list(struct.unpack("=250f", state_data[0x38:0x420]))
        counterContinuousJump = struct.unpack("=I", state_data[0x420:0x424])[0]
        counterQuickTurnJump = struct.unpack("=i", state_data[0x424:0x428])[0]
        isTouchingMoon, isTouchingPoison = struct.unpack("=??", state_data[0x428:0x42a])

        states = []  # total=96
        for i in range(64):
            states.append((stateBitMap1 >> i) & 1 == 1)
        for i in range(32):
            states.append((stateBitMap2 >> i) & 1 == 1)

        return {
            "playerPos": np.array([playerPosX, playerPosY, playerPosZ], dtype=np.float32),
            "playerVel": np.array([playerVelX, playerVelY, playerVelZ], dtype=np.float32),
            "playerQuat": np.array([playerQuatX, playerQuatY, playerQuatZ, playerQuatW], dtype=np.float32),
            "states": np.array(states),
            "raycastResults": np.array(raycastResults, dtype=np.float32),
            "counterContinuousJump": counterContinuousJump,
            "counterQuickTurnJump": counterQuickTurnJump,
            "isTouchingMoon": isTouchingMoon,
            "isTouchingPoison": isTouchingPoison,
        }
    
    def step(self, action):
        buttons_arr = np.asarray(action["buttons"]).astype(bool)
        buttons = 0
        buttons |= buttons_arr[0] << 1  # B
        buttons |= buttons_arr[1] << 4  # Y
        buttons |= buttons_arr[2] << 2  # ZL
        stickLeft = action["stickLeft"]
        stickRight = action["stickRight"]

        data = struct.pack("=cIffff", self.COMMAND_OUT_FRAME, buttons, stickLeft[0], stickLeft[1], stickRight[0], stickRight[1])
        self.conn.send(data)

        state = self.readState()

        observation = state

        reward = 0
        terminated = False
        if state["isTouchingMoon"]:
            reward = 1
            terminated = True
        if state["isTouchingPoison"]:
            reward = -0.1
            terminated = True
        
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.conn.send(struct.pack("=c", self.COMMAND_OUT_RESET))

        state = self.readState()

        observation = state
        info = {}

        return observation, info

    def render(self):
        self.conn.send(struct.pack("=c", self.COMMAND_OUT_RENDER))
        command = self.conn.recv(1)
        if(command != self.COMMAND_IN_RENDER):
            raise Exception("OdysseyPhysics did not send COMMAND_IN_RENDER! Instead: "+str(command)+", followed by "+str(self.conn.recv(300)))
        
        render_data = self.conn.recv(1920*1080*3, socket.MSG_WAITALL)
        if len(render_data) != 1920*1080*3:
            raise Exception("OdysseyPhysics did not send enough data for render! Instead: "+str(len(render_data))+" bytes")
        # shape into nd.array with (x, y, 3) representing RGB values
        return np.flip(np.frombuffer(render_data, dtype=np.dtype('B')).reshape((1080, 1920, 3)), axis=0)
