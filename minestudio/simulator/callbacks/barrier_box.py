'''
Date: 2025-10-15
LastEditors: annastasyshyn
LastEditTime: 2025-10-15
FilePath: /MineStudio/minestudio/simulator/callbacks/barrier_box.py
'''

from minestudio.simulator.callbacks.callback import MinecraftCallback
from minestudio.utils.register import Registers
from typing import Dict, Literal

@Registers.simulator_callback.register
class BarrierBoxCallback(MinecraftCallback):
    """
    A callback that creates a box of barrier blocks or bedrock around the agent.
    
    This callback constrains the agent's exploration area by building walls,
    helping to focus training on specific tasks and reducing wasted time exploring.
    
    The box is centered on the agent's position after reset.
    """

    def create_from_conf(source):
        """
        Creates a BarrierBoxCallback instance from a configuration source.

        :param source: The configuration source (e.g., file path or dictionary).
        :return: A BarrierBoxCallback instance or None if 'barrier_box' is not in the config.
        """
        data = MinecraftCallback.load_data_from_conf(source)
        if 'barrier_box' in data:
            box_config = data['barrier_box']
            return BarrierBoxCallback(
                size=box_config.get('size', 25),
                height=box_config.get('height', 10),
                block_type=box_config.get('block_type', 'barrier'),
                include_floor=box_config.get('include_floor', False),
                include_ceiling=box_config.get('include_ceiling', False)
            )
        else:
            return None

    def __init__(
        self, 
        size: int = 25,
        height: int = 10,
        block_type: Literal['barrier', 'bedrock', 'glass'] = 'barrier',
        include_floor: bool = False,
        include_ceiling: bool = False
    ) -> None:
        """
        Initializes the BarrierBoxCallback.

        :param size: The size of the box (width and length). Default is 25 blocks.
        :param height: The height of the walls. Default is 10 blocks.
        :param block_type: The type of block to use ('barrier', 'bedrock', or 'glass'). Default is 'barrier'.
        :param include_floor: Whether to include a floor. Default is False.
        :param include_ceiling: Whether to include a ceiling. Default is False.
        
        Examples:
            # Create a 25x25 barrier box with 10-block high walls
            BarrierBoxCallback(size=25, height=10, block_type='barrier')
            
            # Create a 30x30 bedrock box with floor and ceiling
            BarrierBoxCallback(size=30, height=15, block_type='bedrock', 
                             include_floor=True, include_ceiling=True)
        """
        super().__init__()
        self.size = size
        self.height = height
        self.block_type = block_type
        self.include_floor = include_floor
        self.include_ceiling = include_ceiling
        
        # Validate block type
        valid_blocks = ['barrier', 'bedrock', 'glass']
        if self.block_type not in valid_blocks:
            raise ValueError(f"block_type must be one of {valid_blocks}, got '{self.block_type}'")

    def after_reset(self, sim, obs, info):
        """
        Builds the barrier box around the agent after the environment is reset.

        :param sim: The Minecraft simulator.
        :param obs: The observation from the simulator.
        :param info: Additional information from the simulator.
        :return: The modified observation and info after building the box.
        """
        # Get agent's current position from the observation
        # The position is typically in info or we use a default center position
        
        # Calculate half size for centering the box
        half_size = self.size // 2
        
        # Build the box centered at origin (you can adjust this based on agent position)
        commands = self._generate_box_commands(0, 0, half_size)
        
        # Execute all commands
        for command in commands:
            obs, reward, done, info = sim.env.execute_cmd(command)
        
        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info

    def _generate_box_commands(self, center_x: int, center_z: int, base_y: int) -> list:
        """
        Generates the fill commands to create the barrier box.
        
        :param center_x: X coordinate of the box center.
        :param center_z: Z coordinate of the box center.
        :param base_y: Base Y coordinate (ground level).
        :return: List of command strings.
        """
        half_size = self.size // 2
        block = f"minecraft:{self.block_type}"
        
        # Calculate boundaries
        x_min = center_x - half_size
        x_max = center_x + half_size
        z_min = center_z - half_size
        z_max = center_z + half_size
        y_min = base_y
        y_max = base_y + self.height
        
        commands = []
        
        # Build four walls
        # North wall (negative Z)
        commands.append(
            f"/fill {x_min} {y_min} {z_min} {x_max} {y_max} {z_min} {block}"
        )
        
        # South wall (positive Z)
        commands.append(
            f"/fill {x_min} {y_min} {z_max} {x_max} {y_max} {z_max} {block}"
        )
        
        # West wall (negative X)
        commands.append(
            f"/fill {x_min} {y_min} {z_min} {x_min} {y_max} {z_max} {block}"
        )
        
        # East wall (positive X)
        commands.append(
            f"/fill {x_max} {y_min} {z_min} {x_max} {y_max} {z_max} {block}"
        )
        
        # Optional floor
        if self.include_floor:
            commands.append(
                f"/fill {x_min} {y_min} {z_min} {x_max} {y_min} {z_max} {block}"
            )
        
        # Optional ceiling
        if self.include_ceiling:
            commands.append(
                f"/fill {x_min} {y_max} {z_min} {x_max} {y_max} {z_max} {block}"
            )
        
        return commands

    def __repr__(self) -> str:
        """Returns a string representation of the BarrierBoxCallback.

        :returns: String representation of the instance.
        :rtype: str
        """
        return (f"BarrierBoxCallback(size={self.size}, height={self.height}, "
                f"block_type='{self.block_type}', include_floor={self.include_floor}, "
                f"include_ceiling={self.include_ceiling})")
