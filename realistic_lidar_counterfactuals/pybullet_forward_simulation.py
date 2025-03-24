import random

import pybullet as p
import pybullet_data
import numpy as np
import os
import time

wheel_distance = 0.287
wheel_radius = 0.034

abs_max_env_size = 12.0
goal_threshold = 0.2 / abs_max_env_size




def compute_wheel_velocities(lin_vel, ang_vel):
    # Function to convert Twist-like command to wheel velocities
    v_left = (lin_vel - (wheel_distance / 2.0) * ang_vel) / wheel_radius
    v_right = (lin_vel + (wheel_distance / 2.0) * ang_vel) / wheel_radius
    return v_left, v_right

class DummyDRLPolicy:
    """
    A dummy DRL policy that computes an action based on the robot's current state.
    Replace this with your actual DRL policy logic.
    """

    def __call__(self, obs):
        return np.array([1.0, 1.0])




class Turtlebot3Simulation:
    def __init__(self, policy=None, timeStep=1. / 240, start_pos=None, start_orientation_euler=None, target_position=None):
        """
        Initializes the simulator with a basic environment containing the ground plane
        and Turtlebot3. The simulation is set up to run as fast as possible.

        This instance explicitly connects in DIRECT mode (headless) for speed.

        Parameters:
            policy: an object with a compute_action() method; if None, a dummy policy is used.
            timeStep: simulation timestep.
            start_pos: initial position for Turtlebot3 (default: [0, 0, 0.1]).
            start_orientation_euler: initial Euler angles for Turtlebot3 (default: [0, 0, 0]).
        """
        # Connect in DIRECT mode for fast, headless simulation.
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.max_linear_vel = 0.26  # 0.1
        self.max_angular_vel = 1.82  # 1.0
        self.left_wheel_joint = 1  # Adjust the joint index based on your URDF configuration
        self.right_wheel_joint = 2

        # Control loop frequency
        control_frequency = 10  # 10 Hz control signal update rate
        control_time_step = 1.0 / control_frequency  # Time per control loop iteration

        # Calculate how many simulation steps to run per control loop iteration
        self.steps_per_control_loop = int(control_time_step / timeStep)

        self.abs_max_env_size = 12.0
        self.goal_threshold = 0.2 / self.abs_max_env_size
        self.num_rays = 180
        self.ray_angle_range = 360
        self.ray_length = 3.5
        self.timeStep = timeStep
        self.policy = policy if policy is not None else DummyDRLPolicy()
        self.start_pos = start_pos if start_pos is not None else [0, 0, 0.05]
        self.target_pos = target_position if target_position is not None else [1, 0]
        self.start_orientation_euler = start_orientation_euler if start_orientation_euler is not None else [0, 0, 0]
        self.start_orientation = p.getQuaternionFromEuler(self.start_orientation_euler)

        self._saved_state_id = None

        # Initialize the simulation environment.
        self._init_simulation()

    def _init_simulation(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timeStep)

        self.plane = p.loadURDF("plane.urdf")
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.turtlebot3_id = p.loadURDF(
            self.script_dir + "/../../pybullet_turtlebot3_env/turtlebot3_description/urdf/turtlebot3_waffle_pi.urdf",
            self.start_pos,
            self.start_orientation
        )

        for _ in range(50):
            p.stepSimulation()

        self._saved_state_id = p.saveState()
        self.objects = []

    def reset_environment(self):
        """
        Resets simulation to clean initial state (plane + robot) and removes all added obstacles.
        """
        # Remove any previously added obstacles
        for obj_id in self.objects:
            p.removeBody(obj_id)
        self.objects = []

        # Restore fast initial state
        if self._saved_state_id is not None:
            p.restoreState(self._saved_state_id)
            p.resetBasePositionAndOrientation(self.turtlebot3_id, self.start_pos, self.start_orientation)
            p.resetBaseVelocity(self.turtlebot3_id, [0, 0, 0], [0, 0, 0])
        else:
            self._init_simulation()

    def add_objects(self, objects_list):
        """
        Adds a list of obstacles to the simulation.
        Only supports:
          - Circles: represented as cylinders (requires "radius" and "height")
          - Cuboids: represented as boxes (requires "half_extents": [half_x, half_y, half_z])

        Each obstacle dictionary must include:
          - "shape": either "circle" or "cuboid".
          - "position": [x, y, z].
          - "orientation": [roll, pitch, yaw] (in radians).

        Examples:
            For a circle:
              {"shape": "circle", "position": [1, 0, 0.5], "orientation": [0, 0, 0],
               "radius": 0.5, "height": 0.2}
            For a cuboid:
              {"shape": "cuboid", "position": [-1, 0, 0.5], "orientation": [0, 0, 0],
               "half_extents": [0.5, 0.3, 0.2]}
        """
        for obj in objects_list:
            shape_type = obj.get("shape")
            pos = obj.get("position", [0, 0, 0])
            orn_euler = obj.get("orientation", [0, 0, 0])
            orn = p.getQuaternionFromEuler(orn_euler)

            if shape_type == "circle":
                radius = obj.get("radius")
                height = obj.get("height")
                if radius is None or height is None:
                    raise ValueError("For a 'circle', both 'radius' and 'height' must be provided.")
                collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
                visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height)
            elif shape_type == "cuboid":
                half_extents = obj.get("half_extents")
                if half_extents is None or len(half_extents) != 3:
                    raise ValueError("For a 'cuboid', 'half_extents' must be provided as a list of three values.")
                collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
                visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents)
            else:
                raise ValueError("Unsupported shape type. Only 'circle' and 'cuboid' are allowed.")

            # Create a static object (mass=0).
            obj_id = p.createMultiBody(baseMass=0,
                                       baseCollisionShapeIndex=collision_shape,
                                       baseVisualShapeIndex=visual_shape,
                                       basePosition=pos,
                                       baseOrientation=orn)
            self.objects.append(obj_id)

    def run_simulation(self, n_timesteps):
        """
        Runs the simulation for a given number of timesteps using the policy controlling Turtlebot3.
        Records and returns the Turtlebot3 trajectory as a list of positions.
        The simulation is executed as fast as possible (no delays).
        """

        obs_list = []
        info_list = []
        action_list = []
        # Assuming the Turtlebot3's two wheel joints are at indices 0 and 1.
        wheel_joint_indices = [0, 1]
        obs = self._get_observation()
        obs_list.append(obs)
        for _ in range(n_timesteps):

            action = self.policy(obs)

            obs, info = self.step(action)

            obs_list.append(obs)
            info_list.append(info)
            action_list.append(action)

            #time.sleep(self.timeStep*10)
        return obs_list, info_list, action_list

    def perform_lidar_scan(self):
        # Remove previous debug lines and text only if the render mode is set to 'human'
        """if self.render_mode == 'human':
            for line_id in self.debug_line_ids:
                p.removeUserDebugItem(line_id)
            for text_id in self.debug_text_ids:
                p.removeUserDebugItem(text_id)

            self.debug_line_ids.clear()
            self.debug_text_ids.clear()"""

        # Get the robot's current position and orientation in the world frame
        robot_position, robot_orientation = p.getBasePositionAndOrientation(self.turtlebot3_id)

        # Convert the quaternion orientation to a rotation matrix
        robot_matrix = p.getMatrixFromQuaternion(robot_orientation)
        robot_rotation = np.array(robot_matrix).reshape(3, 3)

        # LIDAR's local position relative to base_link from the URDF
        lidar_offset_local = np.array([-0.064, 0.0, 0.0])  # LIDAR's position in the base_link frame

        # Rotate the LIDAR's position using the robot's orientation (to move from local to world frame)
        lidar_offset_world = robot_rotation @ lidar_offset_local

        # Calculate the LIDAR's position in the world frame by adding the robot's global position
        lidar_position_world = np.array(robot_position) + lidar_offset_world

        # LIDAR rays start at the calculated world position with a slight height offset
        ray_start = lidar_position_world + np.array([0, 0, 0.1])  # Slight height offset for LIDAR sensor

        # Generate all angles at once
        angles_rad = np.linspace(0.0, 2 * np.pi, self.num_rays, endpoint=False)

        # Directions in local frame (shape: [num_rays, 3])
        directions_local = np.stack((np.cos(angles_rad), np.sin(angles_rad), np.zeros_like(angles_rad)), axis=1)

        # Transform directions to world frame using the robot's rotation matrix
        directions_world = directions_local @ robot_rotation.T

        # Compute ray end positions (shape: [num_rays, 3])
        ray_end_positions = ray_start + directions_world * self.ray_length

        # Since ray_start is the same for all rays, replicate it to match the shape
        ray_start_positions = np.tile(ray_start, (self.num_rays, 1))

        # Perform ray casting in PyBullet
        ray_results = p.rayTestBatch(ray_start_positions.tolist(), ray_end_positions.tolist())

        # Extract hit fractions and compute distances
        hit_fractions = np.array([result[2] for result in ray_results])
        distances = self.ray_length * hit_fractions
        #print(f"min distance lidar: {min(distances)}")
        return distances.tolist()

    def step(self, action):

        lin_vel_cmd = np.clip(action[0] * self.max_linear_vel, -self.max_linear_vel, self.max_linear_vel)
        ang_vel_cmd = np.clip(action[1] * self.max_angular_vel, -self.max_angular_vel, self.max_angular_vel)

        v_left, v_right = compute_wheel_velocities(lin_vel_cmd, ang_vel_cmd)

        # Batch apply wheel velocities in PyBullet
        p.setJointMotorControlArray(
            self.turtlebot3_id,
            [self.left_wheel_joint, self.right_wheel_joint],
            p.VELOCITY_CONTROL,
            targetVelocities=[v_left, v_right]
        )

        # Perform batch stepping
        for _ in range(self.steps_per_control_loop):
            p.stepSimulation()

        observation = self._get_observation()

        current_distance = observation[-1]
        min_range = 0.20 / 3.5  # Normalized scan range
        state_np = np.array(observation)
        scan_range = state_np[:-3]

        # Get the robot's position, orientation, and velocity
        position, orientation = p.getBasePositionAndOrientation(self.turtlebot3_id)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.turtlebot3_id)

        # Calculate x, y position and velocity
        x_pos, y_pos = position[0], position[1]
        x_vel, y_vel = linear_velocity[0], linear_velocity[1]

        # Calculate yaw angle (orientation around z-axis) and angular velocity
        angle = p.getEulerFromQuaternion(orientation)[2]
        angular_vel = angular_velocity[2]  # Yaw rate


        collision = False
        goal_reached = False

        if min_range > min(scan_range) > 0:
            collision = True

        elif current_distance < goal_threshold:
            goal_reached = True

        info = {
            "is_success": goal_reached,
            "is_failure": collision,
            "x_pos": x_pos,
            "y_pos": y_pos,
            "x_vel": x_vel,
            "y_vel": y_vel,
            "angle": angle,
            "angular_vel": angular_vel
        }

        #print(f"terminated: {terminated}")


        return observation, info

    def _get_observation(self):
        # Get the current position and orientation of the robot
        robot_position, robot_orientation = p.getBasePositionAndOrientation(self.turtlebot3_id)
        robot_position = np.array(robot_position[:2])  # Extract x, y position
        target_vector = self.target_pos - robot_position  # Vector from robot to target

        # Calculate the distance to the target
        distance_to_target = np.linalg.norm(target_vector)
        distance_to_target = round(distance_to_target, 2)
        distance_to_target = np.clip(distance_to_target, 0.0, self.abs_max_env_size)/ self.abs_max_env_size

        # Calculate the robot's heading angle
        robot_yaw = p.getEulerFromQuaternion(robot_orientation)[2]  # Extract yaw (z-axis rotation)

        # Calculate the angle to the target
        angle_to_target = np.arctan2(target_vector[1], target_vector[0])  # Angle to the target in world frame

        # Calculate the relative angle to the target (difference between robot's heading and target angle)
        relative_angle = angle_to_target - robot_yaw
        relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))  # Normalize to [-pi, pi]

        # Calculate cosine and sine of the relative angle
        cos_angle = (np.cos(relative_angle)+1.0)/2.0
        sin_angle = (np.sin(relative_angle)+1.0)/2.0

        # Get lidar reading
        lidar_reading = np.array([0.0, 0.0], dtype=np.float32)
        while (lidar_reading == 0.0).any():
            lidar_reading = np.array(self.perform_lidar_scan(), dtype=np.float32)/3.5
            p.stepSimulation()

        # Return observation: distance, cos(angle), sin(angle)
        return_val = np.concatenate((
            lidar_reading,
            np.array([cos_angle, sin_angle, distance_to_target], dtype=np.float32)
        ))
        return return_val


if __name__ == "__main__":
    # Example usage:
    sim = Turtlebot3Simulation()  # Now connected in DIRECT mode.
    import time

    # External routines can now repeatedly call add_objects, run_simulation, and reset_environment as needed.

    reset_time = 0.0
    add_object_time = 0.0
    run_sim_time = 0.0

    for i in range(10):
        # Reset the environment for another trial.
        start_time = time.time()
        sim.reset_environment()
        reset_time += time.time() - start_time

        obstacles = [
            {"shape": "circle", "position": [random.uniform(-10.0, 10.0), random.uniform(-1.0, 1.0), 0.25], "orientation": [0, 0, 0],
             "radius": random.uniform(0.5,1.0), "height": 0.5},
            {"shape": "cuboid", "position": [random.uniform(-10.0, 10.0), random.uniform(-1.0, 1.0), 0.25], "orientation": [0, 0, 0],
             "half_extents": [random.uniform(0.5,1.0), random.uniform(0.5,1.0), 0.5]}
        ]
        start_time = time.time()
        sim.add_objects(obstacles)
        add_object_time += time.time() - start_time

        start_time = time.time()
        trajectory = sim.run_simulation(50)
        run_sim_time += time.time() - start_time

    p.disconnect()

    print(f"reset time: {reset_time}")
    print(f"add_object_time: {add_object_time}")
    print(f"run_sim_time: {run_sim_time}")
