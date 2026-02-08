import math

class Camera:
    """Manages camera position and orientation"""

    def __init__(self):
        self.position = [-10.0, 2.0, 0.0]
        self.target = [-9.0, 2.0, 0.0]
        self.up = [0.0, 1.0, 0.0]
        self.yaw = 0.0
        self.pitch = 0.0
        self.mouse_sensitivity = 0.002

    def update_vectors(self):
        """Update camera basis vectors based on yaw and pitch"""
        # Calculate forward vector from yaw and pitch
        forward_x = math.cos(self.yaw) * math.cos(self.pitch)
        forward_y = math.sin(self.pitch)
        forward_z = math.sin(self.yaw) * math.cos(self.pitch)

        # Normalize forward vector
        length = math.sqrt(forward_x*forward_x + forward_y*forward_y + forward_z*forward_z)
        if length > 0:
            forward_x /= length
            forward_y /= length
            forward_z /= length

        # Calculate right vector (cross with world up)
        right_x = forward_z
        right_y = 0.0
        right_z = -forward_x

        # Normalize right vector
        length = math.sqrt(right_x*right_x + right_y*right_y + right_z*right_z)
        if length > 0:
            right_x /= length
            right_y /= length
            right_z /= length

        # Calculate up vector (cross right with forward)
        up_x = right_y * forward_z - right_z * forward_y
        up_y = right_z * forward_x - right_x * forward_z
        up_z = right_x * forward_y - right_y * forward_x

        # Normalize up vector
        length = math.sqrt(up_x*up_x + up_y*up_y + up_z*up_z)
        if length > 0:
            up_x /= length
            up_y /= length
            up_z /= length

        # Update camera target based on new forward direction
        self.target = [
            self.position[0] + forward_x,
            self.position[1] + forward_y,
            self.position[2] + forward_z
        ]

        # Update camera up vector
        self.up = [up_x, up_y, up_z]

    def move_forward(self, amount):
        """Move camera forward"""
        dir_forward = [self.target[i] - self.position[i] for i in range(3)]
        length = math.sqrt(sum(d * d for d in dir_forward))
        if length > 0:
            dir_forward = [d / length for d in dir_forward]
            self.position = [self.position[i] + dir_forward[i] * amount for i in range(3)]
            self.target = [self.target[i] + dir_forward[i] * amount for i in range(3)]

    def move_backward(self, amount):
        """Move camera backward"""
        dir_forward = [self.target[i] - self.position[i] for i in range(3)]
        length = math.sqrt(sum(d * d for d in dir_forward))
        if length > 0:
            dir_forward = [d / length for d in dir_forward]
            self.position = [self.position[i] - dir_forward[i] * amount for i in range(3)]
            self.target = [self.target[i] - dir_forward[i] * amount for i in range(3)]

    def strafe_left(self, amount):
        """Strafe camera left"""
        dir_forward = [self.target[i] - self.position[i] for i in range(3)]
        world_up = [0.0, 1.0, 0.0]
        dir_right = [
            dir_forward[1] * world_up[2] - dir_forward[2] * world_up[1],
            dir_forward[2] * world_up[0] - dir_forward[0] * world_up[2],
            dir_forward[0] * world_up[1] - dir_forward[1] * world_up[0]
        ]
        length = math.sqrt(sum(d * d for d in dir_right))
        if length > 0:
            dir_right = [d / length for d in dir_right]
            self.position = [self.position[i] + dir_right[i] * amount for i in range(3)]
            self.target = [self.target[i] + dir_right[i] * amount for i in range(3)]

    def strafe_right(self, amount):
        """Strafe camera right"""
        dir_forward = [self.target[i] - self.position[i] for i in range(3)]
        world_up = [0.0, 1.0, 0.0]
        dir_right = [
            dir_forward[1] * world_up[2] - dir_forward[2] * world_up[1],
            dir_forward[2] * world_up[0] - dir_forward[0] * world_up[2],
            dir_forward[0] * world_up[1] - dir_forward[1] * world_up[0]
        ]
        length = math.sqrt(sum(d * d for d in dir_right))
        if length > 0:
            dir_right = [d / length for d in dir_right]
            self.position = [self.position[i] - dir_right[i] * amount for i in range(3)]
            self.target = [self.target[i] - dir_right[i] * amount for i in range(3)]

    def move_up(self, amount):
        """Move camera up"""
        self.position[1] += amount
        self.target[1] += amount

    def move_down(self, amount):
        """Move camera down"""
        self.position[1] -= amount
        self.target[1] -= amount

    def rotate(self, dx, dy):
        """Rotate camera with mouse input"""
        self.yaw -= dx * self.mouse_sensitivity
        self.pitch = max(-math.pi/2.1, min(math.pi/2.1,
            self.pitch - dy * self.mouse_sensitivity))
        self.update_vectors()

