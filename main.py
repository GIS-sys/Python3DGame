import pygame
import numpy as np
import math
import sys
import multiprocessing as mp
from multiprocessing import Pool, shared_memory
from typing import List, Tuple, Optional
import time

# Constants
WIDTH, HEIGHT = 320, 240  # Reduced for better performance
FOV = math.pi / 3  # 60 degrees
MAX_DEPTH = 2  # Reduced for better performance
NUM_CORES = max(1, mp.cpu_count() - 1)  # Leave one core free

# Color constants
BACKGROUND = np.array([0.1, 0.1, 0.15])
WHITE = np.array([1.0, 1.0, 1.0])
RED = np.array([1.0, 0.2, 0.2])
GREEN = np.array([0.2, 1.0, 0.2])
BLUE = np.array([0.2, 0.2, 1.0])
YELLOW = np.array([1.0, 1.0, 0.2])

class Vector3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self):
        l = self.length()
        if l > 0:
            return self / l
        return self

    def to_array(self):
        return np.array([self.x, self.y, self.z])

    def to_tuple(self):
        return (self.x, self.y, self.z)

    @staticmethod
    def from_tuple(t):
        return Vector3(t[0], t[1], t[2])

class Ray:
    def __init__(self, origin: Vector3, direction: Vector3):
        self.origin = origin
        self.direction = direction.normalize()

    def to_tuple(self):
        return (self.origin.to_tuple(), self.direction.to_tuple())

    @staticmethod
    def from_tuple(t):
        return Ray(Vector3.from_tuple(t[0]), Vector3.from_tuple(t[1]))

class Material:
    def __init__(self, color: np.ndarray, reflectivity: float = 0.0):
        self.color = color
        self.reflectivity = reflectivity

    def to_tuple(self):
        return (tuple(self.color.tolist()), self.reflectivity)

    @staticmethod
    def from_tuple(t):
        return Material(np.array(t[0]), t[1])

class Sphere:
    def __init__(self, center: Vector3, radius: float, material: Material):
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray: Ray) -> Tuple[Optional[float], Optional[Vector3]]:
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None, None

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        if t1 > 0.001:
            t = t1
        elif t2 > 0.001:
            t = t2
        else:
            return None, None

        hit_point = ray.origin + ray.direction * t
        normal = (hit_point - self.center).normalize()
        return t, normal

    def to_tuple(self):
        return (self.center.to_tuple(), self.radius, self.material.to_tuple())

    @staticmethod
    def from_tuple(t):
        return Sphere(Vector3.from_tuple(t[0]), t[1], Material.from_tuple(t[2]))

class Cube:
    def __init__(self, center: Vector3, size: float, material: Material):
        self.center = center
        self.size = size
        self.material = material
        self.min = Vector3(center.x - size/2, center.y - size/2, center.z - size/2)
        self.max = Vector3(center.x + size/2, center.y + size/2, center.z + size/2)

    def intersect(self, ray: Ray) -> Tuple[Optional[float], Optional[Vector3]]:
        tmin = (self.min.x - ray.origin.x) / ray.direction.x
        tmax = (self.max.x - ray.origin.x) / ray.direction.x

        if tmin > tmax:
            tmin, tmax = tmax, tmin

        tymin = (self.min.y - ray.origin.y) / ray.direction.y
        tymax = (self.max.y - ray.origin.y) / ray.direction.y

        if tymin > tymax:
            tymin, tymax = tymax, tymin

        if (tmin > tymax) or (tymin > tmax):
            return None, None

        if tymin > tmin:
            tmin = tymin
        if tymax < tmax:
            tmax = tymax

        tzmin = (self.min.z - ray.origin.z) / ray.direction.z
        tzmax = (self.max.z - ray.origin.z) / ray.direction.z

        if tzmin > tzmax:
            tzmin, tzmax = tzmax, tzmin

        if (tmin > tzmax) or (tzmin > tmax):
            return None, None

        if tzmin > tmin:
            tmin = tzmin
        if tzmax < tmax:
            tmax = tzmax

        if tmin > 0.001:
            t = tmin
            hit_point = ray.origin + ray.direction * t
            normal = self.get_normal(hit_point)
            return t, normal

        return None, None

    def get_normal(self, point: Vector3) -> Vector3:
        eps = 0.001
        if abs(point.x - self.min.x) < eps:
            return Vector3(-1, 0, 0)
        elif abs(point.x - self.max.x) < eps:
            return Vector3(1, 0, 0)
        elif abs(point.y - self.min.y) < eps:
            return Vector3(0, -1, 0)
        elif abs(point.y - self.max.y) < eps:
            return Vector3(0, 1, 0)
        elif abs(point.z - self.min.z) < eps:
            return Vector3(0, 0, -1)
        else:
            return Vector3(0, 0, 1)

    def to_tuple(self):
        return (self.center.to_tuple(), self.size, self.material.to_tuple())

    @staticmethod
    def from_tuple(t):
        return Cube(Vector3.from_tuple(t[0]), t[1], Material.from_tuple(t[2]))

class Light:
    def __init__(self, position: Vector3, intensity: float):
        self.position = position
        self.intensity = intensity

    def to_tuple(self):
        return (self.position.to_tuple(), self.intensity)

    @staticmethod
    def from_tuple(t):
        return Light(Vector3.from_tuple(t[0]), t[1])

class Camera:
    def __init__(self, position: Vector3, target: Vector3, up: Vector3):
        self.position = position
        self.target = target
        self.up = up
        self.update_vectors()

    def update_vectors(self):
        self.forward = (self.target - self.position).normalize()
        self.right = self.forward.cross(self.up).normalize()
        self.up = self.right.cross(self.forward).normalize()

    def move(self, dx: float, dy: float, dz: float):
        self.position += self.right * dx
        self.position += self.up * dy
        self.position += self.forward * dz
        self.target += self.right * dx
        self.target += self.up * dy
        self.target += self.forward * dz

    def rotate(self, yaw: float, pitch: float):
        # Update forward vector
        self.forward = self.forward * math.cos(yaw) + self.right * math.sin(yaw)
        self.forward = self.forward.normalize()

        # Update right vector
        self.right = self.forward.cross(Vector3(0, 1, 0)).normalize()

        # Update up vector
        self.up = self.right.cross(self.forward).normalize()

        # Update target
        self.target = self.position + self.forward

    def to_tuple(self):
        return (self.position.to_tuple(), self.target.to_tuple(), self.up.to_tuple())

    @staticmethod
    def from_tuple(t):
        cam = Camera(Vector3.from_tuple(t[0]), Vector3.from_tuple(t[1]), Vector3.from_tuple(t[2]))
        cam.update_vectors()
        return cam

class SceneData:
    """Simple data container that can be pickled"""
    def __init__(self, objects_data, lights_data):
        self.objects_data = objects_data
        self.lights_data = lights_data

def trace_ray(ray_tuple, scene_data, camera_tuple, max_depth, width, height):
    """Worker function for parallel rendering"""
    # Reconstruct objects
    ray = Ray.from_tuple(ray_tuple)
    camera = Camera.from_tuple(camera_tuple)

    # Create scene objects from serialized data
    objects = []
    for obj_data in scene_data.objects_data:
        if len(obj_data) == 3:  # Sphere
            objects.append(Sphere.from_tuple(obj_data))
        else:  # Cube
            objects.append(Cube.from_tuple(obj_data))

    lights = []
    for light_data in scene_data.lights_data:
        lights.append(Light.from_tuple(light_data))

    # Trace ray
    if max_depth <= 0:
        return BACKGROUND

    closest_t = float('inf')
    closest_obj = None
    closest_normal = None

    for obj in objects:
        t, normal = obj.intersect(ray)
        if t and t < closest_t:
            closest_t = t
            closest_obj = obj
            closest_normal = normal

    if closest_obj is None:
        return BACKGROUND

    # Calculate hit point
    hit_point = ray.origin + ray.direction * closest_t

    # Calculate lighting
    color = np.zeros(3)
    for light in lights:
        # Calculate light direction
        light_dir = (light.position - hit_point).normalize()

        # Check for shadows
        shadow_ray = Ray(hit_point + light_dir * 0.001, light_dir)
        in_shadow = False
        for obj in objects:
            t, _ = obj.intersect(shadow_ray)
            if t and t < (light.position - hit_point).length():
                in_shadow = True
                break

        if not in_shadow:
            # Diffuse lighting
            diffuse = max(0, closest_normal.dot(light_dir))
            color += closest_obj.material.color * diffuse * light.intensity

    # Reflection
    if closest_obj.material.reflectivity > 0:
        reflect_dir = ray.direction - closest_normal * 2 * ray.direction.dot(closest_normal)
        reflect_ray = Ray(hit_point + reflect_dir * 0.001, reflect_dir)
        # Create new scene data for recursion
        new_scene_data = SceneData(scene_data.objects_data, scene_data.lights_data)
        reflect_color = trace_ray(reflect_ray.to_tuple(), new_scene_data, camera_tuple, max_depth - 1, width, height)
        color = color * (1 - closest_obj.material.reflectivity) + reflect_color * closest_obj.material.reflectivity

    return np.clip(color, 0, 1)

class Renderer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.scene_objects = []
        self.scene_lights = []
        self.camera = Camera(
            Vector3(0, 2, 5),
            Vector3(0, 0, 0),
            Vector3(0, 1, 0)
        )
        self.setup_scene()

    def setup_scene(self):
        # Add objects
        self.scene_objects.append(Sphere(Vector3(-2, 1, 0), 1, Material(RED, 0.3)))
        self.scene_objects.append(Sphere(Vector3(2, 1, 0), 1, Material(BLUE, 0.5)))
        self.scene_objects.append(Sphere(Vector3(0, 1, -2), 1, Material(GREEN, 0.1)))
        self.scene_objects.append(Cube(Vector3(0, -1, 0), 10, Material(YELLOW, 0.2)))

        # Add lights
        self.scene_lights.append(Light(Vector3(3, 5, 3), 1.0))
        self.scene_lights.append(Light(Vector3(-3, 5, -3), 0.7))

    def render_row(self, args):
        """Render a single row - used by parallel workers"""
        y, scene_data, camera_tuple = args
        aspect_ratio = self.width / self.height
        row_colors = []

        for x in range(self.width):
            # Calculate ray direction from camera
            px = (2 * (x + 0.5) / self.width - 1) * math.tan(FOV / 2) * aspect_ratio
            py = (1 - 2 * (y + 0.5) / self.height) * math.tan(FOV / 2)

            # Reconstruct camera
            camera = Camera.from_tuple(camera_tuple)
            ray_dir = (
                camera.forward +
                camera.right * px +
                camera.up * py
            ).normalize()

            ray = Ray(camera.position, ray_dir)

            # Trace ray
            color = trace_ray(ray.to_tuple(), scene_data, camera_tuple, MAX_DEPTH, self.width, self.height)
            row_colors.append((x, color))

        return y, row_colors

    def render_frame_parallel(self):
        """Render frame using multiprocessing"""
        start_time = time.time()

        # Prepare scene data for workers
        objects_data = []
        for obj in self.scene_objects:
            objects_data.append(obj.to_tuple())

        lights_data = []
        for light in self.scene_lights:
            lights_data.append(light.to_tuple())

        scene_data = SceneData(objects_data, lights_data)

        # Prepare arguments for each row
        args_list = [(y, scene_data, self.camera.to_tuple()) for y in range(self.height)]

        # Render in parallel
        pixels = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # Use smaller chunks for better load balancing
        with Pool(processes=NUM_CORES) as pool:
            results = pool.map(self.render_row, args_list, chunksize=4)

            # Collect results
            for y, row_colors in results:
                for x, color in row_colors:
                    pixels[y, x] = color

        render_time = time.time() - start_time
        fps = 1.0 / render_time if render_time > 0 else 0
        return render_time, fps, pixels

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("3D Raytracing Engine - WASD Movement")
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(WIDTH, HEIGHT)
        self.running = True
        self.font = pygame.font.SysFont(None, 24)

        # Movement state
        self.keys = {
            pygame.K_w: False,
            pygame.K_s: False,
            pygame.K_a: False,
            pygame.K_d: False,
            pygame.K_q: False,
            pygame.K_e: False
        }

        # Mouse state for camera rotation
        self.mouse_pos = None
        self.mouse_sensitivity = 0.002
        self.yaw = 0
        self.pitch = 0

        # Performance tracking
        self.last_render_time = 0
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in self.keys:
                    self.keys[event.key] = True
                if event.key == pygame.K_ESCAPE:
                    self.running = False
            elif event.type == pygame.KEYUP:
                if event.key in self.keys:
                    self.keys[event.key] = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    pygame.mouse.set_visible(False)
                    pygame.event.set_grab(True)
                    self.mouse_pos = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    pygame.mouse.set_visible(True)
                    pygame.event.set_grab(False)
                    self.mouse_pos = None
            elif event.type == pygame.MOUSEMOTION and self.mouse_pos is not None:
                dx, dy = event.rel
                self.yaw -= dx * self.mouse_sensitivity
                self.pitch = max(-math.pi/2.1, min(math.pi/2.1,
                    self.pitch - dy * self.mouse_sensitivity))
                self.renderer.camera.rotate(self.yaw, self.pitch)

    def update_movement(self):
        move_speed = 0.1

        if self.keys[pygame.K_w]:
            self.renderer.camera.move(0, 0, -move_speed)
        if self.keys[pygame.K_s]:
            self.renderer.camera.move(0, 0, move_speed)
        if self.keys[pygame.K_a]:
            self.renderer.camera.move(-move_speed, 0, 0)
        if self.keys[pygame.K_d]:
            self.renderer.camera.move(move_speed, 0, 0)
        if self.keys[pygame.K_q]:
            self.renderer.camera.move(0, -move_speed, 0)
        if self.keys[pygame.K_e]:
            self.renderer.camera.move(0, move_speed, 0)

    def render_ui(self, render_time, fps, pixels):
        # Convert pixel data to pygame surface
        pixel_data = (pixels * 255).astype(np.uint8)
        surf = pygame.surfarray.make_surface(pixel_data.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))

        # Update FPS counter
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time

        # Render UI text
        texts = [
            f"FPS: {self.fps:.1f}",
            f"Render Time: {render_time*1000:.1f}ms",
            "WASD: Move  Q/E: Up/Down",
            "Mouse: Look  ESC: Quit",
            f"CPU Cores: {NUM_CORES}",
            f"Resolution: {WIDTH}x{HEIGHT}"
        ]

        for i, text in enumerate(texts):
            text_surf = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surf, (10, 10 + i * 25))

    def run(self):
        # Warm up the pool
        print("Initializing renderer...")

        while self.running:
            self.handle_events()
            self.update_movement()

            # Render the scene
            try:
                render_time, fps, pixels = self.renderer.render_frame_parallel()
                self.last_render_time = render_time
            except Exception as e:
                print(f"Render error: {e}")
                self.running = False
                break

            # Display the result
            self.render_ui(render_time, fps, pixels)
            pygame.display.flip()

            # Cap at 30 FPS for display (raytracing is CPU intensive)
            self.clock.tick(30)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    # Set multiprocessing start method
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    print(f"Starting 3D Raytracing Engine...")
    print(f"Using {NUM_CORES} CPU cores")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print("Controls:")
    print("  WASD - Movement")
    print("  Q/E - Up/Down")
    print("  Mouse - Look around")
    print("  ESC - Quit")

    game = Game()
    game.run()

