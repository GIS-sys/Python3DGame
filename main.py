import pygame
import numpy as np
import math
import sys
import multiprocessing as mp
from multiprocessing import Pool, shared_memory
import ctypes
from typing import List, Tuple, Optional
import time

# Constants
WIDTH, HEIGHT = 800, 600
FOV = math.pi / 3  # 60 degrees
MAX_DEPTH = 3
NUM_CORES = mp.cpu_count()

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

class Ray:
    def __init__(self, origin: Vector3, direction: Vector3):
        self.origin = origin
        self.direction = direction.normalize()

class Material:
    def __init__(self, color: np.ndarray, reflectivity: float = 0.0):
        self.color = color
        self.reflectivity = reflectivity

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

class Light:
    def __init__(self, position: Vector3, intensity: float):
        self.position = position
        self.intensity = intensity

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

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []

    def add_object(self, obj):
        self.objects.append(obj)

    def add_light(self, light):
        self.lights.append(light)

    def trace_ray(self, ray: Ray, depth: int = 0) -> np.ndarray:
        if depth >= MAX_DEPTH:
            return BACKGROUND

        closest_t = float('inf')
        closest_obj = None
        closest_normal = None

        for obj in self.objects:
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
        for light in self.lights:
            # Calculate light direction
            light_dir = (light.position - hit_point).normalize()

            # Check for shadows
            shadow_ray = Ray(hit_point + light_dir * 0.001, light_dir)
            in_shadow = False
            for obj in self.objects:
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
            reflect_color = self.trace_ray(reflect_ray, depth + 1)
            color = color * (1 - closest_obj.material.reflectivity) + reflect_color * closest_obj.material.reflectivity

        return np.clip(color, 0, 1)

class Renderer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.scene = Scene()
        self.camera = Camera(
            Vector3(0, 2, 5),
            Vector3(0, 0, 0),
            Vector3(0, 1, 0)
        )
        self.setup_scene()

        # Create shared memory for pixel data
        self.shared_array = mp.Array(ctypes.c_float, width * height * 3)
        self.pixels = np.frombuffer(self.shared_array.get_obj(), dtype=np.float32).reshape(height, width, 3)

    def setup_scene(self):
        # Add objects
        self.scene.add_object(Sphere(Vector3(-2, 1, 0), 1, Material(RED, 0.3)))
        self.scene.add_object(Sphere(Vector3(2, 1, 0), 1, Material(BLUE, 0.5)))
        self.scene.add_object(Sphere(Vector3(0, 1, -2), 1, Material(GREEN, 0.1)))
        self.scene.add_object(Cube(Vector3(0, -1, 0), 10, Material(YELLOW, 0.2)))

        # Add lights
        self.scene.add_light(Light(Vector3(3, 5, 3), 1.0))
        self.scene.add_light(Light(Vector3(-3, 5, -3), 0.7))

    def render_row(self, y):
        aspect_ratio = self.width / self.height
        row_pixels = []

        for x in range(self.width):
            # Calculate ray direction
            px = (2 * (x + 0.5) / self.width - 1) * math.tan(FOV / 2) * aspect_ratio
            py = (1 - 2 * (y + 0.5) / self.height) * math.tan(FOV / 2)

            ray_dir = (
                self.camera.forward +
                self.camera.right * px +
                self.camera.up * py
            ).normalize()

            ray = Ray(self.camera.position, ray_dir)
            color = self.scene.trace_ray(ray)
            row_pixels.append((x, y, color))

        return row_pixels

    def render_frame_parallel(self):
        start_time = time.time()

        # Use multiprocessing to render rows in parallel
        with Pool(processes=NUM_CORES) as pool:
            results = pool.map(self.render_row, range(self.height))

            # Collect results into pixel buffer
            for row in results:
                for x, y, color in row:
                    self.pixels[y, x] = color

        render_time = time.time() - start_time
        fps = 1.0 / render_time if render_time > 0 else 0
        return render_time, fps

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

    def render_ui(self, render_time, fps):
        # Convert pixel data to pygame surface
        pixel_data = (self.renderer.pixels * 255).astype(np.uint8)
        surf = pygame.surfarray.make_surface(pixel_data.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))

        # Render UI text
        texts = [
            f"FPS: {fps:.1f}",
            f"Render Time: {render_time*1000:.1f}ms",
            "WASD: Move  Q/E: Up/Down",
            "Mouse: Look  ESC: Quit",
            f"CPU Cores: {NUM_CORES}"
        ]

        for i, text in enumerate(texts):
            text_surf = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surf, (10, 10 + i * 25))

    def run(self):
        while self.running:
            self.handle_events()
            self.update_movement()

            # Render the scene
            render_time, fps = self.renderer.render_frame_parallel()

            # Display the result
            self.render_ui(render_time, fps)
            pygame.display.flip()

            # Cap at 60 FPS for display
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)

    game = Game()
    game.run()

