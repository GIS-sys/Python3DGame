import pygame
import math
import sys
import time
import numpy as np
import taichi as ti

# Initialize Taichi with GPU backend
ti.init(arch=ti.gpu, default_fp=ti.f32)

# Constants
WIDTH, HEIGHT = 640, 480  # Increased resolution for GPU
FOV = math.pi / 3  # 60 degrees
MAX_DEPTH = 4  # Increased depth for better reflections
BACKGROUND_COLOR = ti.Vector([0.1, 0.1, 0.15])

# Taichi fields for GPU storage
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
accumulator = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
frame_count = ti.field(dtype=ti.i32, shape=())

# Material properties
@ti.dataclass
class Material:
    color: ti.math.vec3
    reflectivity: ti.f32
    roughness: ti.f32

# Sphere definition
@ti.dataclass
class Sphere:
    center: ti.math.vec3
    radius: ti.f32
    material_idx: ti.i32

# Cube definition
@ti.dataclass
class Cube:
    center: ti.math.vec3
    size: ti.f32
    material_idx: ti.i32

# Light definition
@ti.dataclass
class Light:
    position: ti.math.vec3
    intensity: ti.f32
    color: ti.math.vec3

# Camera definition
@ti.dataclass
class Camera:
    position: ti.math.vec3
    target: ti.math.vec3
    up: ti.math.vec3
    fov: ti.f32
    aspect_ratio: ti.f32

    @ti.func
    def get_ray(self, u: ti.f32, v: ti.f32) -> ti.math.vec3:
        # Calculate ray direction in world space
        forward = ti.math.normalize(self.target - self.position)
        right = ti.math.normalize(ti.math.cross(forward, self.up))
        up = ti.math.cross(right, forward)

        # Calculate ray direction
        half_height = ti.tan(self.fov * 0.5)
        half_width = self.aspect_ratio * half_height
        direction = forward + right * (2.0 * u - 1.0) * half_width + up * (1.0 - 2.0 * v) * half_height
        return ti.math.normalize(direction)

# Scene data stored in Taichi fields
max_spheres = 16
max_cubes = 16
max_lights = 8
max_materials = 16

spheres = Sphere.field()
ti.root.dense(ti.i, max_spheres).place(spheres)
cubes = Cube.field()
ti.root.dense(ti.i, max_cubes).place(cubes)
lights = Light.field()
ti.root.dense(ti.i, max_lights).place(lights)
materials = Material.field()
ti.root.dense(ti.i, max_materials).place(materials)

# Scene counters
num_spheres = ti.field(dtype=ti.i32, shape=())
num_cubes = ti.field(dtype=ti.i32, shape=())
num_lights = ti.field(dtype=ti.i32, shape=())
num_materials = ti.field(dtype=ti.i32, shape=())

@ti.kernel
def clear_accumulator():
    """Clear the accumulation buffer"""
    for i, j in accumulator:
        accumulator[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.func
def ray_sphere_intersect(ray_origin: ti.math.vec3, ray_dir: ti.math.vec3,
                         sphere_center: ti.math.vec3, radius: ti.f32) -> ti.f32:
    """Ray-sphere intersection with early exit if no hit"""
    oc = ray_origin - sphere_center
    a = ti.math.dot(ray_dir, ray_dir)
    b = 2.0 * ti.math.dot(oc, ray_dir)
    c = ti.math.dot(oc, oc) - radius * radius
    discriminant = b * b - 4.0 * a * c

    t = -1.0
    if discriminant >= 0:
        sqrt_disc = ti.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        if t1 > 0.001:
            t = t1
        elif t2 > 0.001:
            t = t2

    return t

@ti.func
def ray_cube_intersect(ray_origin: ti.math.vec3, ray_dir: ti.math.vec3,
                       cube_center: ti.math.vec3, size: ti.f32) -> ti.f32:
    """Ray-cube intersection using slab method"""
    half_size = size * 0.5
    min_bounds = cube_center - half_size
    max_bounds = cube_center + half_size

    tmin = (min_bounds.x - ray_origin.x) / ray_dir.x
    tmax = (max_bounds.x - ray_origin.x) / ray_dir.x

    if tmin > tmax:
        tmin, tmax = tmax, tmin

    tymin = (min_bounds.y - ray_origin.y) / ray_dir.y
    tymax = (max_bounds.y - ray_origin.y) / ray_dir.y

    if tymin > tymax:
        tymin, tymax = tymax, tymin

    if (tmin > tymax) or (tymin > tmax):
        return -1.0

    if tymin > tmin:
        tmin = tymin
    if tymax < tmax:
        tmax = tymax

    tzmin = (min_bounds.z - ray_origin.z) / ray_dir.z
    tzmax = (max_bounds.z - ray_origin.z) / ray_dir.z

    if tzmin > tzmax:
        tzmin, tzmax = tzmax, tzmin

    if (tmin > tzmax) or (tzmin > tmax):
        return -1.0

    if tzmin > tmin:
        tmin = tzmin
    if tzmax < tmax:
        tmax = tzmax

    if tmin > 0.001:
        return tmin

    return -1.0

@ti.func
def cube_normal(point: ti.math.vec3, cube_center: ti.math.vec3, size: ti.f32) -> ti.math.vec3:
    """Calculate cube face normal at intersection point"""
    half_size = size * 0.5
    eps = 0.001

    if ti.abs(point.x - (cube_center.x - half_size)) < eps:
        return ti.math.vec3(-1, 0, 0)
    elif ti.abs(point.x - (cube_center.x + half_size)) < eps:
        return ti.math.vec3(1, 0, 0)
    elif ti.abs(point.y - (cube_center.y - half_size)) < eps:
        return ti.math.vec3(0, -1, 0)
    elif ti.abs(point.y - (cube_center.y + half_size)) < eps:
        return ti.math.vec3(0, 1, 0)
    elif ti.abs(point.z - (cube_center.z - half_size)) < eps:
        return ti.math.vec3(0, 0, -1)
    else:
        return ti.math.vec3(0, 0, 1)

@ti.func
def trace_ray(ray_origin: ti.math.vec3, ray_dir: ti.math.vec3, depth: ti.i32) -> ti.math.vec3:
    """Main ray tracing function with reflections"""
    if depth <= 0:
        return BACKGROUND_COLOR

    # Find closest intersection
    closest_t = 1e10
    closest_normal = ti.math.vec3(0, 0, 1)
    closest_material_idx = -1
    is_sphere = True
    closest_center = ti.math.vec3(0, 0, 0)
    closest_size = 0.0

    # Check spheres
    for i in range(num_spheres[None]):
        sphere = spheres[i]
        t = ray_sphere_intersect(ray_origin, ray_dir, sphere.center, sphere.radius)
        if t > 0.001 and t < closest_t:
            closest_t = t
            hit_point = ray_origin + ray_dir * t
            closest_normal = ti.math.normalize(hit_point - sphere.center)
            closest_material_idx = sphere.material_idx
            is_sphere = True
            closest_center = sphere.center
            closest_size = sphere.radius

    # Check cubes
    for i in range(num_cubes[None]):
        cube = cubes[i]
        t = ray_cube_intersect(ray_origin, ray_dir, cube.center, cube.size)
        if t > 0.001 and t < closest_t:
            closest_t = t
            closest_material_idx = cube.material_idx
            is_sphere = False
            closest_center = cube.center
            closest_size = cube.size

    # If no hit, return background
    if closest_material_idx < 0:
        return BACKGROUND_COLOR

    # Calculate hit point and normal
    hit_point = ray_origin + ray_dir * closest_t
    if not is_sphere:
        closest_normal = cube_normal(hit_point, closest_center, closest_size)

    material = materials[closest_material_idx]

    # Calculate lighting
    color = ti.math.vec3(0, 0, 0)

    for i in range(num_lights[None]):
        light = lights[i]
        light_dir = ti.math.normalize(light.position - hit_point)

        # Shadow check
        in_shadow = False
        shadow_ray_origin = hit_point + light_dir * 0.001

        # Check spheres
        for j in range(num_spheres[None]):
            sphere = spheres[j]
            if ray_sphere_intersect(shadow_ray_origin, light_dir, sphere.center, sphere.radius) > 0:
                in_shadow = True
                break

        # Check cubes if not in shadow
        if not in_shadow:
            for j in range(num_cubes[None]):
                cube = cubes[j]
                if ray_cube_intersect(shadow_ray_origin, light_dir, cube.center, cube.size) > 0:
                    in_shadow = True
                    break

        if not in_shadow:
            # Diffuse lighting
            diffuse = ti.max(0.0, ti.math.dot(closest_normal, light_dir))
            # Add some ambient
            ambient = 0.1
            intensity = light.intensity * (diffuse + ambient)
            color += material.color * light.color * intensity

    # Reflection
    if material.reflectivity > 0 and depth > 1:
        reflect_dir = ti.math.reflect(ray_dir, closest_normal)
        # Add some roughness
        if material.roughness > 0:
            roughness = material.roughness * 0.5
            reflect_dir += ti.math.vec3(
                ti.random() * roughness - roughness * 0.5,
                ti.random() * roughness - roughness * 0.5,
                ti.random() * roughness - roughness * 0.5
            )
            reflect_dir = ti.math.normalize(reflect_dir)

        reflect_color = trace_ray(hit_point + reflect_dir * 0.001, reflect_dir, depth - 1)
        color = color * (1.0 - material.reflectivity) + reflect_color * material.reflectivity

    return ti.math.clamp(color, 0.0, 1.0)

@ti.kernel
def render(camera_pos: ti.math.vec3, camera_target: ti.math.vec3,
           camera_up: ti.math.vec3, frame: ti.i32):
    """Main rendering kernel that runs on GPU"""
    camera = Camera()
    camera.position = camera_pos
    camera.target = camera_target
    camera.up = camera_up
    camera.fov = FOV
    camera.aspect_ratio = WIDTH / HEIGHT

    for i, j in pixels:
        # Anti-aliasing with random sampling
        u = (i + ti.random()) / WIDTH
        v = (j + ti.random()) / HEIGHT

        ray_dir = camera.get_ray(u, v)
        color = trace_ray(camera.position, ray_dir, MAX_DEPTH)

        # Accumulate for progressive rendering
        if frame == 0:
            accumulator[i, j] = color
        else:
            # Moving average
            accumulator[i, j] = (accumulator[i, j] * frame + color) / (frame + 1)

        pixels[i, j] = accumulator[i, j]

class Scene:
    def __init__(self):
        # Initialize counters
        num_spheres[None] = 0
        num_cubes[None] = 0
        num_lights[None] = 0
        num_materials[None] = 0
        frame_count[None] = 0

        # Clear the accumulator using a kernel
        clear_accumulator()

        self.setup_scene()

    def add_material(self, color, reflectivity=0.0, roughness=0.0):
        idx = num_materials[None]
        if idx < max_materials:
            materials[idx] = Material(color=ti.math.vec3(color),
                                      reflectivity=reflectivity,
                                      roughness=roughness)
            num_materials[None] += 1
            return idx
        return -1

    def add_sphere(self, center, radius, material_idx):
        idx = num_spheres[None]
        if idx < max_spheres and material_idx >= 0:
            spheres[idx] = Sphere(center=ti.math.vec3(center),
                                  radius=radius,
                                  material_idx=material_idx)
            num_spheres[None] += 1

    def add_cube(self, center, size, material_idx):
        idx = num_cubes[None]
        if idx < max_cubes and material_idx >= 0:
            cubes[idx] = Cube(center=ti.math.vec3(center),
                              size=size,
                              material_idx=material_idx)
            num_cubes[None] += 1

    def add_light(self, position, intensity=1.0, color=(1.0, 1.0, 1.0)):
        idx = num_lights[None]
        if idx < max_lights:
            lights[idx] = Light(position=ti.math.vec3(position),
                                intensity=intensity,
                                color=ti.math.vec3(color))
            num_lights[None] += 1

    def setup_scene(self):
        # Add materials
        red_mat = self.add_material((1.0, 0.2, 0.2), reflectivity=0.3)
        blue_mat = self.add_material((0.2, 0.2, 1.0), reflectivity=0.5)
        green_mat = self.add_material((0.2, 1.0, 0.2), reflectivity=0.1)
        yellow_mat = self.add_material((1.0, 1.0, 0.2), reflectivity=0.2, roughness=0.1)
        gray_mat = self.add_material((0.7, 0.7, 0.7), reflectivity=0.05, roughness=0.3)

        # Add objects
        self.add_sphere((-2, 1, 0), 1.0, red_mat)
        self.add_sphere((2, 1, 0), 1.0, blue_mat)
        self.add_sphere((0, 1.5, -2), 1.0, green_mat)
        self.add_cube((0, -1, 0), 10.0, yellow_mat)

        # Add some extra objects for more complex scene
        self.add_sphere((-1, 0.5, 2), 0.5, gray_mat)
        self.add_cube((3, 0.5, -1), 1.0, red_mat)

        # Add lights
        self.add_light((3, 5, 3), 1.0, (1.0, 1.0, 1.0))
        self.add_light((-3, 5, -3), 0.7, (0.9, 0.9, 1.0))
        self.add_light((0, 8, 0), 0.5, (1.0, 1.0, 0.8))

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("GPU Raytracing Engine - WASD Movement")
        self.clock = pygame.time.Clock()
        self.scene = Scene()
        self.running = True

        # Camera
        self.camera_pos = ti.math.vec3(0, 2, 5)
        self.camera_target = ti.math.vec3(0, 0, 0)
        self.camera_up = ti.math.vec3(0, 1, 0)

        # Movement state
        self.keys = {
            pygame.K_w: False,
            pygame.K_s: False,
            pygame.K_a: False,
            pygame.K_d: False,
            pygame.K_q: False,
            pygame.K_e: False,
            pygame.K_LSHIFT: False,
            pygame.K_SPACE: False
        }

        # Mouse state for camera rotation
        self.mouse_pos = None
        self.mouse_sensitivity = 0.002
        self.yaw = 0.0
        self.pitch = 0.0

        # Performance tracking
        self.frame_times = []
        self.fps = 0
        self.font = pygame.font.SysFont(None, 24)

        # Camera vectors
        self.forward = ti.math.normalize(self.camera_target - self.camera_pos)
        self.right = ti.math.normalize(ti.math.cross(self.forward, self.camera_up))
        self.up = ti.math.cross(self.right, self.forward)

        print("GPU Raytracing Engine Initialized")
        print(f"Using Taichi backend: {ti.cfg.arch}")
        print(f"Resolution: {WIDTH}x{HEIGHT}")
        print("Controls:")
        print("  WASD - Movement")
        print("  Q/E - Roll left/right")
        print("  Shift/Space - Down/Up")
        print("  Mouse - Look around")
        print("  ESC - Quit")

    def update_camera_vectors(self):
        """Update camera basis vectors based on yaw and pitch"""
        self.forward = ti.math.vec3(
            math.cos(self.yaw) * math.cos(self.pitch),
            math.sin(self.pitch),
            math.sin(self.yaw) * math.cos(self.pitch)
        )
        self.forward = ti.math.normalize(self.forward)

        self.right = ti.math.normalize(ti.math.cross(self.forward, ti.math.vec3(0, 1, 0)))
        self.up = ti.math.cross(self.right, self.forward)

        self.camera_target = self.camera_pos + self.forward

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in self.keys:
                    self.keys[event.key] = True
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    # Reset accumulation on R key
                    frame_count[None] = 0
                    clear_accumulator()
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
                self.update_camera_vectors()

    def update_movement(self, dt):
        move_speed = 3.0 * dt
        rot_speed = 1.5 * dt

        # Movement
        if self.keys[pygame.K_w]:
            self.camera_pos += self.forward * move_speed
        if self.keys[pygame.K_s]:
            self.camera_pos -= self.forward * move_speed
        if self.keys[pygame.K_a]:
            self.camera_pos -= self.right * move_speed
        if self.keys[pygame.K_d]:
            self.camera_pos += self.right * move_speed
        if self.keys[pygame.K_SPACE]:
            self.camera_pos += ti.math.vec3(0, 1, 0) * move_speed
        if self.keys[pygame.K_LSHIFT]:
            self.camera_pos -= ti.math.vec3(0, 1, 0) * move_speed

        # Rotation (Q/E for roll)
        if self.keys[pygame.K_q]:
            # Roll left
            angle = rot_speed
            self.up = self.up * math.cos(angle) + ti.math.cross(self.forward, self.up) * math.sin(angle)
            self.right = ti.math.normalize(ti.math.cross(self.forward, self.up))
        if self.keys[pygame.K_e]:
            # Roll right
            angle = -rot_speed
            self.up = self.up * math.cos(angle) + ti.math.cross(self.forward, self.up) * math.sin(angle)
            self.right = ti.math.normalize(ti.math.cross(self.forward, self.up))

        self.camera_target = self.camera_pos + self.forward

    def render_ui(self, render_time):
        # Convert pixel data to pygame surface
        pixel_array = pixels.to_numpy()
        pixel_array = np.swapaxes(pixel_array, 0, 1)
        surf = pygame.surfarray.make_surface(pixel_array * 255)
        self.screen.blit(surf, (0, 0))

        # Calculate FPS
        self.frame_times.append(render_time)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)

        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        self.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        # Render UI text
        texts = [
            f"FPS: {self.fps:.1f}",
            f"Frame Time: {render_time*1000:.1f}ms",
            f"Samples: {frame_count[None] + 1}",
            "WASD: Move  Shift/Space: Up/Down",
            "Q/E: Roll  Mouse: Look  R: Reset",
            f"Resolution: {WIDTH}x{HEIGHT}",
            f"GPU: {ti.cfg.arch}"
        ]

        for i, text in enumerate(texts):
            text_surf = self.font.render(text, True, (255, 255, 255))
            # Add background for readability
            bg_rect = text_surf.get_rect()
            bg_rect.x = 10
            bg_rect.y = 10 + i * 25
            pygame.draw.rect(self.screen, (0, 0, 0, 128), bg_rect.inflate(10, 5))
            self.screen.blit(text_surf, (15, 12 + i * 25))

    def run(self):
        last_time = time.time()

        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            self.handle_events()
            self.update_movement(dt)

            # Render frame on GPU
            render_start = time.time()

            # Increment frame count for accumulation
            current_frame = frame_count[None]

            # Launch GPU kernel
            render(self.camera_pos, self.camera_target, self.camera_up, current_frame)
            ti.sync()  # Wait for GPU to finish

            frame_count[None] = current_frame + 1
            render_time = time.time() - render_start

            # Display the result
            self.render_ui(render_time)
            pygame.display.flip()

            # Cap at 60 FPS
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()

