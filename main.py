import pygame
import math
import sys
import time
import numpy as np
import taichi as ti

# Initialize Taichi with GPU backend
ti.init(arch=ti.gpu, default_fp=ti.f32)

# Constants
WIDTH, HEIGHT = 640, 480
FOV = math.pi / 3  # 60 degrees
MAX_DEPTH = 4
BACKGROUND_COLOR = ti.Vector([0.1, 0.1, 0.15])

# Scene limits
MAX_SPHERES = 16
MAX_CUBES = 16
MAX_LIGHTS = 8
MAX_MATERIALS = 16

# ==================== GPU DATA STRUCTURES ====================
@ti.dataclass
class Material:
    color: ti.math.vec3
    reflectivity: ti.f32
    roughness: ti.f32

@ti.dataclass
class Sphere:
    center: ti.math.vec3
    radius: ti.f32
    material_idx: ti.i32

@ti.dataclass
class Cube:
    center: ti.math.vec3
    size: ti.f32
    material_idx: ti.i32

@ti.dataclass
class Light:
    position: ti.math.vec3
    intensity: ti.f32
    color: ti.math.vec3

# GPU storage
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
accumulator = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
frame_count = ti.field(dtype=ti.i32, shape=())

spheres = Sphere.field()
ti.root.dense(ti.i, MAX_SPHERES).place(spheres)
cubes = Cube.field()
ti.root.dense(ti.i, MAX_CUBES).place(cubes)
lights = Light.field()
ti.root.dense(ti.i, MAX_LIGHTS).place(lights)
materials = Material.field()
ti.root.dense(ti.i, MAX_MATERIALS).place(materials)

# Scene counters
num_spheres = ti.field(dtype=ti.i32, shape=())
num_cubes = ti.field(dtype=ti.i32, shape=())
num_lights = ti.field(dtype=ti.i32, shape=())
num_materials = ti.field(dtype=ti.i32, shape=())

# ==================== GPU KERNELS ====================
@ti.kernel
def clear_accumulator():
    """Clear the accumulation buffer"""
    for i, j in accumulator:
        accumulator[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.func
def normalize_vec(v: ti.math.vec3) -> ti.math.vec3:
    """Safe vector normalization without conditional returns"""
    length = ti.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    result = ti.math.vec3(0.0, 0.0, 1.0)  # Default value
    if length > 0.0:
        result = v / length
    return result

@ti.func
def ray_sphere_intersect(ray_origin: ti.math.vec3, ray_dir: ti.math.vec3,
                         sphere_center: ti.math.vec3, radius: ti.f32) -> ti.f32:
    """Ray-sphere intersection"""
    oc = ray_origin - sphere_center
    a = ti.math.dot(ray_dir, ray_dir)
    b = 2.0 * ti.math.dot(oc, ray_dir)
    c = ti.math.dot(oc, oc) - radius * radius
    discriminant = b * b - 4.0 * a * c

    t = -1.0  # Default value
    if discriminant >= 0:
        sqrt_disc = ti.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        # Choose the closest valid intersection
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

    t = -1.0  # Default return value

    # Check X slab
    tmin = (min_bounds.x - ray_origin.x) / ray_dir.x
    tmax = (max_bounds.x - ray_origin.x) / ray_dir.x

    if tmin > tmax:
        tmin, tmax = tmax, tmin

    # Check Y slab
    tymin = (min_bounds.y - ray_origin.y) / ray_dir.y
    tymax = (max_bounds.y - ray_origin.y) / ray_dir.y

    if tymin > tymax:
        tymin, tymax = tymax, tymin

    if not (tmin > tymax or tymin > tmax):
        # Update tmin/tmax with Y slab
        if tymin > tmin:
            tmin = tymin
        if tymax < tmax:
            tmax = tymax

        # Check Z slab
        tzmin = (min_bounds.z - ray_origin.z) / ray_dir.z
        tzmax = (max_bounds.z - ray_origin.z) / ray_dir.z

        if tzmin > tzmax:
            tzmin, tzmax = tzmax, tzmin

        if not (tmin > tzmax or tzmin > tmax):
            # Update tmin/tmax with Z slab
            if tzmin > tmin:
                tmin = tzmin
            if tzmax < tmax:
                tmax = tzmax

            # Return the closest valid intersection
            if tmin > 0.001:
                t = tmin

    return t

@ti.func
def cube_normal(point: ti.math.vec3, cube_center: ti.math.vec3, size: ti.f32) -> ti.math.vec3:
    """Calculate cube face normal at intersection point"""
    half_size = size * 0.5
    eps = 0.001

    # Initialize with default normal
    normal = ti.math.vec3(0, 0, 1)

    if ti.abs(point.x - (cube_center.x - half_size)) < eps:
        normal = ti.math.vec3(-1, 0, 0)
    elif ti.abs(point.x - (cube_center.x + half_size)) < eps:
        normal = ti.math.vec3(1, 0, 0)
    elif ti.abs(point.y - (cube_center.y - half_size)) < eps:
        normal = ti.math.vec3(0, -1, 0)
    elif ti.abs(point.y - (cube_center.y + half_size)) < eps:
        normal = ti.math.vec3(0, 1, 0)
    elif ti.abs(point.z - (cube_center.z - half_size)) < eps:
        normal = ti.math.vec3(0, 0, -1)

    return normal

@ti.func
def trace_ray(ray_origin: ti.math.vec3, ray_dir: ti.math.vec3) -> ti.math.vec3:
    """Main ray tracing function with iterative reflections (no recursion)"""
    color = ti.math.vec3(0, 0, 0)
    ray_weight = ti.math.vec3(1, 1, 1)  # How much this ray contributes
    current_origin = ray_origin
    current_dir = ray_dir

    # Iterate up to MAX_DEPTH bounces
    for bounce in range(MAX_DEPTH):
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
            t = ray_sphere_intersect(current_origin, current_dir, sphere.center, sphere.radius)
            if t > 0.001 and t < closest_t:
                closest_t = t
                hit_point = current_origin + current_dir * t
                closest_normal = normalize_vec(hit_point - sphere.center)
                closest_material_idx = sphere.material_idx
                is_sphere = True
                closest_center = sphere.center
                closest_size = sphere.radius

        # Check cubes
        for i in range(num_cubes[None]):
            cube = cubes[i]
            t = ray_cube_intersect(current_origin, current_dir, cube.center, cube.size)
            if t > 0.001 and t < closest_t:
                closest_t = t
                closest_material_idx = cube.material_idx
                is_sphere = False
                closest_center = cube.center
                closest_size = cube.size

        # If no hit, add background and break
        if closest_material_idx < 0:
            color += BACKGROUND_COLOR * ray_weight
            break

        # Calculate hit point
        hit_point = current_origin + current_dir * closest_t

        # Get normal for cube if needed
        cube_normal_vec = ti.math.vec3(0, 0, 1)
        if not is_sphere:
            cube_normal_vec = cube_normal(hit_point, closest_center, closest_size)

        # Use sphere normal or cube normal
        normal_to_use = closest_normal if is_sphere else cube_normal_vec

        material = materials[closest_material_idx]

        # Calculate lighting for this bounce
        bounce_color = ti.math.vec3(0, 0, 0)

        # Ambient lighting
        ambient = 0.1
        bounce_color += material.color * ambient

        # Process each light
        for i in range(num_lights[None]):
            light = lights[i]
            light_dir_vec = light.position - hit_point
            light_dir = normalize_vec(light_dir_vec)

            # Shadow check
            in_shadow = False
            shadow_ray_origin = hit_point + light_dir * 0.001

            # Check spheres for shadow
            for j in range(num_spheres[None]):
                sphere = spheres[j]
                shadow_t = ray_sphere_intersect(shadow_ray_origin, light_dir, sphere.center, sphere.radius)
                if shadow_t > 0:
                    in_shadow = True
                    break

            # Check cubes for shadow if not in shadow
            if not in_shadow:
                for j in range(num_cubes[None]):
                    cube = cubes[j]
                    shadow_t = ray_cube_intersect(shadow_ray_origin, light_dir, cube.center, cube.size)
                    if shadow_t > 0:
                        in_shadow = True
                        break

            # Add diffuse lighting if not in shadow
            if not in_shadow:
                diffuse = ti.max(0.0, ti.math.dot(normal_to_use, light_dir))
                bounce_color += material.color * light.color * diffuse * light.intensity

        # Add this bounce's contribution
        color += bounce_color * ray_weight

        # Check if we should continue with reflection
        if material.reflectivity > 0.0:
            # Calculate reflection direction
            reflect_dir = ti.math.reflect(current_dir, normal_to_use)

            # Add roughness if present
            if material.roughness > 0.0:
                roughness = material.roughness * 0.5
                reflect_dir = reflect_dir + ti.math.vec3(
                    (ti.random() * 2.0 - 1.0) * roughness,
                    (ti.random() * 2.0 - 1.0) * roughness,
                    (ti.random() * 2.0 - 1.0) * roughness
                )
                reflect_dir = normalize_vec(reflect_dir)

            # Update for next iteration
            current_origin = hit_point + reflect_dir * 0.001
            current_dir = reflect_dir

            # Reduce ray weight by reflectivity
            ray_weight = ray_weight * material.reflectivity

            # If ray weight is too small, stop
            if ray_weight.x < 0.01 and ray_weight.y < 0.01 and ray_weight.z < 0.01:
                break
        else:
            # No reflection, stop
            break

    return ti.math.clamp(color, 0.0, 1.0)

@ti.kernel
def render(camera_pos: ti.math.vec3, camera_target: ti.math.vec3,
           camera_up: ti.math.vec3, frame: ti.i32):
    """Main rendering kernel that runs on GPU"""
    # Calculate camera basis vectors
    forward = normalize_vec(camera_target - camera_pos)
    right = normalize_vec(ti.math.cross(forward, camera_up))
    up = ti.math.cross(right, forward)

    # Calculate view parameters
    half_height = ti.tan(FOV * 0.5)
    half_width = (WIDTH / HEIGHT) * half_height

    for i, j in pixels:
        # Calculate ray direction with anti-aliasing
        u = ((i + ti.random()) / WIDTH - 0.5) * 2.0
        v = ((j + ti.random()) / HEIGHT - 0.5) * 2.0

        ray_dir = forward + right * u * half_width + up * v * half_height
        ray_dir = normalize_vec(ray_dir)

        # Trace the ray (iterative, no recursion)
        color = trace_ray(camera_pos, ray_dir)

        # Accumulate for progressive rendering
        if frame == 0:
            accumulator[i, j] = color
        else:
            accumulator[i, j] = (accumulator[i, j] * frame + color) / (frame + 1)

        pixels[i, j] = accumulator[i, j]

# ==================== PYTHON CLASSES ====================
class Scene:
    """Python-side scene manager"""
    def __init__(self):
        # Reset counters
        num_spheres[None] = 0
        num_cubes[None] = 0
        num_lights[None] = 0
        num_materials[None] = 0
        frame_count[None] = 0

        # Clear accumulator
        clear_accumulator()

        # Setup scene
        self.setup_scene()

    def add_material(self, color, reflectivity=0.0, roughness=0.0):
        """Add material to GPU memory"""
        idx = num_materials[None]
        if idx < MAX_MATERIALS:
            materials[idx] = Material(
                color=ti.Vector(color),
                reflectivity=reflectivity,
                roughness=roughness
            )
            num_materials[None] += 1
            return idx
        return -1

    def add_sphere(self, center, radius, material_idx):
        """Add sphere to GPU memory"""
        idx = num_spheres[None]
        if idx < MAX_SPHERES and material_idx >= 0:
            spheres[idx] = Sphere(
                center=ti.Vector(center),
                radius=radius,
                material_idx=material_idx
            )
            num_spheres[None] += 1

    def add_cube(self, center, size, material_idx):
        """Add cube to GPU memory"""
        idx = num_cubes[None]
        if idx < MAX_CUBES and material_idx >= 0:
            cubes[idx] = Cube(
                center=ti.Vector(center),
                size=size,
                material_idx=material_idx
            )
            num_cubes[None] += 1

    def add_light(self, position, intensity=1.0, color=(1.0, 1.0, 1.0)):
        """Add light to GPU memory"""
        idx = num_lights[None]
        if idx < MAX_LIGHTS:
            lights[idx] = Light(
                position=ti.Vector(position),
                intensity=intensity,
                color=ti.Vector(color)
            )
            num_lights[None] += 1

    def setup_scene(self):
        """Setup the scene with objects and lights"""
        # Materials
        red_mat = self.add_material((1.0, 0.2, 0.2), reflectivity=0.3)
        blue_mat = self.add_material((0.2, 0.2, 1.0), reflectivity=0.5)
        green_mat = self.add_material((0.2, 1.0, 0.2), reflectivity=0.1)
        yellow_mat = self.add_material((1.0, 1.0, 0.2), reflectivity=0.2, roughness=0.1)
        gray_mat = self.add_material((0.7, 0.7, 0.7), reflectivity=0.05, roughness=0.3)

        # Objects
        self.add_sphere((-2, 1, 0), 1.0, red_mat)
        self.add_sphere((2, 1, 0), 1.0, blue_mat)
        self.add_sphere((0, 1.5, -2), 1.0, green_mat)
        self.add_cube((0, -1, 0), 10.0, yellow_mat)
        self.add_sphere((-1, 0.5, 2), 0.5, gray_mat)
        self.add_cube((3, 0.5, -1), 1.0, red_mat)

        # Lights
        self.add_light((3, 5, 3), 1.0)
        self.add_light((-3, 5, -3), 0.7, (0.9, 0.9, 1.0))
        self.add_light((0, 8, 0), 0.5, (1.0, 1.0, 0.8))

class Game:
    """Main game class for PyGame interaction"""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("GPU Raytracing Engine - WASD Movement")
        self.clock = pygame.time.Clock()
        self.scene = Scene()
        self.running = True

        # Camera in Python scope (will be converted to ti.math.vec3 for GPU)
        self.camera_pos = [0.0, 2.0, 5.0]
        self.camera_target = [0.0, 0.0, 0.0]
        self.camera_up = [0.0, 1.0, 0.0]

        # Camera orientation
        self.yaw = 0.0
        self.pitch = 0.0

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

        # Mouse control
        self.mouse_pos = None
        self.mouse_sensitivity = 0.002

        # Performance tracking
        self.frame_times = []
        self.fps = 0
        self.font = pygame.font.SysFont(None, 24)

        print("GPU Raytracing Engine Initialized")
        print(f"Using Taichi backend: {ti.cfg.arch}")
        print(f"Resolution: {WIDTH}x{HEIGHT}")
        print("Controls:")
        print("  WASD - Movement")
        print("  Q/E - Roll left/right")
        print("  Shift/Space - Down/Up")
        print("  Mouse - Look around")
        print("  R - Reset accumulation")
        print("  ESC - Quit")

    def update_camera_vectors(self):
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
        self.camera_target = [
            self.camera_pos[0] + forward_x,
            self.camera_pos[1] + forward_y,
            self.camera_pos[2] + forward_z
        ]

        # Update camera up vector
        self.camera_up = [up_x, up_y, up_z]

    def handle_events(self):
        """Handle PyGame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in self.keys:
                    self.keys[event.key] = True
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    # Reset accumulation
                    frame_count[None] = 0
                    clear_accumulator()
            elif event.type == pygame.KEYUP:
                if event.key in self.keys:
                    self.keys[event.key] = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Start mouse look
                pygame.mouse.set_visible(False)
                pygame.event.set_grab(True)
                self.mouse_pos = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                # Stop mouse look
                pygame.mouse.set_visible(True)
                pygame.event.set_grab(False)
                self.mouse_pos = None
            elif event.type == pygame.MOUSEMOTION and self.mouse_pos is not None:
                # Handle mouse look
                dx, dy = event.rel
                self.yaw -= dx * self.mouse_sensitivity
                self.pitch = max(-math.pi/2.1, min(math.pi/2.1,
                    self.pitch - dy * self.mouse_sensitivity))
                self.update_camera_vectors()

    def update_movement(self, dt):
        """Update camera position based on keyboard input"""
        move_speed = 3.0 * dt

        # Calculate forward, right, and up vectors for movement
        forward = [
            self.camera_target[0] - self.camera_pos[0],
            self.camera_target[1] - self.camera_pos[1],
            self.camera_target[2] - self.camera_pos[2]
        ]

        # Normalize forward
        length = math.sqrt(forward[0]**2 + forward[1]**2 + forward[2]**2)
        if length > 0:
            forward = [f/length for f in forward]

        # Calculate right vector (cross forward with world up)
        world_up = [0.0, 1.0, 0.0]
        right = [
            forward[1] * world_up[2] - forward[2] * world_up[1],
            forward[2] * world_up[0] - forward[0] * world_up[2],
            forward[0] * world_up[1] - forward[1] * world_up[0]
        ]

        # Normalize right
        length = math.sqrt(right[0]**2 + right[1]**2 + right[2]**2)
        if length > 0:
            right = [r/length for r in right]

        # Apply movement
        if self.keys[pygame.K_w]:
            self.camera_pos = [p + f * move_speed for p, f in zip(self.camera_pos, forward)]
            self.camera_target = [t + f * move_speed for t, f in zip(self.camera_target, forward)]

        if self.keys[pygame.K_s]:
            self.camera_pos = [p - f * move_speed for p, f in zip(self.camera_pos, forward)]
            self.camera_target = [t - f * move_speed for t, f in zip(self.camera_target, forward)]

        if self.keys[pygame.K_a]:
            self.camera_pos = [p - r * move_speed for p, r in zip(self.camera_pos, right)]
            self.camera_target = [t - r * move_speed for t, r in zip(self.camera_target, right)]

        if self.keys[pygame.K_d]:
            self.camera_pos = [p + r * move_speed for p, r in zip(self.camera_pos, right)]
            self.camera_target = [t + r * move_speed for t, r in zip(self.camera_target, right)]

        if self.keys[pygame.K_SPACE]:
            self.camera_pos[1] += move_speed
            self.camera_target[1] += move_speed

        if self.keys[pygame.K_LSHIFT]:
            self.camera_pos[1] -= move_speed
            self.camera_target[1] -= move_speed

        # Update camera vectors for roll (Q/E)
        if self.keys[pygame.K_q] or self.keys[pygame.K_e]:
            # Simple roll by rotating up vector
            angle = 1.5 * dt
            if self.keys[pygame.K_e]:
                angle = -angle

            up = self.camera_up
            # Rotate up vector around forward axis
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            # Find axis perpendicular to forward and current up
            axis = [
                forward[1] * up[2] - forward[2] * up[1],
                forward[2] * up[0] - forward[0] * up[2],
                forward[0] * up[1] - forward[1] * up[0]
            ]

            # Rodrigues rotation formula
            dot = up[0]*forward[0] + up[1]*forward[1] + up[2]*forward[2]
            for i in range(3):
                up[i] = up[i] * cos_a + axis[i] * sin_a + forward[i] * dot * (1 - cos_a)

            # Normalize and store
            length = math.sqrt(up[0]**2 + up[1]**2 + up[2]**2)
            if length > 0:
                self.camera_up = [u/length for u in up]

    def render_ui(self, render_time):
        """Render the UI overlay"""
        # Convert GPU pixels to PyGame surface
        pixel_array = pixels.to_numpy()
        pixel_array = np.swapaxes(pixel_array, 0, 1)
        surf = pygame.surfarray.make_surface(pixel_array * 255)
        self.screen.blit(surf, (0, 0))

        # Update FPS calculation
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
            # Text background for readability
            bg_rect = text_surf.get_rect()
            bg_rect.x = 10
            bg_rect.y = 10 + i * 25
            pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect.inflate(10, 5))
            self.screen.blit(text_surf, (15, 12 + i * 25))

    def run(self):
        """Main game loop"""
        last_time = time.time()

        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # Handle input
            self.handle_events()

            # Update camera movement
            self.update_movement(dt)

            # Start GPU rendering
            render_start = time.time()

            # Get current frame count
            current_frame = frame_count[None]

            # Convert Python lists to ti.math.vec3 for GPU
            camera_pos_vec = ti.math.vec3(self.camera_pos[0], self.camera_pos[1], self.camera_pos[2])
            camera_target_vec = ti.math.vec3(self.camera_target[0], self.camera_target[1], self.camera_target[2])
            camera_up_vec = ti.math.vec3(self.camera_up[0], self.camera_up[1], self.camera_up[2])

            # Launch GPU kernel
            render(camera_pos_vec, camera_target_vec, camera_up_vec, current_frame)
            ti.sync()  # Wait for GPU to finish

            # Update frame count
            frame_count[None] = current_frame + 1

            # Calculate render time
            render_time = time.time() - render_start

            # Display result
            self.render_ui(render_time)
            pygame.display.flip()

            # Cap frame rate
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()

