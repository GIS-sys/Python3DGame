import pygame
import sys
import time
import taichi as ti
import math
from core.scene_manager import SceneManager
from core.gpu_kernels import render
from game.camera import Camera

class GameEngine:
    """Main game engine combining PyGame and Taichi"""

    def __init__(self, width=1280, height=960):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("GPU Raytracing Engine - WASD Movement")
        self.clock = pygame.time.Clock()

        # Initialize Taichi
        ti.init(arch=ti.gpu, default_fp=ti.f32)

        # Create scene manager
        self.scene = SceneManager()
        self.scene.setup_default_scene()

        # Create camera
        self.camera = Camera()
        self.camera.update_vectors()

        # Game state
        self.running = True
        self.keys = {
            pygame.K_w: False,
            pygame.K_s: False,
            pygame.K_a: False,
            pygame.K_d: False,
            pygame.K_LSHIFT: False,
            pygame.K_SPACE: False
        }

        # Object animation
        self.rotation_time = 0.0
        self.rotating_sphere_idx = 2

        # Mouse control
        self.mouse_pos = None

        # Performance tracking
        self.frame_times = []
        self.fps = 0
        self.font = pygame.font.SysFont(None, 24)

        print("GPU Raytracing Engine Initialized")
        print(f"Using Taichi backend: {ti.cfg.arch}")
        print(f"Resolution: {width}x{height}")
        print("Controls:")
        print("  WASD - Movement")
        print("  Shift/Space - Down/Up")
        print("  Mouse - Look around")
        print("  R - Reset accumulation")
        print("  ESC - Quit")

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
                    self.scene.clear_accumulation()
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
                self.camera.rotate(dx, dy)

    def update_objects(self, dt):
        """Update object animations"""
        self.rotation_time += dt

        # Update the rotating sphere position
        sphere_idx = self.rotating_sphere_idx
        if sphere_idx < self.scene.num_spheres[None]:
            # Create a figure-8 or lemniscate motion
            angle = self.rotation_time * 0.8  # Rotate 0.8 radians per second

            # Lemniscate (figure-8) pattern
            a = 2.0  # Size of the figure-8
            denominator = 1.0 + math.sin(angle) * math.sin(angle)

            new_x = a * math.cos(angle) / denominator
            new_z = a * math.sin(angle) * math.cos(angle) / denominator
            new_y = 1.5 + 0.3 * math.sin(self.rotation_time * 1.5)  # Bob up and down

            # Get the current sphere and update its position
            current_sphere = self.scene.spheres[sphere_idx]  # Read from GPU memory
            new_center = ti.Vector([new_x, new_y, new_z])

            # Write back to GPU memory
            self.scene.spheres[sphere_idx] = type(current_sphere)(
                center=new_center,
                radius=current_sphere.radius,
                material_idx=current_sphere.material_idx
            )

    def update_movement(self, dt):
        """Update camera position based on keyboard input"""
        move_speed = 3.0 * dt

        # Apply movement
        if self.keys[pygame.K_w]:
            self.camera.move_forward(move_speed)
        if self.keys[pygame.K_s]:
            self.camera.move_backward(move_speed)
        if self.keys[pygame.K_a]:
            self.camera.strafe_left(move_speed)
        if self.keys[pygame.K_d]:
            self.camera.strafe_right(move_speed)
        if self.keys[pygame.K_SPACE]:
            self.camera.move_up(move_speed)
        if self.keys[pygame.K_LSHIFT]:
            self.camera.move_down(move_speed)

    def render_ui(self, render_time):
        """Render the UI overlay"""
        # Convert GPU pixels to PyGame surface
        pixel_array = self.scene.get_pixels_as_numpy()
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
            f"Samples: {self.scene.frame_count[None] + 1}",
            "WASD: Move  Shift/Space: Up/Down",
            "Mouse: Look  R: Reset",
            f"Resolution: {self.width}x{self.height}",
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

            # Update object animations
            self.update_objects(dt)

            # Start GPU rendering
            render_start = time.time()

            # Get current frame count
            current_frame = self.scene.frame_count[None]

            # Convert camera data to ti.math.vec3 for GPU
            camera_pos_vec = ti.math.vec3(self.camera.position[0],
                                          self.camera.position[1],
                                          self.camera.position[2])
            camera_target_vec = ti.math.vec3(self.camera.target[0],
                                             self.camera.target[1],
                                             self.camera.target[2])
            camera_up_vec = ti.math.vec3(self.camera.up[0],
                                         self.camera.up[1],
                                         self.camera.up[2])

            # Launch GPU kernel
            render(camera_pos_vec, camera_target_vec, camera_up_vec, current_frame,
                   self.scene.pixels, self.scene.accumulator,
                   self.scene.spheres, self.scene.cubes, self.scene.lights, self.scene.materials,
                   self.scene.num_spheres, self.scene.num_cubes, self.scene.num_lights)
            ti.sync()  # Wait for GPU to finish

            # Update frame count
            self.scene.frame_count[None] = current_frame + 1

            # Calculate render time
            render_time = time.time() - render_start

            # Display result
            self.render_ui(render_time)
            pygame.display.flip()

            # Cap frame rate
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

