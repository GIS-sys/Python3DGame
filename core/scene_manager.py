import taichi as ti
import numpy as np
from core.constants import MAX_SPHERES, MAX_CUBES, MAX_LIGHTS, MAX_MATERIALS
from core.gpu_structs import Sphere, Cube, Material, Light
from core.gpu_kernels import clear_accumulator

class SceneManager:
    """Manages the scene on both Python and GPU sides"""

    def __init__(self):
        # GPU storage
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(1280, 960))
        self.accumulator = ti.Vector.field(3, dtype=ti.f32, shape=(1280, 960))
        self.frame_count = ti.field(dtype=ti.i32, shape=())

        self.spheres = Sphere.field()
        ti.root.dense(ti.i, MAX_SPHERES).place(self.spheres)
        self.cubes = Cube.field()
        ti.root.dense(ti.i, MAX_CUBES).place(self.cubes)
        self.lights = Light.field()
        ti.root.dense(ti.i, MAX_LIGHTS).place(self.lights)
        self.materials = Material.field()
        ti.root.dense(ti.i, MAX_MATERIALS).place(self.materials)

        # Scene counters
        self.num_spheres = ti.field(dtype=ti.i32, shape=())
        self.num_cubes = ti.field(dtype=ti.i32, shape=())
        self.num_lights = ti.field(dtype=ti.i32, shape=())
        self.num_materials = ti.field(dtype=ti.i32, shape=())

        # Reset counters
        self.reset_scene()

    def reset_scene(self):
        """Reset the scene to initial state"""
        self.num_spheres[None] = 0
        self.num_cubes[None] = 0
        self.num_lights[None] = 0
        self.num_materials[None] = 0
        self.frame_count[None] = 0
        clear_accumulator(self.accumulator)

    def add_material(self, color, reflectivity=0.0, roughness=0.0):
        """Add material to GPU memory"""
        idx = self.num_materials[None]
        if idx < MAX_MATERIALS:
            self.materials[idx] = Material(
                color=ti.Vector(color),
                reflectivity=reflectivity,
                roughness=roughness
            )
            self.num_materials[None] += 1
            return idx
        return -1

    def add_sphere(self, center, radius, material_idx):
        """Add sphere to GPU memory"""
        idx = self.num_spheres[None]
        if idx < MAX_SPHERES and material_idx >= 0:
            self.spheres[idx] = Sphere(
                center=ti.Vector(center),
                radius=radius,
                material_idx=material_idx
            )
            self.num_spheres[None] += 1

    def add_cube(self, center, size, material_idx):
        """Add cube to GPU memory"""
        idx = self.num_cubes[None]
        if idx < MAX_CUBES and material_idx >= 0:
            self.cubes[idx] = Cube(
                center=ti.Vector(center),
                size=size,
                material_idx=material_idx
            )
            self.num_cubes[None] += 1

    def add_light(self, position, intensity=1.0, color=(1.0, 1.0, 1.0)):
        """Add light to GPU memory"""
        idx = self.num_lights[None]
        if idx < MAX_LIGHTS:
            self.lights[idx] = Light(
                position=ti.Vector(position),
                intensity=intensity,
                color=ti.Vector(color)
            )
            self.num_lights[None] += 1

    def setup_default_scene(self):
        """Setup the default scene with objects and lights"""
        # Materials
        red_mat = self.add_material((1.0, 0.2, 0.2), reflectivity=0.0)
        blue_mat = self.add_material((0.2, 0.2, 1.0), reflectivity=0.5)
        green_mat = self.add_material((0.2, 1.0, 0.2), reflectivity=1.0)
        yellow_mat = self.add_material((1.0, 1.0, 0.2), reflectivity=0.2, roughness=0.1)
        gray_mat = self.add_material((0.7, 0.7, 0.7), reflectivity=0.05, roughness=0.3)

        # Objects
        self.add_sphere((-2, 1, 0), 1.0, red_mat)
        self.add_sphere((2, 1, 0), 1.0, blue_mat)
        self.add_sphere((0, 1.5, -2), 1.0, green_mat)
        self.add_cube((0, -1, 0), 1.0, yellow_mat)
        self.add_sphere((-1, 0.5, 2), 0.5, gray_mat)
        self.add_cube((3, 0.5, -1), 1.0, red_mat)

        # Lights
        self.add_light((3, 5, 3), 1.0)
        self.add_light((-3, 5, -3), 0.7, (0.9, 0.9, 1.0))
        self.add_light((0, 8, 0), 0.5, (1.0, 1.0, 0.8))

    def get_pixels_as_numpy(self):
        """Get pixel data as numpy array for display"""
        return self.pixels.to_numpy()

    def clear_accumulation(self):
        """Clear the accumulation buffer"""
        clear_accumulator(self.accumulator)
        self.frame_count[None] = 0

