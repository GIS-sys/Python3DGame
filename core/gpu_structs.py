import taichi as ti

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

