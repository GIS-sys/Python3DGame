import taichi as ti
import math
from core.gpu_structs import Sphere, Cube, Material, Light
from core.constants import WIDTH, HEIGHT, FOV, MAX_DEPTH, BACKGROUND_COLOR

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
def trace_ray(ray_origin: ti.math.vec3, ray_dir: ti.math.vec3,
              spheres, cubes, lights, materials,
              num_spheres, num_cubes, num_lights) -> ti.math.vec3:
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
            color += ti.math.vec3(BACKGROUND_COLOR) * ray_weight
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
def clear_accumulator(accumulator: ti.template()):
    """Clear the accumulation buffer"""
    for i, j in accumulator:
        accumulator[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def render(camera_pos: ti.math.vec3, camera_target: ti.math.vec3,
           camera_up: ti.math.vec3, frame: ti.i32,
           pixels: ti.template(), accumulator: ti.template(),
           spheres: ti.template(), cubes: ti.template(), 
           lights: ti.template(), materials: ti.template(),
           num_spheres: ti.template(), num_cubes: ti.template(), 
           num_lights: ti.template()):
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
        color = trace_ray(camera_pos, ray_dir, spheres, cubes, lights, materials,
                          num_spheres, num_cubes, num_lights)

        # Accumulate for progressive rendering
        accumulator[i, j] = color
        pixels[i, j] = accumulator[i, j]

