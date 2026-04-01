import bpy
import math
import os
from mathutils import Vector

# ========== 配置参数 ==========
INPUT_FILE = "/Users/yjy/Desktop/3DView/Zhan/depth_warp_vs/renderpeople_free_posed_people_MAX/rp_mei_posed_001_MAX/rp_mei_posed_001_100k.fbx"
OUTPUT_DIR = "./3dTo2d"
OUTPUT_FORMAT = "PNG"
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# 相机核心参数（重点调整）
CAMERA_DISTANCE = 1.5  # 相机与模型的水平距离（非斜距）
CAMERA_FOV = 25  # 相机视场角（度）
EYE_HEIGHT_RATIO = 0.9  # 模型脚底到眼睛的高度比例
CAMERA_PITCH_OFFSET = 0.0  # 俯仰角微调（正值仰拍，负值俯拍，单位：度）
IGNORE_BASE_THRESHOLD = 0.1  # 忽略模型底部微小凸起（如底座，单位：米）

# 视角设置
LEFT_VIEW_ANGLE = -5
RIGHT_VIEW_ANGLE = 5

# 光照设置
LIGHT_ENERGY = 300
USE_HDRI = False
HDRI_PATH = ""

# ========== 核心修正函数 ==========
def clear_scene():
    """清除场景中的所有对象"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def import_model(filepath):
    """导入3D模型"""
    file_ext = os.path.splitext(filepath)[1].lower()
    if file_ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif file_ext == '.obj':
        bpy.ops.import_scene.obj(filepath=filepath)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")
    print(f"已导入模型: {filepath}")

def get_model_foot_and_eye_height():
    """精准计算模型脚底高度和平视高度（排除底座/地面干扰）"""
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not mesh_objects:
        return 0.0, 1.8  # 默认值（成人身高1.8米）
    
    # 收集所有顶点的世界坐标
    all_vertices = []
    for obj in mesh_objects:
        for vertex in obj.data.vertices:
            world_coord = obj.matrix_world @ Vector(vertex.co)
            all_vertices.append(world_coord)
    
    # 计算Z轴坐标分布，排除底座/噪声（取5%分位数作为真实脚底）
    z_coords = sorted([v.z for v in all_vertices])
    foot_z = z_coords[int(len(z_coords) * 0.05)]  # 5%分位数避免底座干扰
    max_z = max(z_coords)
    model_height = max_z - foot_z
    
    # 计算平视高度（眼睛位置）
    eye_z = foot_z + EYE_HEIGHT_RATIO * model_height
    
    print(f"模型真实脚底Z: {foot_z:.2f}m")
    print(f"模型总高度: {model_height:.2f}m")
    print(f"平视高度（眼睛）Z: {eye_z:.2f}m")
    return foot_z, eye_z, model_height

def setup_camera(angle_degrees, horizontal_distance, eye_z, foot_z, model_center_xy):
    """
    配置平视相机：
    - 相机高度 = 眼睛高度
    - 朝向 = 与相机同高度的模型中心（XY不变，Z=eye_z）
    - 强制水平视线（无俯仰）
    """
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    
    # 1. 计算相机XY位置（仅水平旋转，无Z偏移）
    angle_rad = math.radians(angle_degrees)
    camera.location.x = model_center_xy[0] + horizontal_distance * math.sin(angle_rad)
    camera.location.y = model_center_xy[1] - horizontal_distance * math.cos(angle_rad)
    camera.location.z = eye_z  # 相机高度 = 模型眼睛高度
    
    # 2. 计算平视目标点（与相机同Z高度，避免俯/仰拍）
    target_point = Vector((
        model_center_xy[0],
        model_center_xy[1],
        eye_z  # 关键：目标点Z = 相机Z，强制水平视线
    ))
    
    # 3. 计算相机朝向（纯水平，无俯仰）
    direction = target_point - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    
    # 4. 微调俯仰角（可选）
    pitch_offset_rad = math.radians(CAMERA_PITCH_OFFSET)
    camera.rotation_euler.x += pitch_offset_rad  # X轴是俯仰轴
    
    # 相机参数
    camera.data.lens_unit = 'FOV'
    camera.data.angle = math.radians(CAMERA_FOV)
    
    print(f"相机位置: {camera.location}")
    print(f"相机朝向目标: {target_point}")
    print(f"相机俯仰角: {math.degrees(camera.rotation_euler.x):.2f}度")
    return camera

def get_model_center_xy():
    """获取模型XY平面中心（排除Z轴干扰）"""
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not mesh_objects:
        return (0.0, 0.0)
    
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    for obj in mesh_objects:
        for vertex in obj.data.vertices:
            world_coord = obj.matrix_world @ Vector(vertex.co)
            min_x = min(min_x, world_coord.x)
            max_x = max(max_x, world_coord.x)
            min_y = min(min_y, world_coord.y)
            max_y = max(max_y, world_coord.y)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    return (center_x, center_y)

def setup_lighting(center, camera_obj=None):
    """设置场景光照"""
    if USE_HDRI and HDRI_PATH:
        world = bpy.context.scene.world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        nodes.clear()
        
        node_env = nodes.new('ShaderNodeTexEnvironment')
        node_env.image = bpy.data.images.load(HDRI_PATH)
        node_background = nodes.new('ShaderNodeBackground')
        node_output = nodes.new('ShaderNodeOutputWorld')
        
        world.node_tree.links.new(node_env.outputs['Color'], node_background.inputs['Color'])
        world.node_tree.links.new(node_background.outputs['Background'], node_output.inputs['Surface'])
    else:
        if camera_obj:
            bpy.ops.object.light_add(type='SPOT', location=camera_obj.location)
            front_light = bpy.context.active_object
            front_light.data.energy = LIGHT_ENERGY * 1.5
            direction = Vector(center) - front_light.location
            rot_quat = direction.to_track_quat('-Z', 'Y')
            front_light.rotation_euler = rot_quat.to_euler()
            front_light.data.spot_size = math.radians(60)
        else:
            bpy.ops.object.light_add(type='SUN', location=(center[0] + 5, center[1] - 5, center[2] + 5))
            key_light = bpy.context.active_object
            key_light.data.energy = LIGHT_ENERGY
            
            bpy.ops.object.light_add(type='AREA', location=(center[0] - 3, center[1] - 3, center[2] + 2))
            fill_light = bpy.context.active_object
            fill_light.data.energy = LIGHT_ENERGY * 0.5
            
            bpy.ops.object.light_add(type='SPOT', location=(center[0], center[1] + 5, center[2] + 3))
            rim_light = bpy.context.active_object
            rim_light.data.energy = LIGHT_ENERGY * 0.7

def setup_render_settings():
    """设置渲染参数"""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 128
    scene.render.resolution_x = IMAGE_WIDTH
    scene.render.resolution_y = IMAGE_HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = OUTPUT_FORMAT
    if OUTPUT_FORMAT == 'PNG':
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.image_settings.compression = 15
    scene.render.film_transparent = True

def render_view(camera, output_path):
    """渲染指定相机视角"""
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"已渲染: {output_path}")

def main():
    """主函数"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clear_scene()
    import_model(INPUT_FILE)
    
    # 1. 获取模型核心参数
    foot_z, eye_z, model_height = get_model_foot_and_eye_height()
    model_center_xy = get_model_center_xy()
    print(f"模型XY中心: {model_center_xy}")
    
    # 2. 计算相机水平距离（基于模型宽度/深度，非高度）
    horizontal_distance = CAMERA_DISTANCE * (model_height * 0.8)  # 适配人体比例
    
    # 3. 渲染设置
    setup_render_settings()
    
    # 4. 渲染左视图
    left_camera = setup_camera(LEFT_VIEW_ANGLE, horizontal_distance, eye_z, foot_z, model_center_xy)
    setup_lighting(model_center_xy + (eye_z,), left_camera)  # 光源朝向平视目标点
    left_output = os.path.join(OUTPUT_DIR, "left_view.png")
    render_view(left_camera, left_output)
    # bpy.data.objects.remove(left_camera, do_unlink=True)  #移除左光源
    
    # 5. 渲染右视图
    right_camera = setup_camera(RIGHT_VIEW_ANGLE, horizontal_distance, eye_z, foot_z, model_center_xy)
    light_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'LIGHT']
    # for obj in light_objects:
        # bpy.data.objects.remove(obj, do_unlink=True) #移除左光源
    # setup_lighting(model_center_xy + (eye_z,), right_camera) #重置右光源
    right_output = os.path.join(OUTPUT_DIR, "right_view.png")
    render_view(right_camera, right_output)
    
    print("\n渲染完成!")
    print(f"输出目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()