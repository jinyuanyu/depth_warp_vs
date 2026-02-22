import bpy
import math
import os
import sys
import shutil
import subprocess
from mathutils import Vector

# ========== 配置参数 ==========
# 输入模型路径
INPUT_FILE = "/Users/yjy/Desktop/3DView/Zhan/depth_warp_vs/renderpeople_free_posed_people_MAX/rp_mei_posed_001_MAX/rp_mei_posed_001_100k.fbx"

# 输出根目录
DATA_ROOT = "./data_dataset"

# 外部脚本配置 (必须修改!!!)
# 指向你系统中安装了 pytorch/opencv 的 python 解释器路径
SYSTEM_PYTHON_PATH = "/usr/bin/python3"  # <--- 请修改这里

# realtime_d455.py 的绝对路径
REALTIME_SCRIPT_PATH = "/Users/yjy/Desktop/3DView/Zhan/depth_warp_vs/realtime_d455.py" # <--- 请修改这里

# 图像参数
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 384
SAMPLES = 64

# 相机与模型参数
CAMERA_DISTANCE = 1.5
CAMERA_FOV = 50
EYE_HEIGHT_RATIO = 0.9
CAMERA_PITCH_OFFSET = 0.0

# ========== 视角循环设置 ==========
# 设定围绕模型旋转的范围 (度)
ROTATION_START = 0      # 起始角度
ROTATION_END = 180      # 结束角度 (不包含)
ROTATION_STEP = 5      # 每隔多少度生成一对

# 双目相对偏移 (度)
# 假设当前旋转角为 A，左眼为 A + LEFT_OFFSET，右眼为 A + RIGHT_OFFSET
LEFT_VIEW_OFFSET = -2.0
RIGHT_VIEW_OFFSET = 2.0

# 是否在生成左右视图后立即计算深度图
COMPUTE_DEPTH_AFTER_RENDER = True  # True: 生成后立即计算深度图; False: 只生成左右视图

# 光照
LIGHT_ENERGY = 300

# ========== 核心函数 ==========

def clear_scene():
    """清除场景"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes: bpy.data.meshes.remove(block)
    for block in bpy.data.materials: bpy.data.materials.remove(block)
    for block in bpy.data.textures: bpy.data.textures.remove(block)
    for block in bpy.data.images: bpy.data.images.remove(block)

def import_model(filepath):
    """导入模型"""
    file_ext = os.path.splitext(filepath)[1].lower()
    if file_ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif file_ext == '.obj':
        bpy.ops.import_scene.obj(filepath=filepath)
    else:
        raise ValueError(f"不支持格式: {file_ext}")

def get_model_metrics():
    """获取模型几何信息"""
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not mesh_objects: return 0, 1.8, 1.8, (0,0)
    
    all_vertices = []
    for obj in mesh_objects:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        for vertex in obj.data.vertices:
            world_coord = obj.matrix_world @ Vector(vertex.co)
            all_vertices.append(world_coord)
    
    z_coords = sorted([v.z for v in all_vertices])
    foot_z = z_coords[int(len(z_coords) * 0.05)]
    max_z = max(z_coords)
    model_height = max_z - foot_z
    eye_z = foot_z + EYE_HEIGHT_RATIO * model_height
    
    xs = [v.x for v in all_vertices]
    ys = [v.y for v in all_vertices]
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    
    return foot_z, eye_z, model_height, (center_x, center_y)

def setup_camera(angle_degrees, dist, eye_z, center_xy):
    """设置相机"""
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    
    angle_rad = math.radians(angle_degrees)
    # 极坐标定位
    cam.location.x = center_xy[0] + dist * math.sin(angle_rad)
    cam.location.y = center_xy[1] - dist * math.cos(angle_rad)
    cam.location.z = eye_z
    
    target = Vector((center_xy[0], center_xy[1], eye_z))
    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    cam.rotation_euler.x += math.radians(CAMERA_PITCH_OFFSET)
    
    cam.data.lens_unit = 'FOV'
    cam.data.angle = math.radians(CAMERA_FOV)
    return cam

def setup_lighting(center):
    """简单三点布光"""
    bpy.ops.object.light_add(type='AREA', location=(center[0]+3, center[1]-3, center[2]+3))
    light = bpy.context.active_object
    light.data.energy = LIGHT_ENERGY
    light.data.size = 2
    
    bpy.ops.object.light_add(type='AREA', location=(center[0]-3, center[1]-3, center[2]+1))
    light = bpy.context.active_object
    light.data.energy = LIGHT_ENERGY * 0.4
    
    bpy.ops.object.light_add(type='SPOT', location=(center[0], center[1]+4, center[2]+3))
    light = bpy.context.active_object
    light.data.energy = LIGHT_ENERGY * 0.8
    light.rotation_euler = (Vector(center)-light.location).to_track_quat('-Z', 'Y').to_euler()

def setup_render_settings():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    # GPU 设置
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    if cprefs.get_devices():
        cprefs.compute_device_type = 'CUDA' # 或 'OPTIX', 'METAL'
        scene.cycles.device = 'GPU'
    
    scene.cycles.samples = SAMPLES
    scene.render.resolution_x = IMAGE_WIDTH
    scene.render.resolution_y = IMAGE_HEIGHT
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

def call_external_depth_estimation(image_path, output_depth_path):
    """
    调用外部 python 脚本生成深度图
    """
    if not os.path.exists(SYSTEM_PYTHON_PATH):
        print(f"[Error] 找不到系统 Python: {SYSTEM_PYTHON_PATH}")
        return False
    if not os.path.exists(REALTIME_SCRIPT_PATH):
        print(f"[Error] 找不到深度估计脚本: {REALTIME_SCRIPT_PATH}")
        return False

    # 脚本所在目录 (作为工作目录)
    script_dir = os.path.dirname(REALTIME_SCRIPT_PATH)
    script_name = os.path.basename(REALTIME_SCRIPT_PATH)

    # 构建命令: python realtime_d455.py --image xxx.png --grayscale --encoder vitl
    cmd = [
        SYSTEM_PYTHON_PATH,
        script_name,
        "--image", image_path,
        "--grayscale", # 使用灰度图
        "--encoder", "vitl" # 使用大模型以获得更好精度
    ]

    print(f"-> 调用深度估计: {' '.join(cmd)}")
    try:
        # cwd参数很重要，确保脚本能找到它的 checkpoints
        subprocess.run(cmd, check=True, cwd=script_dir)
        
        # realtime_d455.py 默认保存为 'depth_result.png' 在脚本目录下
        default_output = os.path.join(script_dir, "depth_result.png")
        
        if os.path.exists(default_output):
            if os.path.exists(output_depth_path):
                os.remove(output_depth_path)
            shutil.move(default_output, output_depth_path)
            print(f"-> 深度图已保存: {output_depth_path}")
            return True
        else:
            print(f"[Error] 未找到生成的 depth_result.png")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"[Error] 深度估计脚本运行失败: {e}")
        return False

def main():
    # 0. 准备路径
    dir_left = os.path.join(DATA_ROOT, "left")
    dir_right = os.path.join(DATA_ROOT, "right")
    dir_depth = os.path.join(DATA_ROOT, "depth_left") # 深度图通常基于左视角
    
    for d in [dir_left, dir_right]:
        os.makedirs(d, exist_ok=True)
        
    print(f"=== 开始批量生成左右视图数据 ===")
    
    # 1. 初始化场景
    clear_scene()
    setup_render_settings()
    import_model(INPUT_FILE)
    
    # 2. 获取几何信息
    foot_z, eye_z, h, center_xy = get_model_metrics()
    dist = CAMERA_DISTANCE * (h * 0.8)
    setup_lighting((center_xy[0], center_xy[1], eye_z))
    
    # 3. 循环生成左右视图
    # range() 不支持 float，所以用 integer
    angle_list = list(range(ROTATION_START, ROTATION_END, ROTATION_STEP))
    
    for angle_step in angle_list:
        
        print(f"\n--- 处理角度: {angle_step}度 ---")
        
        # 计算当前视角的文件名后缀
        file_suffix = f"angle_{angle_step:03d}.png"
        
        # === 渲染左视图 ===
        current_left_angle = angle_step + LEFT_VIEW_OFFSET
        left_cam = setup_camera(current_left_angle, dist, eye_z, center_xy)
        bpy.context.scene.camera = left_cam
        
        left_img_path = os.path.join(dir_left, file_suffix)
        # Blender 渲染路径
        bpy.context.scene.render.filepath = left_img_path
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(left_cam, do_unlink=True)
        print(f"左视图渲染完成: {left_img_path}")
        
        # === 渲染右视图 ===
        current_right_angle = angle_step + RIGHT_VIEW_OFFSET
        right_cam = setup_camera(current_right_angle, dist, eye_z, center_xy)
        bpy.context.scene.camera = right_cam
        
        right_img_path = os.path.join(dir_right, file_suffix)
        bpy.context.scene.render.filepath = right_img_path
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(right_cam, do_unlink=True)
        print(f"右视图渲染完成: {right_img_path}")
        
        # === 可选：生成深度图 ===
        if COMPUTE_DEPTH_AFTER_RENDER:
            os.makedirs(dir_depth, exist_ok=True)
            depth_img_path = os.path.join(dir_depth, file_suffix)
            print(f"正在为左视图生成深度图...")
            success = call_external_depth_estimation(left_img_path, depth_img_path)
            if success:
                print(f"深度图生成成功: {depth_img_path}")
            else:
                print(f"深度图生成失败")
    
    print(f"\n=== 左右视图生成完成! ===")
    print(f"左视图保存到: {dir_left}")
    print(f"右视图保存到: {dir_right}")
    if COMPUTE_DEPTH_AFTER_RENDER:
        print(f"深度图保存到: {dir_depth}")
    print(f"总计生成 {len(angle_list)} 对左右视图")

if __name__ == "__main__":
    main()
#/Applications/Blender.app/Contents/MacOS/Blender --background --python dataCreate.py