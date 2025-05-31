from PIL import Image
import os

def resize_images_in_directory(directory_path, target_width=1280, target_height=720):
    """
    Resizes all images (JPG, JPEG, PNG) in a specified directory to a target resolution.

    Args:
        directory_path (str): The path to the directory containing images.
        target_width (int): The target width for resizing.
        target_height (int): The target height for resizing.
    """
    supported_formats = ('.jpg', '.jpeg', '.png') # 可以根据需要添加更多格式

    if not os.path.isdir(directory_path):
        print(f"错误：目录 '{directory_path}' 不存在。")
        return

    print(f"开始处理目录 '{directory_path}' 中的图片...")

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # 检查是否是文件以及是否是支持的图片格式
        if os.path.isfile(file_path) and filename.lower().endswith(supported_formats):
            try:
                img = Image.open(file_path)
                original_width, original_height = img.size

                if original_width == target_width and original_height == target_height:
                    print(f"图片 '{filename}' 已经是目标分辨率 {target_width}x{target_height}，跳过。")
                    continue

                print(f"正在调整图片 '{filename}' (原始尺寸: {original_width}x{original_height}) 到 {target_width}x{target_height}...")
                
                # 使用 ANTIALIAS 滤镜以获得较好的缩放质量
                resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS) # 或者 Image.ANTIALIAS (在旧版本 Pillow 中)
                
                # 覆盖原图保存，或者您可以选择保存到新的文件名或新目录
                resized_img.save(file_path)
                print(f"图片 '{filename}' 已成功调整并保存。")

            except Exception as e:
                print(f"处理图片 '{filename}' 时发生错误: {e}")
        elif os.path.isfile(file_path):
            print(f"跳过非支持格式的文件: '{filename}'")

    print("所有图片处理完毕。")

if __name__ == "__main__":
    # ----- 使用方法 -----
    # 1. 将下面的 'your_image_directory_path' 替换为您的图片文件夹的实际路径
    # 2. （可选）如果您需要不同的目标分辨率，可以修改 target_width 和 target_height 的值

    image_directory = '/mnt/A/xiongzhuang/Projects/GS-Blur/data/treehill/input'  # 例如: '/mnt/A/xiongzhuang/Projects/GS-Blur/datasets/my_scene/input'
    
    # 检查路径是否设置
    if image_directory == 'your_image_directory_path':
        print("请先在脚本中设置 'image_directory' 变量为您的图片文件夹路径！")
    else:
        resize_images_in_directory(image_directory, target_width=1280, target_height=720)