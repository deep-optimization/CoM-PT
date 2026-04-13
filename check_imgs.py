import os
import warnings
from PIL import Image, ExifTags, UnidentifiedImageError

def is_image_corrupted(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # 验证图片文件是否完整
        return False
    except (IOError, SyntaxError, UnidentifiedImageError) as e:
        print(f"文件损坏: {file_path} - {e}")
        return True

def check_and_fix_exif(file_path):
    try:
        with Image.open(file_path) as img:
            exif_data = img._getexif()
            if exif_data is None:
                pass
                # print(f"缺失 EXIF 数据: {file_path}")
                # 这里可以添加修复 EXIF 数据的逻辑
                # 例如，重新保存图片以添加基本的 EXIF 数据
                # img.save(file_path, "JPEG")
                # print(f"已修复 EXIF 数据: {file_path}")
            return False
    except (IOError, SyntaxError, UnidentifiedImageError) as e:
        print(f"文件损坏: {file_path} - {e}")
        return True

def check_file(file_path):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            with Image.open(file_path) as img:
                img.verify()
        except (IOError, SyntaxError, UnidentifiedImageError) as e:
                os.remove(file_path)  # 删除截断的文件
                print(f"文件损坏: {file_path} - {e}")
                return False

        for warning in w:
            if "Corrupt EXIF data" in str(warning.message):
                print(f"缺失 EXIF 数据: {file_path}")
                with Image.open(file_path) as img:
                    img.save(file_path)
                print(f"已修改EXIF: {file_path}")
                            
            if "Truncated File Read" in str(warning.message):
                print(f"发现截断文件: {file_path}")
                os.remove(file_path)  # 删除截断的文件
                print(f"已删除截断文件: {file_path}")

            return False

    return True

def check_images_in_folder(folder_path):
    corrupted_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                file_path = os.path.join(root, file)
                flag = check_file(file_path)

    return corrupted_files

if __name__ == "__main__":
    folder_path = input("请输入要检查的文件夹路径: ")
    corrupted_files = check_images_in_folder(folder_path)
    if corrupted_files:
        print("以下文件已损坏并已删除:")
        for file in corrupted_files:
            print(file)
    else:
        print("所有图片文件均完好无损。")