import os
import os.path as op
from PIL import Image


class ImagePathTester:
    def __init__(self, image_dir):
        self.image_dir = image_dir

    def _build_image_path_map(self):
        """构建seqID到图片路径的映射"""
        image_map = {}
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    seq_id = op.splitext(file)[0]  # 假设文件名就是seqID
                    image_map[seq_id] = op.join(root, file)
        return image_map

    def test_image_loading(self, test_id=None):
        """测试图片加载功能"""
        image_map = self._build_image_path_map()

        if not image_map:
            print("No images found in the directory!")
            return

        # 如果没有指定test_id，使用第一个找到的图片
        test_id = test_id or next(iter(image_map.keys()))

        if test_id in image_map:
            print(f"\nTesting with ID: {test_id}")
            print(f"Image path: {image_map[test_id]}")

            try:
                # 尝试打开图片
                img = Image.open(image_map[test_id])
                print(f"Image loaded successfully! Size: {img.size}")
                img.close()
                return True
            except Exception as e:
                print(f"Failed to load image: {e}")
                return False
        else:
            print(f"Test ID {test_id} not found in image map!")
            print("Available IDs (first 10):", list(image_map.keys())[:10])
            return False


if __name__ == "__main__":
    # 设置测试参数
    test_image_dir = r"/root/autodl-tmp/LLaRA/data/ref/steam/steam_posters"  # 修改为你的实际路径
    test_id = "123"  # 修改为你想要测试的具体ID，或设为None自动选择第一个

    # 创建测试器并运行测试
    tester = ImagePathTester(test_image_dir)
    success = tester.test_image_loading(test_id)
    # 输出完整映射（前10项）
    full_map = tester._build_image_path_map()
    print("\nSample of image mapping (first 10 items):")
    for i, (k, v) in enumerate(list(full_map.items())[:10]):
        print(f"{k}: {v}")

    print(f"\nTest {'passed' if success else 'failed'}!")