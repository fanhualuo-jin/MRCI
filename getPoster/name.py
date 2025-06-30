def read_game_list(file_path):
    """从txt文件读取游戏列表，返回(id, 游戏名)元组列表"""
    game_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '::' in line:
                    game_id, game_name = line.split('::', 1)
                    game_list.append((game_id.strip(), game_name.strip()))
        return game_list
    except Exception as e:
        print(f"读取游戏列表文件失败: {e}")
        return []
file_path = "id2name.txt"  # 假设文件在当前文件夹下
game_list = read_game_list(file_path)
print(game_list)