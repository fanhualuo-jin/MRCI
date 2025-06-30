import os
import requests
from bs4 import BeautifulSoup

# Clash 代理配置（如果不需要代理，可以将 proxies 设置为 None）
proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890',
}

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.5',  # 设置接受语言为英语
}

# 输出文件夹
OUTPUT_DIR = 'steam_posters'

def setup():
    """初始化环境，创建输出文件夹"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def get_steam_poster(game_name):
    """获取 Steam 游戏的海报 URL"""
    try:
        # 搜索游戏
        search_url = f"https://store.steampowered.com/search/?term={game_name}"
        response = requests.get(search_url, headers=headers, proxies=proxies, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        game_link = soup.select_one('a.search_result_row[href]')
        if not game_link:
            print(f"未找到游戏: {game_name}")
            return None

        # 获取游戏详情页
        game_url = game_link['href']
        response = requests.get(game_url, headers=headers, proxies=proxies, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # 提取海报
        poster = soup.select_one('img.game_header_image_full') or soup.select_one('img.game_header_image')
        return poster['src'] if poster else None

    except Exception as e:
        print(f"获取海报失败: {game_name} ({e})")
        return None

def download_poster(poster_url, game_name,game_id):
    """下载并保存海报"""
    try:
        response = requests.get(poster_url, stream=True, proxies=proxies, timeout=10)
        response.raise_for_status()

        # 文件名处理
        # ext = poster_url.split('.')[-1].split('?')[0].lower()
        # ext = ext if ext in ['jpg', 'jpeg', 'png', 'webp'] else 'jpg'
        # safe_name = ''.join(c for c in game_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
        # filename = f"{OUTPUT_DIR}/{safe_name}.{ext}"
        ext = poster_url.split('.')[-1].split('?')[0].lower()
        ext = ext if ext in ['jpg', 'jpeg', 'png', 'webp'] else 'jpg'
        filename = f"{OUTPUT_DIR}/{game_id}.{ext}"



        # 保存文件
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        print(f"海报已保存: {filename}")
        return True

    except Exception as e:
        print(f"下载失败: {game_name} ({e})")
        return False

def read_game_list(file_path):
    """从txt文件读取游戏列表，返回(id, 游戏名)元组列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [
                tuple(line.strip().split('::', 1))  # 分割每行并返回元组
                for line in f
                if line.strip() and '::' in line
            ]
    except Exception as e:
        print(f"读取文件失败: {e}")
        return []

def main():
    setup()
    game_list = read_game_list('id2name.txt')  # 假设文件名为 id2name.txt
    if not game_list:
        print("未找到有效的游戏名称列表")
        return

    print(f"开始处理 {len(game_list)} 个游戏...")
    for game_id, game_name in game_list:
        print(f"\n处理游戏: {game_id}::{game_name}")
        poster_url = get_steam_poster(game_name)
        if poster_url:
            download_poster(poster_url, game_name,game_id)

    print("\n全部完成!")

if __name__ == "__main__":
    main()