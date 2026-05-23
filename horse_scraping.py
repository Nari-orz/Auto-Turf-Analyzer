import time
import requests
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from io import StringIO

def parse_race_class(race_name: str) -> float:
    if not isinstance(race_name, str):
        return 0.0
    rn = race_name.strip().upper()
    if any(x in rn for x in ["G1", "GI", "Ｇ１", "ＧⅠ", "J.G1", "J.GI"]):
        return 9.0
    if any(x in rn for x in ["G2", "GII", "Ｇ２", "ＧⅡ", "J.G2", "J.GII"]):
        return 8.0
    if any(x in rn for x in ["G3", "GIII", "Ｇ３", "ＧⅢ", "J.G3", "J.GIII"]):
        return 7.0
    if any(x in rn for x in ["OP", "オープン", "LISTED", " Ｌ ", "(L)"]):
        return 6.0
    if any(x in rn for x in ["3勝クラス", "1600万下", "1600万"]):
        return 5.0
    if any(x in rn for x in ["2勝クラス", "1000万下", "1000万"]):
        return 4.0
    if any(x in rn for x in ["1勝クラス", "500万下", "500万"]):
        return 3.0
    if "未勝利" in rn:
        return 2.0
    if "新馬" in rn:
        return 1.0
    return 0.0

def get_horse_past_features(horse_id: str, current_race_id: str) -> dict:
    """
    個別の馬IDの過去のレース成績表をdb.netkeiba.comから取得し、
    今回のレースIDより前の過去のレースのみを使って、以下の4つの特徴量を計算して返す。
    
    特徴量:
    - prev_rank: 前走の着順（数値）
    - avg_up_3f_3runs: 過去3走の平均上がり3ハロンタイム (秒)
    - win_rate: 通算勝率（1着回数 / 出走回数）
    - prev_class: 前走のレースクラス（数値エンコード）
    """
    features = {
        'prev_rank': np.nan,
        'avg_up_3f_3runs': np.nan,
        'win_rate': 0.0,
        'prev_class': np.nan
    }
    
    if not horse_id or pd.isna(horse_id) or str(horse_id).strip() == "":
        return features
        
    url = f"https://db.netkeiba.com/horse/result/{horse_id}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
    }
    
    try:
        # サーバー負荷防止のため1秒待機
        time.sleep(1.0)
        
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'euc-jp'
        
        if response.status_code != 200:
            raise ValueError(f"HTTP Error {response.status_code}")
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 馬の過去走成績テーブルを検索 (db_h_race_results クラスを検索)
        table = soup.find('table', class_=lambda x: x and 'db_h_race_results' in x)
        if not table:
            # フォールバック: ページ内の最初の table でヘッダーに「着順」や「着」があるものを探索
            tables = soup.find_all('table')
            for t in tables:
                headers_text = [th.text.strip() for th in t.find_all('th')]
                if any('着順' in h or '着' in h for h in headers_text):
                    table = t
                    break
        
        if not table:
            raise ValueError("過去走成績テーブルが見つかりません (新馬・初出走など)")
            
        # HTMLをPandasで読み込み
        html_str = str(table)
        dfs = pd.read_html(StringIO(html_str))
        if not dfs or len(dfs) == 0:
            raise ValueError("テーブルのHTMLパースに失敗しました")
            
        df_history = dfs[0]
        
        # カラム名の余分な空白を除去
        df_history.columns = df_history.columns.astype(str).str.replace(r'\s+', '', regex=True)
        
        # -------------------------------------------------------------
        # 文字化けトラップを100%回避するインデックス決め打ち
        # -------------------------------------------------------------
        # ネット競馬の成績表の列位置は不変:
        # 11: 着順, 27: 上り3F
        idx_rank = 11
        idx_up3f = 27
        
        # セーフティフォールバック（列数が足りない場合は部分一致探索）
        if len(df_history.columns) <= max(idx_rank, idx_up3f):
            idx_rank = -1
            idx_up3f = -1
            for idx, col in enumerate(df_history.columns):
                col_str = str(col).strip()
                if '着' in col_str:
                    idx_rank = idx
                elif '上' in col_str or '3F' in col_str:
                    idx_up3f = idx
                
        # BeautifulSoup側のDOMから、各行のレースIDをhrefから抽出してマッピング
        rows = table.find_all('tr')[1:]
        past_race_ids = []
        for row in rows:
            tds = row.find_all('td')
            r_id = ""
            for td in tds:
                a_tag = td.find('a')
                if a_tag:
                    href = a_tag.get('href', '')
                    match = re.search(r'/race/(\d{12})', href)
                    if match:
                        r_id = match.group(1)
                        break
            past_race_ids.append(r_id)
            
        if len(past_race_ids) == len(df_history):
            df_history['race_id'] = past_race_ids
        else:
            df_history['race_id'] = ""
            
        # -------------------------------------------------------------
        # 今回のレースIDより過去、かつ有効な国内レース（12桁数値）のみに絞る
        # -------------------------------------------------------------
        current_race_id_str = str(current_race_id).strip()
        if 'race_id' in df_history.columns and current_race_id_str != "":
            # race_idが12桁の数字かつcurrent_race_idより小さいものを抽出 (海外レースを完全除外)
            df_past = df_history[
                (df_history['race_id'].astype(str).str.match(r'^\d{12}$')) & 
                (df_history['race_id'].astype(str) < current_race_id_str)
            ].copy()
        else:
            df_past = df_history.copy()
            
        if len(df_past) == 0:
            raise ValueError("今回のレースより過去の有効な成績データがありません")
            
        # 1. 前走の着順（数値）の算出
        if idx_rank != -1:
            # 有効な国内レースの直近の着順を取得
            prev_val = df_past.iloc[0, idx_rank]
            try:
                digits = re.sub(r'\D', '', str(prev_val))
                if digits:
                    features['prev_rank'] = float(digits)
            except ValueError:
                features['prev_rank'] = np.nan
                
        # 2. 過去3走の平均上がり3ハロンタイムの算出
        if idx_up3f != -1:
            # 有効な国内レースの上がり3Fタイムを取り出して数値化 (NaNは除外)
            up_3f_series = pd.to_numeric(df_past.iloc[:, idx_up3f], errors='coerce').dropna()
            # 直近3走の平均値
            if len(up_3f_series) > 0:
                features['avg_up_3f_3runs'] = float(up_3f_series.head(3).mean())
                
        # 3. 通算勝率（1着回数 / 出走回数）の算出
        if idx_rank != -1:
            # 国内の有効な着順のみを集計
            ranks = pd.to_numeric(df_past.iloc[:, idx_rank].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')
            valid_runs = len(ranks.dropna())
            if valid_runs > 0:
                wins = len(ranks[ranks == 1])
                features['win_rate'] = float(wins / valid_runs)
                
        # 4. 前走クラスの算出
        if len(df_past) > 0 and len(df_past.columns) > 4:
            prev_race_name = df_past.iloc[0, 4]
            features['prev_class'] = parse_race_class(str(prev_race_name))
                
    except Exception as e:
        print(f"馬ID {horse_id} の過去データ取得をスキップしました (理由: {e})")
        
    return features

import json
import os

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
JOCKEY_CACHE_PATH = os.path.join(CACHE_DIR, "jockey_stats_cache.json")
TRAINER_CACHE_PATH = os.path.join(CACHE_DIR, "trainer_stats_cache.json")

def load_json_cache(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        return super(MyEncoder, self).default(obj)

def save_json_cache(cache_data, path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=4, cls=MyEncoder)
    except Exception as e:
        print(f"[ERROR] Failed to save cache to {path}: {e}")

_jockey_cache = None
_trainer_cache = None

def get_jockey_stats(jockey_id: str) -> dict:
    global _jockey_cache
    if _jockey_cache is None:
        _jockey_cache = load_json_cache(JOCKEY_CACHE_PATH)
        
    stats = {
        'jockey_win_rate': 0.0,
        'jockey_place_rate': 0.0
    }
    
    if not jockey_id or pd.isna(jockey_id) or str(jockey_id).strip() == "" or str(jockey_id).strip() == "nan":
        return stats
        
    jockey_id_str = str(jockey_id).strip()
    if jockey_id_str in _jockey_cache:
        return _jockey_cache[jockey_id_str]
        
    url = f"https://db.netkeiba.com/jockey/{jockey_id_str}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        time.sleep(1.0)
        res = requests.get(url, headers=headers, timeout=10)
        res.encoding = 'EUC-JP'
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            tables = soup.find_all('table')
            for idx, t in enumerate(tables):
                dfs = pd.read_html(StringIO(str(t)))
                if not dfs:
                    continue
                df = dfs[0]
                df.columns = [str(c).strip() for c in df.columns]
                if len(df) > 0 and df.iloc[0, 0] == '累計':
                    if len(df.columns) > 11:
                        def parse_percent(val_str):
                            if not val_str or not isinstance(val_str, str):
                                return 0.0
                            m = re.search(r'([\d.]+)', val_str)
                            if m:
                                return float(m.group(1)) / 100.0
                            return 0.0
                        
                        stats['jockey_win_rate'] = parse_percent(str(df.iloc[0, 9]))
                        stats['jockey_place_rate'] = parse_percent(str(df.iloc[0, 11]))
                    break
            
            _jockey_cache[jockey_id_str] = stats
            save_json_cache(_jockey_cache, JOCKEY_CACHE_PATH)
            
    except Exception as e:
        print(f"[ERROR horse_scraping get_jockey_stats for ID={jockey_id_str}]: {e}")
        
    return stats

def get_trainer_stats(trainer_id: str) -> dict:
    global _trainer_cache
    if _trainer_cache is None:
        _trainer_cache = load_json_cache(TRAINER_CACHE_PATH)
        
    stats = {
        'trainer_win_rate': 0.0,
        'trainer_place_rate': 0.0
    }
    
    if not trainer_id or pd.isna(trainer_id) or str(trainer_id).strip() == "" or str(trainer_id).strip() == "nan":
        return stats
        
    trainer_id_str = str(trainer_id).strip()
    if trainer_id_str in _trainer_cache:
        return _trainer_cache[trainer_id_str]
        
    url = f"https://db.netkeiba.com/trainer/{trainer_id_str}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        time.sleep(1.0)
        res = requests.get(url, headers=headers, timeout=10)
        res.encoding = 'EUC-JP'
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            tables = soup.find_all('table')
            for idx, t in enumerate(tables):
                dfs = pd.read_html(StringIO(str(t)))
                if not dfs:
                    continue
                df = dfs[0]
                df.columns = [str(c).strip() for c in df.columns]
                if len(df) > 0 and df.iloc[0, 0] == '累計':
                    if len(df.columns) > 11:
                        def parse_percent(val_str):
                            if not val_str or not isinstance(val_str, str):
                                return 0.0
                            m = re.search(r'([\d.]+)', val_str)
                            if m:
                                return float(m.group(1)) / 100.0
                            return 0.0
                        
                        stats['trainer_win_rate'] = parse_percent(str(df.iloc[0, 9]))
                        stats['trainer_place_rate'] = parse_percent(str(df.iloc[0, 11]))
                    break
            
            _trainer_cache[trainer_id_str] = stats
            save_json_cache(_trainer_cache, TRAINER_CACHE_PATH)
            
    except Exception as e:
        print(f"[ERROR horse_scraping get_trainer_stats for ID={trainer_id_str}]: {e}")
        
    return stats

if __name__ == "__main__":
    test_horse_id = "2019105219" # イクイノックス
    test_race_id = "202305020411"  # 日本ダービー (2023年)
    print(f"--- Scraping horse past features for ID: {test_horse_id} before race {test_race_id} ---")
    res = get_horse_past_features(test_horse_id, test_race_id)
    print("Calculated Past Features:", res)
    
    print("\n--- Testing Jockey/Trainer Stats scraping ---")
    print("Jockey 01096:", get_jockey_stats("01096"))
    print("Trainer 01167:", get_trainer_stats("01167"))
