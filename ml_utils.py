import re
import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    取得したレースデータのDataFrameに対し、LightGBM等の機械学習モデルで学習できるように前処理を行う。
    
    前処理内容:
    1. 目的変数 'is_win' の作成 (着順=1なら1、それ以外は0)
    2. 不要な文字列列・リーク列（馬名, 騎手, 単勝オッズ, 元着順）の削除
    3. カテゴリ変数 '性齢' の処理 ('性別'をOne-Hotエンコーディングし、'年齢'を数値化)
    4. '馬体重' の処理 (馬体重本体と体重増減に正規表現で分離・数値化、欠損値補完)
    5. 'タイム' の処理 (分:秒形式を総秒数に変換、欠損値補完)
    6. '上り' およびその他数値列のクレンジングと欠損値平均補完
    """
    # 破壊的変更を防ぐためコピーを作成
    data = df.copy()
    
    # ---------------------------------------------
    # 1. 目的変数 is_win の作成
    # ---------------------------------------------
    if '着順' in data.columns:
        # 文字列にキャストして前後スペースを除去
        rank_str = data['着順'].astype(str).str.strip()
        # '1' または '01' のものを1、それ以外を0とする二値ターゲットを作成
        data['is_win'] = rank_str.apply(lambda x: 1 if x in ('1', '01') else 0)
    
    # ---------------------------------------------
    # 2. カテゴリ変数 '性齢' の処理
    # ---------------------------------------------
    if '性齢' in data.columns:
        # 性別（先頭1文字: 牡, 牝, セ）と 年齢（2文字目以降）を抽出
        data['sex'] = data['性齢'].astype(str).str[0]
        data['age'] = pd.to_numeric(data['性齢'].astype(str).str[1:], errors='coerce')
        
        # 年齢の欠損値を中央値（またはデフォルト3）で補完
        data['age'] = data['age'].fillna(data['age'].median() if not data['age'].isna().all() else 3.0)
        
        # 性別のOne-Hotエンコーディング (牡, 牝, セ)
        sex_dummies = pd.get_dummies(data['sex'], prefix='sex', dtype=float)
        # 生成されたダミー変数を結合
        data = pd.concat([data, sex_dummies], axis=1)
        
        # 切り出し元のカラムを削除
        data.drop(columns=['性齢', 'sex'], inplace=True, errors='ignore')
    
    # ---------------------------------------------
    # 3. 馬体重の分離・数値化
    # ---------------------------------------------
    if '馬体重' in data.columns:
        # 例: 444(-4) や 470(0) のような形式から正規表現で本体と増減を抽出
        def parse_weight(val):
            if pd.isna(val):
                return np.nan, np.nan
            val_str = str(val).strip()
            match = re.match(r'(\d+)\s*\(([-+]?\d+)\)', val_str)
            if match:
                return float(match.group(1)), float(match.group(2))
            # カッコがない数値のみの場合の考慮
            match_only_num = re.match(r'^(\d+)$', val_str)
            if match_only_num:
                return float(match_only_num.group(1)), 0.0
            return np.nan, np.nan

        parsed = data['馬体重'].apply(parse_weight)
        data['horse_weight'] = [p[0] for p in parsed]
        data['weight_change'] = [p[1] for p in parsed]
        
        # 体重の欠損値は平均値で補完
        data['horse_weight'] = data['horse_weight'].fillna(data['horse_weight'].mean() if not data['horse_weight'].isna().all() else 470.0)
        # 体重増減の欠損値は 0 で補完
        data['weight_change'] = data['weight_change'].fillna(0.0)
        
        # 元の馬体重カラムを削除
        data.drop(columns=['馬体重'], inplace=True, errors='ignore')

    # ---------------------------------------------
    # 4. その他出走前数値列のクレンジングと欠損値処理 (斤量, 馬番)
    # ---------------------------------------------
    if '斤量' in data.columns:
        data['jockey_weight'] = pd.to_numeric(data['斤量'], errors='coerce')
        data['jockey_weight'] = data['jockey_weight'].fillna(data['jockey_weight'].mean() if not data['jockey_weight'].isna().all() else 54.0)
        
    if '馬番' in data.columns:
        data['horse_number'] = pd.to_numeric(data['馬番'], errors='coerce')
        data['horse_number'] = data['horse_number'].fillna(1.0)

    # ---------------------------------------------
    # 5. 新規追加した過去実績特徴量の欠損値処理
    # ---------------------------------------------
    if 'prev_rank' in data.columns:
        data['prev_rank'] = pd.to_numeric(data['prev_rank'], errors='coerce')
        data['prev_rank'] = data['prev_rank'].fillna(data['prev_rank'].mean() if not data['prev_rank'].isna().all() else 8.0)
        
    if 'avg_up_3f_3runs' in data.columns:
        data['avg_up_3f_3runs'] = pd.to_numeric(data['avg_up_3f_3runs'], errors='coerce')
        data['avg_up_3f_3runs'] = data['avg_up_3f_3runs'].fillna(data['avg_up_3f_3runs'].mean() if not data['avg_up_3f_3runs'].isna().all() else 36.0)
        
    if 'win_rate' in data.columns:
        data['win_rate'] = pd.to_numeric(data['win_rate'], errors='coerce')
        data['win_rate'] = data['win_rate'].fillna(0.0)

    # ---------------------------------------------
    # 5.5 新規追加した騎手・調教師特徴量の欠損値処理
    # ---------------------------------------------
    if 'jockey_win_rate' in data.columns:
        data['jockey_win_rate'] = pd.to_numeric(data['jockey_win_rate'], errors='coerce')
        data['jockey_win_rate'] = data['jockey_win_rate'].fillna(data['jockey_win_rate'].mean() if not data['jockey_win_rate'].isna().all() else 0.05)
        
    if 'jockey_place_rate' in data.columns:
        data['jockey_place_rate'] = pd.to_numeric(data['jockey_place_rate'], errors='coerce')
        data['jockey_place_rate'] = data['jockey_place_rate'].fillna(data['jockey_place_rate'].mean() if not data['jockey_place_rate'].isna().all() else 0.15)
        
    if 'trainer_win_rate' in data.columns:
        data['trainer_win_rate'] = pd.to_numeric(data['trainer_win_rate'], errors='coerce')
        data['trainer_win_rate'] = data['trainer_win_rate'].fillna(data['trainer_win_rate'].mean() if not data['trainer_win_rate'].isna().all() else 0.07)
        
    if 'trainer_place_rate' in data.columns:
        data['trainer_place_rate'] = pd.to_numeric(data['trainer_place_rate'], errors='coerce')
        data['trainer_place_rate'] = data['trainer_place_rate'].fillna(data['trainer_place_rate'].mean() if not data['trainer_place_rate'].isna().all() else 0.20)

    # ---------------------------------------------
    # 5.6 新規追加したクラス・オッズ・人気特徴量の欠損値処理
    # ---------------------------------------------
    if '単勝オッズ' in data.columns:
        data['odds'] = pd.to_numeric(data['単勝オッズ'], errors='coerce')
        data['odds'] = data['odds'].fillna(data['odds'].mean() if not data['odds'].isna().all() else 10.0)
        
    if '人気' in data.columns:
        data['popularity'] = pd.to_numeric(data['人気'], errors='coerce')
        data['popularity'] = data['popularity'].fillna(data['popularity'].mean() if not data['popularity'].isna().all() else 8.0)
        
    if 'prev_class' in data.columns:
        data['prev_class'] = pd.to_numeric(data['prev_class'], errors='coerce')
        data['prev_class'] = data['prev_class'].fillna(0.0)

    # ---------------------------------------------
    # 6. 不要なリーク列・事後データ・文字列列の削除
    # ---------------------------------------------
    # 予測時に取得できない未来の事後データ（タイム, 上り, 着差等）やリーク列、文字列列を完全にドロップ
    drop_cols = ['馬名', '騎手', '騎手ID', '調教師', '調教師ID', '単勝オッズ', '人気', '着順', 'タイム', '上り', '着差', 'time_seconds', 'up_3f', '斤量', '馬番', '馬ID']
    data.drop(columns=drop_cols, inplace=True, errors='ignore')

    # float型への最終的なクレンジング
    for col in data.columns:
        if col != 'race_id' and col != 'is_win':
            data[col] = pd.to_numeric(data[col], errors='coerce').astype(float)
            
    # 全体的な最終欠損値補完
    data = data.fillna(0.0)

    return data
