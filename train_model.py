import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier

from ml_utils import preprocess_data

def train_lightgbm_model(csv_path: str, model_save_path: str = "horse_racing_model.pkl") -> dict:
    """
    前処理したデータを使って、LightGBMで「1着になる確率（is_win）」を予測するモデルを学習・保存し、評価結果を返す。
    
    戻り値:
        dict: 学習結果メトリクスと可視化用のPlotly Fig
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")
        
    # 1. データの読み込みと前処理
    df_raw = pd.read_csv(csv_path)
    
    # 過去実績特徴量の動的取得 (馬IDごとに get_horse_past_features を呼び出し)
    # サーバー負荷防止のため1秒ディレイを挟みつつユニークな馬IDのデータを収集
    if '馬ID' in df_raw.columns:
        print("[INFO] Fetching horse past features from netkeiba... (1s delay per horse)")
        unique_horses = df_raw[['馬ID', 'race_id']].dropna().drop_duplicates()
        
        past_features_list = []
        from horse_scraping import get_horse_past_features
        total_horses = len(unique_horses)
        
        for idx, row in unique_horses.reset_index(drop=True).iterrows():
            h_id = str(row['馬ID']).split('.')[0]
            r_id = str(row['race_id'])
            print(f"   [{idx+1}/{total_horses}] Horse ID: {h_id} (Race ID: {r_id}) fetching past features...")
            
            try:
                # 馬ごとの過去特徴量をスクレイピング
                feats = get_horse_past_features(h_id, r_id)
            except Exception as e:
                print(f"馬ID {h_id} の過去データ取得をスキップしました (理由: {e})")
                feats = {
                    'prev_rank': np.nan,
                    'avg_up_3f_3runs': np.nan,
                    'win_rate': 0.0,
                    'prev_class': np.nan
                }
            feats['馬ID'] = row['馬ID']
            feats['race_id'] = row['race_id']
            past_features_list.append(feats)
            
        df_past_feats = pd.DataFrame(past_features_list)
        df_raw = pd.merge(df_raw, df_past_feats, on=['馬ID', 'race_id'], how='left')
        print("[SUCCESS] Horse past features mapping completed!")
    else:
        print("[WARNING] '馬ID' column not found, skipping past features integration.")

    # 騎手特徴量の動的取得
    if '騎手ID' in df_raw.columns:
        print("[INFO] Fetching jockey stats from netkeiba... (1s delay per jockey)")
        from horse_scraping import get_jockey_stats
        unique_jockeys = df_raw['騎手ID'].dropna().unique()
        jockey_stats_list = []
        for idx, j_id in enumerate(unique_jockeys):
            j_id_str = str(j_id).split('.')[0]
            print(f"   [{idx+1}/{len(unique_jockeys)}] Jockey ID: {j_id_str} fetching stats...")
            try:
                stats = get_jockey_stats(j_id_str).copy()
            except Exception as e:
                print(f"騎手ID {j_id_str} の過去データ取得をスキップしました (理由: {e})")
                stats = {
                    'jockey_win_rate': 0.0,
                    'jockey_place_rate': 0.0
                }
            stats['騎手ID'] = j_id
            jockey_stats_list.append(stats)
        df_jockey_stats = pd.DataFrame(jockey_stats_list)
        df_raw = pd.merge(df_raw, df_jockey_stats, on='騎手ID', how='left')
        print("[SUCCESS] Jockey stats mapping completed!")

    # 調教師特徴量の動的取得
    if '調教師ID' in df_raw.columns:
        print("[INFO] Fetching trainer stats from netkeiba... (1s delay per trainer)")
        from horse_scraping import get_trainer_stats
        unique_trainers = df_raw['調教師ID'].dropna().unique()
        trainer_stats_list = []
        for idx, t_id in enumerate(unique_trainers):
            t_id_str = str(t_id).split('.')[0]
            print(f"   [{idx+1}/{len(unique_trainers)}] Trainer ID: {t_id_str} fetching stats...")
            try:
                stats = get_trainer_stats(t_id_str).copy()
            except Exception as e:
                print(f"調教師ID {t_id_str} の過去データ取得をスキップしました (理由: {e})")
                stats = {
                    'trainer_win_rate': 0.0,
                    'trainer_place_rate': 0.0
                }
            stats['調教師ID'] = t_id
            trainer_stats_list.append(stats)
        df_trainer_stats = pd.DataFrame(trainer_stats_list)
        df_raw = pd.merge(df_raw, df_trainer_stats, on='調教師ID', how='left')
        print("[SUCCESS] Trainer stats mapping completed!")
        
    df_processed = preprocess_data(df_raw)
    
    # 特徴量 (X) と 目的変数 (y) の定義
    # 'race_id' と 'is_win' 以外のすべての列を特徴量とする
    X = df_processed.drop(columns=['race_id', 'is_win'], errors='ignore')
    y = df_processed['is_win']
    
    # 2. データの分割 (8割学習, 2割テスト)
    # 不均衡データ（1着の割合が少ない）ため、stratify=yで比率を一定に保つ
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. モデルの学習
    # データ数が少ないテスト環境（計96行）に配慮し、過学習を避ける適度なパラメータを設定
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=31,
        min_child_samples=2,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    # 4. 精度評価
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # テストデータで全クラスが0または1のみになる極端な少データ時のエラー回避
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.5  # クラスが1つしかない場合のフォールバック値
        
    # 5. 特徴量重要度の算出
    importances = model.feature_importances_
    df_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True) # Plotlyの横棒グラフ用に昇順ソート
    
    # 日本語特徴量ラベルの綺麗化マッピング
    rename_dict = {
        'age': '年齢',
        'sex_牡': '性別: 牡',
        'sex_牝': '性別: 牝',
        'sex_セ': '性別: セ',
        'horse_weight': '馬体重 (kg)',
        'weight_change': '体重増減',
        'time_seconds': '走破タイム (秒)',
        'up_3f': '上り3ハロン (秒)',
        'jockey_weight': '斤量 (kg)',
        'horse_number': '馬番',
        'prev_rank': '前走着順',
        'avg_up_3f_3runs': '過去3走平均上り3F',
        'win_rate': '通算勝率',
        'jockey_win_rate': '騎手勝率',
        'jockey_place_rate': '騎手複勝率',
        'trainer_win_rate': '調教師勝率',
        'trainer_place_rate': '調教師複勝率',
        'prev_class': '前走クラス',
        'odds': '単勝オッズ',
        'popularity': '人気順'
    }
    df_importance['Feature_JP'] = df_importance['Feature'].map(rename_dict).fillna(df_importance['Feature'])
    
    # Plotlyによる最高にプレミアムな棒グラフの作成
    fig = px.bar(
        df_importance,
        x='Importance',
        y='Feature_JP',
        orientation='h',
        title='AI Feature Importance',
        labels={'Importance': '重要スコア (Gini Importance)', 'Feature_JP': '特徴量名'},
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter, Outfit'),
        margin=dict(l=150, r=20, t=50, b=50),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(showgrid=False)
    )
    
    # 6. モデルの保存
    joblib.dump(model, model_save_path)
    
    # 結果サマリー
    results = {
        'accuracy': accuracy,
        'auc': auc,
        'model_path': model_save_path,
        'importance_df': df_importance,
        'importance_fig': fig,
        'features': X.columns.tolist()
    }
    
    return results

if __name__ == "__main__":
    # スタンドアロン実行用の検証コード
    csv_file = "race_data_test.csv"
    if os.path.exists(csv_file):
        print(f"--- Training Model with: {csv_file} ---")
        res = train_lightgbm_model(csv_file)
        print(f"[SUCCESS] Model trained and saved to: {res['model_path']}")
        print(f"Accuracy (Test Data): {res['accuracy']:.4f}")
        print(f"AUC (Test Data): {res['auc']:.4f}")
        print("\nFeature Importances:")
        print(res['importance_df'].sort_values(by='Importance', ascending=False).to_string(index=False))
    else:
        print(f"[ERROR] CSV file '{csv_file}' not found. Run scrape_results.py first!")
