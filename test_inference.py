import pandas as pd
import joblib
from app import fetch_netkeiba_shutuba
from ml_utils import preprocess_data
from horse_scraping import get_horse_past_features
import os
import time

def test_inference(race_id):
    print(f"Fetching shutuba for {race_id}...")
    df_scraped = fetch_netkeiba_shutuba(race_id)
    if df_scraped.empty:
        print("Empty dataframe returned from fetch_netkeiba_shutuba.")
        return
        
    print(df_scraped[['馬番', '馬名', '馬ID']].head())
    
    print("\nFetching past features...")
    past_feats = []
    for idx, row in df_scraped.iterrows():
        h_id = str(row['馬ID']).split('.')[0]
        try:
            feats = get_horse_past_features(h_id, race_id)
        except Exception as e:
            print(f"馬ID {h_id} の過去データ取得をスキップしました (理由: {e})")
            feats = {
                'prev_rank': np.nan,
                'avg_up_3f_3runs': np.nan,
                'win_rate': 0.0,
                'prev_class': np.nan
            }
        feats['馬ID'] = row['馬ID']
        past_feats.append(feats)
        time.sleep(1)
        
    df_past_feats = pd.DataFrame(past_feats)
    df_merged = pd.merge(df_scraped, df_past_feats, on='馬ID', how='left')
    
    # 騎手および調教師のスタッツを取得してマージ
    jockey_win_rates = []
    jockey_place_rates = []
    trainer_win_rates = []
    trainer_place_rates = []
    from horse_scraping import get_jockey_stats, get_trainer_stats
    
    for idx, row in df_merged.iterrows():
        j_id = str(row.get('騎手ID', '')).split('.')[0]
        t_id = str(row.get('調教師ID', '')).split('.')[0]
        
        try:
            j_s = get_jockey_stats(j_id).copy()
        except Exception as e:
            print(f"騎手ID {j_id} の過去データ取得をスキップしました (理由: {e})")
            j_s = {
                'jockey_win_rate': 0.0,
                'jockey_place_rate': 0.0
            }
            
        try:
            t_s = get_trainer_stats(t_id).copy()
        except Exception as e:
            print(f"調教師ID {t_id} の過去データ取得をスキップしました (理由: {e})")
            t_s = {
                'trainer_win_rate': 0.0,
                'trainer_place_rate': 0.0
            }
        
        jockey_win_rates.append(j_s['jockey_win_rate'])
        jockey_place_rates.append(j_s['jockey_place_rate'])
        trainer_win_rates.append(t_s['trainer_win_rate'])
        trainer_place_rates.append(t_s['trainer_place_rate'])
        
    df_merged['jockey_win_rate'] = jockey_win_rates
    df_merged['jockey_place_rate'] = jockey_place_rates
    df_merged['trainer_win_rate'] = trainer_win_rates
    df_merged['trainer_place_rate'] = trainer_place_rates
    
    print("\nMerged Data (Before Preprocess):")
    print(df_merged[['馬名', 'prev_rank', 'avg_up_3f_3runs', 'win_rate']].head())
    
    df_for_preprocess = df_merged.copy()
    if '着順' not in df_for_preprocess.columns:
        df_for_preprocess['着順'] = "0"
    if '馬体重' not in df_for_preprocess.columns:
        df_for_preprocess['馬体重'] = "470(0)"
        
    df_processed = preprocess_data(df_for_preprocess)
    
    model_path = "horse_racing_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("\n--- [DEBUG] Model Training Features ---")
        try:
            print(model.feature_name_)
            expected_features = model.feature_name_
        except AttributeError:
            expected_features = ['age', 'horse_weight', 'weight_change', 'jockey_weight', 'horse_number', 
                                 'sex_セ', 'sex_牝', 'sex_牡', 'prev_rank', 'avg_up_3f_3runs', 'win_rate',
                                 'jockey_win_rate', 'jockey_place_rate', 'trainer_win_rate', 'trainer_place_rate',
                                 'prev_class', 'odds', 'popularity']
            
        for col in expected_features:
            if col not in df_processed.columns:
                df_processed[col] = 0.0
                
        X_pred = df_processed[expected_features]
        print("\n--- [DEBUG] 推論用データフレーム (X_pred) ---")
        print(X_pred.head())
        print("-------------------------------------------\n")
        
        probs = model.predict_proba(X_pred)[:, 1]
        sum_prob = probs.sum()
        if sum_prob > 0:
            df_merged['probability'] = (probs / sum_prob) * 100.0
        else:
            df_merged['probability'] = 1.0 / len(df_merged) * 100.0
        
        print("\nPrediction Results:")
        print(df_merged[['馬番', '馬名', 'probability']].head(10))
    else:
        print("Model file not found.")

if __name__ == "__main__":
    test_inference("202605021011")
