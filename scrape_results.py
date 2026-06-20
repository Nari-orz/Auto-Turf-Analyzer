import time
import requests
import re
from bs4 import BeautifulSoup
import pandas as pd

def scrape_race_results(race_id: str) -> pd.DataFrame:
    """
    指定されたレースIDの結果テーブルをnetkeibaからスクレイピングしてDataFrameに変換する。
    """
    url = f"https://db.netkeiba.com/race/{race_id}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        # netkeibaはEUC-JPが使われているのでデコード指定
        response.encoding = 'EUC-JP'
        
        if response.status_code != 200:
            print(f"[Warning] Failed to fetch {race_id}. Status code: {response.status_code}")
            return pd.DataFrame()
            
        soup = BeautifulSoup(response.content, 'html.parser', from_encoding='EUC-JP')
        
        # db.netkeiba.com の結果テーブルは通常 class='race_table_01'
        table = soup.find('table', class_='race_table_01')
        if not table:
            table = soup.find('table', class_='race_table_old')
            
        if not table:
            print(f"[Warning] No table found for {race_id}")
            return pd.DataFrame()
            
        # ヘッダー行を解析して列の位置を特定する
        headers_row = table.find('tr')
        if not headers_row:
            return pd.DataFrame()
            
        cols = [th.text.strip() for th in headers_row.find_all('th')]
        
        # 必要な列のインデックスを取得（別名考慮）
        target_cols = {
            '着順': ['着順'],
            '馬番': ['馬番'],
            '馬名': ['馬名'],
            '性齢': ['性齢'],
            '斤量': ['斤量'],
            '騎手': ['騎手'],
            'タイム': ['タイム'],
            '上り': ['上り', '上がり', '上り3F'],
            '単勝': ['単勝', 'オッズ'],
            '人気': ['人気'],
            '馬体重': ['馬体重'],
            '調教師': ['調教師', '調教']
        }
        
        col_indices = {}
        for key, aliases in target_cols.items():
            idx = -1
            for alias in aliases:
                idx = next((i for i, c in enumerate(cols) if alias in c), -1)
                if idx != -1:
                    break
            col_indices[key] = idx
            
        # 必須カラムが見つからない場合はスキップ
        if -1 in col_indices.values():
            missing = [k for k, v in col_indices.items() if v == -1]
            print(f"[Warning] Missing required columns {missing} for {race_id}")
            return pd.DataFrame()
            
        data = []
        rows = table.find_all('tr')[1:]
        for row in rows:
            tds = row.find_all('td')
            if len(tds) < len(cols):
                continue
                
            row_data = {
                'race_id': race_id
            }
            
            for key, idx in col_indices.items():
                if idx == -1:
                    continue
                cell_text = tds[idx].text.strip()
                if key == '馬名':
                    # 馬名はaタグの中身を優先
                    a_tag = tds[idx].find('a')
                    if a_tag:
                        cell_text = a_tag.text.strip()
                        # href属性から10桁の馬IDを抽出してカラム追加
                        href = a_tag.get('href', '')
                        match = re.search(r'/horse/(\d{10})', href)
                        if match:
                            row_data['馬ID'] = match.group(1)
                        else:
                            row_data['馬ID'] = ""
                    else:
                        cell_text = " ".join(cell_text.split())
                        row_data['馬ID'] = ""
                elif key == '騎手':
                    # 騎手名もaタグの中身を優先し、IDを抽出
                    a_tag = tds[idx].find('a')
                    if a_tag:
                        cell_text = a_tag.text.strip()
                        href = a_tag.get('href', '')
                        match = re.search(r'(?:id=|/jockey/|/jockey/result/recent/)(\d{5})', href)
                        if match:
                            row_data['騎手ID'] = match.group(1)
                        else:
                            row_data['騎手ID'] = ""
                    else:
                        row_data['騎手ID'] = ""
                elif key == '調教師':
                    # 調教師名もaタグの中身を優先し、IDを抽出
                    a_tag = tds[idx].find('a')
                    if a_tag:
                        cell_text = a_tag.text.strip()
                        href = a_tag.get('href', '')
                        match = re.search(r'(?:id=|/trainer/|/trainer/result/recent/)(\d{5})', href)
                        if match:
                            row_data['調教師ID'] = match.group(1)
                        else:
                            row_data['調教師ID'] = ""
                    else:
                        row_data['調教師ID'] = ""
                row_data[key] = cell_text
                
            data.append(row_data)
            
        return pd.DataFrame(data)
        
    except Exception as e:
        print(f"[Error] Exception occurred for {race_id}: {e}")
        return pd.DataFrame()

def main():
    # 設定：2023年 東京競馬場 (05) 第1回開催 (01)
    year = "2023"
    venue = "05"
    kai = "01"
    
    # テスト用として第1日の1〜3レース、第2日の1〜3レースを巡回（合計6レース）
    days = ["01", "02"]
    races = [f"{r:02d}" for r in range(1, 4)]
    
    all_dfs = []
    
    total_races = len(days) * len(races)
    print(f"=== netkeiba 一括スクレイピングツール起動 ===")
    print(f"巡回対象: {year}年 東京競馬場 第1回開催 (1日目〜2日目、各1〜3レース、計{total_races}レース)")
    print("-" * 50)
    
    count = 0
    for day in days:
        for r in races:
            race_id = f"{year}{venue}{kai}{day}{r}"
            count += 1
            print(f"[{count}/{total_races}] レースID: {race_id} データを取得中...")
            
            df = scrape_race_results(race_id)
            if df is not None and not df.empty:
                all_dfs.append(df)
                print(f"   -> 成功： {len(df)}頭のデータを抽出しました。")
            else:
                print("   -> 失敗： データを取得できませんでした。")
                
            # サーバー負荷防止のため、マナーとして1秒以上のインターバル
            time.sleep(1.2)
            
    print("-" * 50)
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        # 「単勝」を「単勝オッズ」に分かりやすくリネーム
        final_df = final_df.rename(columns={'単勝': '単勝オッズ'})
        
        output_file = "race_data_test.csv"
        import os
        file_exists = os.path.exists(output_file)
        
        # 既存のCSVファイルがある場合はヘッダーなしで追記（append）、ない場合は新規作成（write）
        if file_exists:
            final_df.to_csv(output_file, mode='a', index=False, header=False, encoding='utf-8-sig')
            print("[SUCCESS] スクレイピング完了！")
            print(f"既存の '{output_file}' に {len(final_df)}行を追記しました。")
        else:
            final_df.to_csv(output_file, mode='w', index=False, header=True, encoding='utf-8-sig')
            print("[SUCCESS] スクレイピング完了！")
            print(f"新規に '{output_file}' を作成し、{len(final_df)}行を保存しました。")
    else:
        print("[FAILED] スクレイピング結果が空だったため、CSVの保存を行いませんでした。")

if __name__ == "__main__":
    main()
