import pandas as pd
import json

def jsonify_data(data):
    json_string = json.dumps(data, ensure_ascii=False)
    return json_string

def restructure_fixed_charging_stations(df):
    rows = []
    for index, row in df.iterrows():
        for i in range(row['pile_count']):
            rows.append({
                'station_id': f"fcs_{row['station_id']}_{i+1}",
                'longitude': row['longitude'],
                'latitude': row['latitude'],
                'charging_queue':jsonify_data([]),
                'available': True
            })
    
    new_df = pd.DataFrame(rows)
    return new_df


if __name__ == "__main__":
    print(jsonify_data([]))
