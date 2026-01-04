import pandas as pd
import os
import sys
import time
import concurrent.futures

try:
    from opendata import OpenData
except ImportError:
    print("Error: Library 'goldencheetah-opendata' not found.")
    print("   Install it using command: pip install goldencheetah-opendata")
    sys.exit(1)

OUTPUT_FILE = "activities.csv"
MAX_WORKERS = 20
ATHLETE_LIMIT = 100

# Key metrics to extract (Raw Physiology & Mechanics)
METRICS_TO_FETCH = [
    'workout_time',
    'total_distance',
    'elevation_gain',
    'average_speed',
    'average_hr',
    'average_cad',
    'average_run_cad',
    'aerobic_decoupling',  # Endurance metric (Drift)
    'athlete_weight',  # Key context!
    'trimp_points'
]


def process_athlete(athlete):
    athlete_activities = []
    try:
        athlete_meta = athlete.metadata.get('ATHLETE', {})
        yob = athlete_meta.get('yob')
        gender = athlete_meta.get('gender')

        for act in athlete.activities():
            meta = act.metadata
            sport = meta.get('sport', '')

            if sport not in ['Run', 'Running', 'Jogging', 'Street Run', 'Trail Run']:
                continue

            metrics = meta.get('METRICS', {})

            row = {
                'athlete_id': athlete.id,
                'date': meta.get('date'),
                'sport': sport,
                'gender': gender,
                'yob': yob,
            }

            for m in METRICS_TO_FETCH:
                val = metrics.get(m)
                if isinstance(val, list): val = val[0]
                row[m] = val

            run_cad = row.get('average_run_cad')
            gen_cad = row.get('average_cad')

            def safe_float(x):
                try:
                    return float(x)
                except:
                    return 0.0

            if safe_float(run_cad) > 0:
                row['final_cadence'] = run_cad
            else:
                row['final_cadence'] = gen_cad

            athlete_activities.append(row)

    except Exception:
        return []

    return athlete_activities


def ensure_data_exists():
    if os.path.exists(OUTPUT_FILE):
        print(f"File '{OUTPUT_FILE}' already exists. Using local version.")
        return

    print("Initializing OpenData library...")
    od = OpenData()
    all_activities = []

    print("Connecting to GoldenCheetah OpenData (AWS S3)...")

    try:
        athletes = list(od.remote_athletes())
        total_athletes = len(athletes)

        athletes_to_process = athletes[:ATHLETE_LIMIT]
        print(f"Found {total_athletes} remote athletes. Processing first {len(athletes_to_process)}...")

        start_time = time.time()
        processed_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_athlete = {executor.submit(process_athlete, a): a for a in athletes_to_process}

            for future in concurrent.futures.as_completed(future_to_athlete):
                processed_count += 1
                try:
                    data = future.result()
                    if data:
                        all_activities.extend(data)
                except Exception:
                    pass

                if processed_count % 10 == 0 or processed_count == len(athletes_to_process):
                    elapsed = time.time() - start_time
                    sys.stdout.write(
                        f"\r   Progress: {processed_count}/{len(athletes_to_process)} athletes | Found {len(all_activities)} runs")
                    sys.stdout.flush()

        print(f"\nDownload complete.")

        if all_activities:
            df = pd.DataFrame(all_activities)
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"Saved dataset to: {OUTPUT_FILE}")
        else:
            print("No running activities found.")

    except Exception as e:
        print(f"\nCritical Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    ensure_data_exists()