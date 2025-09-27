
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_openaq_city(city='Delhi', parameter='pm25', days=30, limit=10000):
    \"\"\"Fetch recent measurements for a city from OpenAQ v2 API.
    Returns a DataFrame with columns: datetime (UTC), value, unit, location\"\"\"
    base = 'https://api.openaq.org/v2/measurements'
    date_to = datetime.utcnow()
    date_from = date_to - timedelta(days=days)
    params = {
        'city': city,
        'parameter': parameter,
        'date_from': date_from.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'date_to': date_to.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'limit': 100,
        'page': 1,
        'sort': 'desc',
        'order_by': 'date'
    }
    rows = []
    while True:
        resp = requests.get(base, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get('results', [])
        if not results:
            break
        for r in results:
            rows.append({
                'datetime': r['date']['utc'],
                'value': r['value'],
                'unit': r.get('unit'),
                'location': r.get('location'),
                'city': r.get('city'),
                'parameter': r.get('parameter')
            })
        meta = data.get('meta', {})
        found = meta.get('found', 0)
        # paging
        params['page'] += 1
        if params['page']*params['limit'] > found or params['page'] > 100:
            break
        time.sleep(0.2)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    return df

if __name__ == '__main__':
    df = fetch_openaq_city('Delhi', 'pm25', days=60)
    print(df.head())
