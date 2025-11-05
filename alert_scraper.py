import os
import json
from gdacs.api import GDACSAPIReader
from tqdm import tqdm
from datetime import datetime, timezone

OUTPUT_FILE = 'gdacs_alerts.json'

def run_alert_scraper():
    print("Starting GDACS alert scraping process using GDACSAPIReader...")
    
    try:
        client = GDACSAPIReader()
        events_object = client.latest_events(limit=50) 
    except Exception as e:
        print(f"Error connecting to GDACS API: {e}")
        return

    if not hasattr(events_object, 'features') or not events_object.features:
        print("No features found in GDACS response.")
        # Write an empty list to the file so the build doesn't break
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        return

    print(f"Fetched {len(events_object.features)} latest events from GDACS.")
    
    all_alerts = []
    # Use tqdm for a progress bar
    for event in tqdm(events_object.features, desc="Processing events"):
        
        # The event object is a 'feature' with a 'properties' dict
        # We must parse it just like you do in app2.py
        props = {}
        try:
            if isinstance(event, dict):
                props = event.get('properties', {})
            elif hasattr(event, 'properties') and event.properties is not None:
                props = event.properties
            
            if not props:
                print(f"Skipping event, could not parse properties: {event}")
                continue
        except Exception as e:
            print(f"Error parsing one event: {e}")
            continue 

        # Now, safely get all values from the 'props' DICTIONARY
        # This matches the logic from app2.py!
        
        # Create a stable guid
        guid_str = props.get('guid', props.get('name', '')) + props.get('fromdate', '')
        guid = hash(guid_str) # Simple hash for a unique ID
        
        alert = {
            "guid": guid,
            "title": props.get('name', 'No Title'),
            "link": props.get('url', {}).get('report', 'https.www.gdacs.org'),
            "published": props.get('fromdate', datetime.now(timezone.utc).isoformat()),
            "summary": props.get('description', 'No details provided.'),
            "event_type": props.get('eventtype', 'Unknown'),
            "alert_level": props.get('alertlevel', 'Unknown'),
            "country": props.get('country', 'Unknown'),
            "severity": props.get('severity', 'Unknown'),
        }
        
        # Create the 'full_text' for embedding
        alert['full_text'] = (
            f"GDACS Alert: {alert['title']}\n"
            f"Event Type: {alert['event_type']}\n"
            f"Alert Level: {alert['alert_level']}\n"
            f"Country: {alert['country']}\n"
            f"Details: {alert['summary']}"
        )
        
        all_alerts.append(alert)

    # OVERWRITE the file with these latest alerts
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_alerts, f, ensure_ascii=False, indent=4)

    print(f"Successfully OVERWROTE {OUTPUT_FILE} with {len(all_alerts)} alerts.")

if __name__ == "__main__":
    run_alert_scraper()