def insert_property_history(supabase_client, property_id, history_data):
    # 插入历史记录到 Supabase
    for event in history_data:
        response = supabase_client.from_("property_history").insert({
            "property_id": property_id,
            "event_description": event['event_description'],
            "event_date": event['event_date'],
            "interval_since_last_event": event['event_interval']
        }).execute()

        if response.error:
            print(f"Error inserting property history: {response.error}")
        else:
            print(f"Inserted property history for property ID {property_id} on {event['event_date']}.")
