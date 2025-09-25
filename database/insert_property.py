def insert_property(supabase_client, property_data):
    # 插入到 Supabase 的 properties 表中
    response = supabase_client.from_("properties").insert(property_data).execute()
    
    if response.error:
        print(f"Error inserting properties: {response.error}")
        return None
    else:
        property_id = response.data[0]['id']
        print(f"Inserted properties with ID {property_id}.")
        return property_id
