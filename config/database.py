from .supabase_config import create_supabase_client

def get_supabase_client():
    """Get Supabase client instance"""
    return create_supabase_client()