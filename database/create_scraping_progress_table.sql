-- Create scraping_progress table if it doesn't exist
CREATE TABLE IF NOT EXISTS scraping_progress (
    id INTEGER PRIMARY KEY,
    last_processed_id TEXT,
    batch_size INTEGER DEFAULT 1000,
    status VARCHAR(20) DEFAULT 'idle',
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default record for image updater if it doesn't exist
INSERT INTO scraping_progress (id, last_processed_id, batch_size, status, updated_at)
VALUES (1, NULL, 1000, 'idle', NOW())
ON CONFLICT (id) DO NOTHING;