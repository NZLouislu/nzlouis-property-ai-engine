-- Create scraping_progress table
CREATE TABLE IF NOT EXISTS scraping_progress (
    id SERIAL PRIMARY KEY,
    last_processed_id TEXT,
    batch_size INTEGER,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'idle'
);

-- Insert initial records for different scraping processes
-- ID 1: Property image updater (already exists)
INSERT INTO scraping_progress (id, last_processed_id, batch_size, updated_at, status)
SELECT 1, NULL, 1000, NOW(), 'idle'
WHERE NOT EXISTS (SELECT 1 FROM scraping_progress WHERE id = 1);

-- ID 2: Real estate Auckland
INSERT INTO scraping_progress (id, last_processed_id, batch_size, updated_at, status)
SELECT 2, NULL, 1000, NOW(), 'idle'
WHERE NOT EXISTS (SELECT 1 FROM scraping_progress WHERE id = 2);

-- ID 3: Real estate Wellington
INSERT INTO scraping_progress (id, last_processed_id, batch_size, updated_at, status)
SELECT 3, NULL, 1000, NOW(), 'idle'
WHERE NOT EXISTS (SELECT 1 FROM scraping_progress WHERE id = 3);

-- ID 4: Real estate Rent
INSERT INTO scraping_progress (id, last_processed_id, batch_size, updated_at, status)
SELECT 4, NULL, 1000, NOW(), 'idle'
WHERE NOT EXISTS (SELECT 1 FROM scraping_progress WHERE id = 4);

-- ID 5:  Wellington Property
INSERT INTO scraping_progress (id, last_processed_id, batch_size, updated_at, status)
SELECT 5, NULL, 1000, NOW(), 'idle'
WHERE NOT EXISTS (SELECT 1 FROM scraping_progress WHERE id = 5);

-- ID 6: Property History Migration
INSERT INTO scraping_progress (id, last_processed_id, batch_size, updated_at, status)
SELECT 6, NULL, 1000, NOW(), 'idle'
WHERE NOT EXISTS (SELECT 1 FROM scraping_progress WHERE id = 6);

-- Add comments to explain the purpose of each ID
/*
ID 1: Property image updater (main_image.py)
ID 2: Real estate Auckland data scraper (real_estate_auckland.py)
ID 3: Real estate Wellington data scraper (real_estate_wellington.py)
ID 4: Real estate Rent data scraper (real_estate_rent.py)
ID 5: PropertyValue Wellington data scraper (propertyvalue_wellington.py)
ID 6: Property history migration to property_history field (migrate_property_history.py)
*/