-- Reset property history migration status
-- This script resets the migration status after fixing the SQL syntax error

-- Reset the migration status to 'idle' so it can run again
UPDATE scraping_progress 
SET status = 'idle', 
    updated_at = NOW()
WHERE id = 6;

-- Optionally, reset the last_processed_id to start from beginning
-- Uncomment the line below if you want to start migration from the beginning
-- UPDATE scraping_progress SET last_processed_id = NULL WHERE id = 6;

-- Check the current status
SELECT id, status, last_processed_id, updated_at 
FROM scraping_progress 
WHERE id = 6;