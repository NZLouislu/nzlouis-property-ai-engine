-- Add status column to scraping_progress table if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'scraping_progress' 
        AND column_name = 'status'
    ) THEN
        ALTER TABLE scraping_progress ADD COLUMN status VARCHAR(20) DEFAULT 'idle';
    END IF;
END $$;

-- Update existing records to have 'idle' status if they don't have one
UPDATE scraping_progress SET status = 'idle' WHERE status IS NULL;