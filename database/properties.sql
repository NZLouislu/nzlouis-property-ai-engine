
DROP TABLE IF EXISTS real_estate;
CREATE TABLE real_estate (
    id VARCHAR(32) PRIMARY KEY DEFAULT md5(random()::text || clock_timestamp()::text),
    address TEXT,
    status TEXT,
    data TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Drop the properties table if it exists
DROP TABLE IF EXISTS properties;

-- Create the properties table with a 32-character string ID
CREATE TABLE properties
(
    id VARCHAR(32) PRIMARY KEY DEFAULT md5(random()::text || clock_timestamp()::text),
    address TEXT,
    suburb TEXT,
    city TEXT,
    postcode TEXT,
    year_built INTEGER,
    bedrooms INTEGER,
    bathrooms INTEGER,
    car_spaces INTEGER,
    floor_size TEXT,
    land_area TEXT,
    last_sold_price DOUBLE PRECISION,
    last_sold_date DATE,
    capital_value DOUBLE PRECISION,
    land_value DOUBLE PRECISION,
    improvement_value DOUBLE PRECISION,
    has_rental_history BOOLEAN DEFAULT FALSE,
    is_currently_rented BOOLEAN DEFAULT FALSE,
    status TEXT
    -- New field for property status
);

-- Drop the property_history table if it exists
DROP TABLE IF EXISTS property_history;

-- Create the property_history table with a 32-character string property_id
CREATE TABLE property_history
(
    id VARCHAR(32) PRIMARY KEY DEFAULT md5(random()::text || clock_timestamp()::text),
    property_id VARCHAR(32) REFERENCES properties(id) ON DELETE CASCADE,
    event_description TEXT,
    event_date DATE NOT NULL,
    interval_since_last_event TEXT,
    UNIQUE(property_id, event_date, event_description)
);

DROP TABLE IF EXISTS property_status;
CREATE TABLE property_status (
    id VARCHAR(32) PRIMARY KEY DEFAULT md5(random()::text || clock_timestamp()::text),
    property_id VARCHAR(32),
    predicted_status VARCHAR(50),
    confidence_score DECIMAL(5, 4),  -- 置信度得分，范围0-1
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);






-- Insert data into properties table
INSERT INTO properties (address, suburb, city, postcode, year_built, bedrooms, bathrooms, car_spaces, 
                        floor_size, land_area, last_sold_price, last_sold_date, capital_value, 
                        land_value, improvement_value, has_rental_history, is_currently_rented, status)
VALUES
    ('23 Calcutta Street, Khandallah, 6035', 'Khandallah', 'Wellington', '6035', 1940, 4, 1, 2, 
     '140 m2', '506 m2', NULL, NULL, 1320000, 1000000, 320000, TRUE, TRUE, 'For Sale'),
     
    ('24 Calcutta Street, Khandallah, 6035', 'Khandallah', 'Wellington', '6035', 1960, 5, 2, 2, 
     '250 m2', '504 m2', NULL, NULL, NULL, NULL, NULL, FALSE, FALSE, 'Not For Sale'),
     
    ('1/24 Calcutta Street, Khandallah, 6035', 'Khandallah', 'Wellington', '6035', 1969, 2, 1, 1, 
     '130 m2', NULL, 481467, '2021-11-25', 790000, 660000, 130000, FALSE, FALSE, 'Sold'),
     
    ('2/24 Calcutta Street, Khandallah, 6035', 'Khandallah', 'Wellington', '6035', 1969, 2, 1, 1, 
     '120 m2', NULL, 656000, '2018-12-07', 930000, 690000, 240000, FALSE, FALSE, 'Sold'),
     
    ('24a Calcutta Street, Khandallah, 6035', 'Khandallah', 'Wellington', '6035', 1958, 3, 1, 1, 
     '260 m2', '556 m2', 740000, '2016-09-15', 1770000, 990000, 780000, FALSE, FALSE, 'For Sale');

-- Insert data into property_history table
INSERT INTO property_history
    (property_id, event_description, event_date, interval_since_last_event)
VALUES
    ((SELECT id FROM properties WHERE address = '23 Calcutta Street, Khandallah, 6035'), 'Listed for Rent at $850', '2024-09-24', ''),
    ((SELECT id FROM properties WHERE address = '23 Calcutta Street, Khandallah, 6035'), 'Listed for Rent at $850', '2022-08-07', ''),
    ((SELECT id FROM properties WHERE address = '23 Calcutta Street, Khandallah, 6035'), 'Sold for $514,000', '2012-05-15', '12 years 4 months 25 days'),
    ((SELECT id FROM properties WHERE address = '23 Calcutta Street, Khandallah, 6035'), 'Property Built', '1940-01-01', ''),
    ((SELECT id FROM properties WHERE address = '24 Calcutta Street, Khandallah, 6035'), 'Tender — Price Not Disclosed', '2016-09-06', ''),
    ((SELECT id FROM properties WHERE address = '24 Calcutta Street, Khandallah, 6035'), 'Property Built', '1960-01-01', ''),
    ((SELECT id FROM properties WHERE address = '1/24 Calcutta Street, Khandallah, 6035'), 'Sold for $481,467', '2021-11-25', '2 years 10 months 15 days'),
    ((SELECT id FROM properties WHERE address = '1/24 Calcutta Street, Khandallah, 6035'), 'Sold for $400,000', '2016-09-23', '8 years 17 days'),
    ((SELECT id FROM properties WHERE address = '1/24 Calcutta Street, Khandallah, 6035'), 'Sold for $250,000', '2006-10-16', '17 years 11 months 24 days'),
    ((SELECT id FROM properties WHERE address = '1/24 Calcutta Street, Khandallah, 6035'), 'Sold for $307,500', '1997-10-29', '26 years 11 months 11 days'),
    ((SELECT id FROM properties WHERE address = '1/24 Calcutta Street, Khandallah, 6035'), 'Property Built', '1969-01-01', ''),
    ((SELECT id FROM properties WHERE address = '2/24 Calcutta Street, Khandallah, 6035'), 'Sold for $656,000', '2018-12-07', '5 years 10 months 3 days'),
    ((SELECT id FROM properties WHERE address = '2/24 Calcutta Street, Khandallah, 6035'), 'Asking Price — Price Not Disclosed', '2018-11-21', ''),
    ((SELECT id FROM properties WHERE address = '2/24 Calcutta Street, Khandallah, 6035'), 'Sold for $430,000', '2016-09-23', '8 years 17 days'),
    ((SELECT id FROM properties WHERE address = '2/24 Calcutta Street, Khandallah, 6035'), 'Sold for $300,000', '2006-10-16', '17 years 11 months 24 days'),
    ((SELECT id FROM properties WHERE address = '2/24 Calcutta Street, Khandallah, 6035'), 'Sold for $307,500', '1997-10-29', '26 years 11 months 11 days'),
    ((SELECT id FROM properties WHERE address = '2/24 Calcutta Street, Khandallah, 6035'), 'Property Built', '1969-01-01', ''),
    ((SELECT id FROM properties WHERE address = '24a Calcutta Street, Khandallah, 6035'), 'Sold for $740,000', '2016-09-15', '8 years 25 days'),
    ((SELECT id FROM properties WHERE address = '24a Calcutta Street, Khandallah, 6035'), 'Deadline Private Treaty — $670,000', '2011-06-04', ''),
    ((SELECT id FROM properties WHERE address = '24a Calcutta Street, Khandallah, 6035'), 'Property Built', '1958-01-01', '');


-- 更新 properties 表中的 status 字段，如果 real_estate 表中存在对应的 address 并且 properties 的 status 为 NULL，则将其改为 'for sell'
UPDATE properties
SET status = 'for sell'
FROM real_estate
WHERE
    -- 去掉 properties.address 的邮编部分进行匹配
    TRIM(BOTH ',' FROM SPLIT_PART(properties.address, ',', 1) || ', ' || SPLIT_PART(properties.address, ',', 2)) = real_estate.address
    AND properties.status IS NULL;

-- 查询 properties 表中 status 为 'for sell' 的记录数量和具体数据
SELECT COUNT(*) AS for_sell_count
FROM properties
WHERE status = 'for sell';

-- 查询具体的记录
SELECT *
FROM properties
WHERE status = 'for sell';


-- 更新 properties 表中的 status
UPDATE properties
SET status = 'for sell'
WHERE status IS NULL
  AND EXISTS (
    SELECT 1
    FROM real_estate
    WHERE 
      -- 移除邮政编码并标准化地址
      LOWER(TRIM(REGEXP_REPLACE(properties.address, ',\s*\d{4}$', ''))) =
      LOWER(TRIM(real_estate.address))
      OR
      -- 处理街道号码可能的差异（例如 "2/25" vs "25"）
      LOWER(TRIM(REGEXP_REPLACE(REGEXP_REPLACE(properties.address, ',\s*\d{4}$', ''), '^(\d+/)?', ''))) =
      LOWER(TRIM(REGEXP_REPLACE(real_estate.address, '^(\d+/)?', '')))
      OR
      -- 处理可能的额外信息（例如 "Quest Wellington Serviced Apart,"）
      LOWER(TRIM(REGEXP_REPLACE(properties.address, '^.*?,\s*(.*?),\s*\d{4}$', '\1'))) =
      LOWER(TRIM(real_estate.address))
  );

-- 查看更新的结果
SELECT 
    p.id AS property_id, 
    p.address AS property_address, 
    p.status AS property_status,
    r.address AS real_estate_address
FROM 
    properties p
JOIN 
    real_estate r ON (
        LOWER(TRIM(REGEXP_REPLACE(p.address, ',\s*\d{4}$', ''))) = LOWER(TRIM(r.address))
        OR
        LOWER(TRIM(REGEXP_REPLACE(REGEXP_REPLACE(p.address, ',\s*\d{4}$', ''), '^(\d+/)?', ''))) = LOWER(TRIM(REGEXP_REPLACE(r.address, '^(\d+/)?', '')))
        OR
        LOWER(TRIM(REGEXP_REPLACE(p.address, '^.*?,\s*(.*?),\s*\d{4}$', '\1'))) = LOWER(TRIM(r.address))
    )
WHERE 
    p.status = 'for sell';

