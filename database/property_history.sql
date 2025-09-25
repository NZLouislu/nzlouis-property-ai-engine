-- 1. Query to Retrieve Data from the properties Table


SELECT *
FROM properties
WHERE suburb = 'Khandallah';

SELECT COUNT(*) AS total_records
FROM properties
WHERE suburb = 'Khandallah';


SELECT p.*, ph.event_description, ph.event_date
FROM properties p
       LEFT JOIN property_history ph ON p.id = ph.property_id
WHERE p.suburb = 'Khandallah';


--- 查看并delete 重复的数据

SELECT address, COUNT(*) AS duplicate_count
FROM properties
GROUP BY address
HAVING COUNT(*) > 1;

----View the Duplicate Records

SELECT *
FROM properties
WHERE address IN (
    SELECT address
FROM properties
GROUP BY address
HAVING COUNT(*) > 1
)
ORDER BY address;

--- Delete the Duplicate Records

DELETE FROM properties
WHERE id NOT IN (
    SELECT MIN(id)
FROM properties
GROUP BY address
);


---Total Number of Duplicate Records

SELECT COUNT(*) AS total_duplicate_records
FROM properties
WHERE address IN (
    SELECT address
FROM properties
GROUP BY address
HAVING COUNT(*) > 1
);


----Number of Times Each Address is Duplicated

SELECT address, COUNT(*) AS duplicate_count
FROM properties
GROUP BY address
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC;

--------添加 property_history

SELECT 
    p.id AS property_id,
    p.address,
    p.suburb,
    p.city,
    p.postcode,
    p.year_built,
    p.bedrooms,
    p.bathrooms,
    p.car_spaces,
    p.floor_size,
    p.land_area,
    p.last_sold_price,
    p.last_sold_date,
    p.capital_value,
    p.land_value,
    p.improvement_value,
    p.has_rental_history,
    p.is_currently_rented,
    p.status,
    STRING_AGG(
        CONCAT(ph.event_date, ': ', ph.event_description, ' (', ph.interval_since_last_event, ')'), 
        '; ' ORDER BY ph.event_date
    ) AS property_history
FROM 
    properties p
LEFT JOIN 
    property_history ph ON p.id = ph.property_id
GROUP BY 
    p.id, p.address, p.suburb, p.city, p.postcode, p.year_built, p.bedrooms, p.bathrooms, 
    p.car_spaces, p.floor_size, p.land_area, p.last_sold_price, p.last_sold_date, 
    p.capital_value, p.land_value, p.improvement_value, p.has_rental_history, 
    p.is_currently_rented, p.status
ORDER BY 
    p.address;



-----主表与预测表信息
SELECT
    p.address,
    p.suburb,
    p.city,
    ps.predicted_status,
    ps.confidence_score,
    p.last_sold_price,
    p.last_sold_date,
    p.property_history,
    p.year_built,
    p.bedrooms,
    p.bathrooms,
    p.car_spaces,
    p.floor_size,
    p.land_area,
    p.capital_value,
    p.land_value,
    p.improvement_value,
    p.has_rental_history,
    p.is_currently_rented,
    ps.predicted_at
FROM
    properties p
    LEFT JOIN property_status ps 
        ON p.id = ps.property_id
        AND ps.predicted_at = (
            SELECT MAX(ps_inner.predicted_at)
            FROM property_status ps_inner
            WHERE ps_inner.property_id = p.id
        )
WHERE 
    p.status IS NULL
    AND ps.predicted_status IS NOT NULL
    AND ps.confidence_score > 0.8
ORDER BY 
    ps.confidence_score DESC;

-- 清空 property_status 表的数据，
DELETE FROM property_status;



-------查询 2张表的数据 1对多 

SELECT 
    p.id AS property_id,
    p.address,
    p.suburb,
    p.city,
    p.postcode,
    p.year_built,
    p.bedrooms,
    p.bathrooms,
    p.car_spaces,
    p.floor_size,
    p.land_area,
    p.last_sold_price,
    p.last_sold_date,
    p.capital_value,
    p.land_value,
    p.improvement_value,
    p.has_rental_history,
    p.is_currently_rented,
    p.status,
    STRING_AGG(
        CONCAT(ph.event_date, ': ', ph.event_description, ' (', ph.interval_since_last_event, ')'), 
        '; ' ORDER BY ph.event_date
    ) AS property_history
FROM 
    properties p
LEFT JOIN 
    property_history ph ON p.id = ph.property_id
GROUP BY 
    p.id, p.address, p.suburb, p.city, p.postcode, p.year_built, p.bedrooms, p.bathrooms, 
    p.car_spaces, p.floor_size, p.land_area, p.last_sold_price, p.last_sold_date, 
    p.capital_value, p.land_value, p.improvement_value, p.has_rental_history, 
    p.is_currently_rented, p.status
ORDER BY 
    p.address;


---------添加 property_history 字段
ALTER TABLE properties
ADD COLUMN property_history TEXT;

-----更新 property_history 字段的数据
UPDATE properties p
SET property_history = subquery.property_history
FROM (
    SELECT 
        p.id AS property_id,
        STRING_AGG(
            CONCAT(ph.event_date, ': ', ph.event_description, ' (', ph.interval_since_last_event, ')'), 
            '; ' ORDER BY ph.event_date
        ) AS property_history
    FROM 
        properties p
    LEFT JOIN 
        property_history ph ON p.id = ph.property_id
    GROUP BY 
        p.id
) AS subquery
WHERE p.id = subquery.property_id;



-----------修改address 的 status 为  for sell,  recentely sold 

UPDATE properties
SET status = 'for sell'
WHERE address IN (
    '8 Elgin Way, Khandallah, 6035',
    '29 Lucknow Terrace, Khandallah, 6035',
    '1 Izard Road, Khandallah, 6035',
    '2/19 Onslow Road, Khandallah, 6035',
    '19 Torwood Road, Khandallah, 6035',
    '39a Amapur Drive, Khandallah, 6035',
    '3/18 Agra Crescent, Khandallah, 6035',
    '42 Amapur Drive, Khandallah, 6035',
    '104 Nicholson Road, Khandallah, 6035',
    '8/75 Delhi Crescent, Khandallah, 6035',
    '93a Jubilee Road, Khandallah, 6035',
    '19 Benares Street, Khandallah, 6035',
    '17 Benares Street, Khandallah, 6035',
    '49 Jubilee Road, Khandallah, 6035',
    '18a Lucknow Terrace, Khandallah, 6035',
    '41a Clutha Avenue, Khandallah, 6035',
    '20 Amapur Drive, Khandallah, 6035',
    '26c Mysore Street, Khandallah, 6035',
    '49 Calcutta Street, Khandallah, 6035',
    '8 Simla Crescent, Khandallah, 6035',
    '48 Clark Street, Khandallah, 6035',
    '48 Amritsar Street, Khandallah, 6035',
    '5 Kabul Street, Khandallah, 6035',
    '13 Kapil Grove, Khandallah, 6035',
    '51a Mandalay Terrace, Khandallah, 6035',
    '34 Delhi Crescent, Khandallah, 6035',
    '37 Jubilee Road, Khandallah, 6035',
    '84a Madras Street, Khandallah, 6035',
    '118a Nicholson Road, Khandallah, 6035',
    '9b Onslow Road, Khandallah, 6035',
    '17 Elgin Way, Khandallah, 6035',
    '14 Karachi Crescent, Khandallah, 6035'
);




UPDATE properties
SET status = 'recently sold'
WHERE address IN (
    '38 Cashmere Avenue, Khandallah, 6035',
    '65a Homebush Road, Khandallah, 6035',
    '27 Satara Crescent, Khandallah, 6035',
    '45 Box Hill, Khandallah, 6035',
    '27 Amritsar Street, Khandallah, 6035',
    '52 Nicholson Road, Khandallah, 6035',
    '20 Imran Terrace, Khandallah, 6035',
    '5 Mysore Street, Khandallah, 6035',
    '42a Nicholson Road, Khandallah, 6035',
    '71 Cashmere Avenue, Khandallah, 6035',
    '37 Nicholson Road, Khandallah, 6035',
    '37 Woodmancote Road, Khandallah, 6035',
    '33 Clark Street, Khandallah, 6035',
    '90a Khandallah Road, Khandallah, 6035',
    '1/72 Homebush Road, Khandallah, 6035',
    '2/10 Box Hill, Khandallah, 6035',
    '5 Shastri Terrace, Khandallah, 6035',
    '3 Ramphal Terrace, Khandallah, 6035',
    '11a Himalaya Crescent, Khandallah, 6035',
    '36a Mandalay Terrace, Khandallah, 6035',
    '17/2 Onslow Road, Khandallah, 6035',
    '16 Jubilee Road, Khandallah, 6035',
    '2a Kabul Street, Khandallah, 6035',
    '23 Ngatoto Street, Khandallah, 6035',
    '17 Raumati Terrace, Khandallah, 6035',
    '62a Lohia Street, Khandallah, 6035',
    '12 Kapil Grove, Khandallah, 6035',
    '54 Amritsar Street, Khandallah, 6035',
    '58 Amritsar Street, Khandallah, 6035',
    '17f Onslow Road, Khandallah, 6035',
    '1b Shalimar Crescent, Khandallah, 6035',
    '39 Amritsar Street, Khandallah, 6035',
    '2/8 Narbada Crescent, Khandallah, 6035',
    '5 Elgin Way, Khandallah, 6035',
    '9 Indira Place, Khandallah, 6035'
);

UPDATE properties
SET status = 'recently sold'
WHERE address IN (
    '18 Jubilee Road, Khandallah, 6035',
    '1/104 Madras Street, Khandallah, 6035',
    '67a Cashmere Avenue, Khandallah, 6035',
    '21 Clark Street, Khandallah, 6035',
    '5 Mamaku Grove, Khandallah, 6035',
    '2 Lakshmi Place, Khandallah, 6035',
    '29 Satara Crescent, Khandallah, 6035',
    '50-52 Rama Crescent, Khandallah, 6035',
    '39 Lucknow Terrace, Khandallah, 6035',
    '61a Cashmere Avenue, Khandallah, 6035',
    '9 Homebush Road, Khandallah, 6035',
    '15 Swansea Street, Khandallah, 6035',
    '42 Jubilee Road, Khandallah, 6035',
    '7 Nepal Place, Khandallah, 6035',
    '83 Madras Street, Khandallah, 6035',
    '73a Jubilee Road, Khandallah, 6035',
    '14 Raumati Terrace, Khandallah, 6035',
    '24 Amapur Drive, Khandallah, 6035',
    '52 Amritsar Street, Khandallah, 6035',
    '31a Mandalay Terrace, Khandallah, 6035',
    '5/50 Mandalay Terrace, Khandallah, 6035',
    '62 Jubilee Road, Khandallah, 6035',
    '2/15a Box Hill, Khandallah, 6035',
    '28b Punjab Street, Khandallah, 6035',
    '16 Bengal Street, Khandallah, 6035',
    '35 Jubilee Road, Khandallah, 6035',
    '7 Mathieson Avenue, Khandallah, 6035',
    '2 Poona Street, Khandallah, 6035',
    '113 Calcutta Street, Khandallah, 6035',
    '23 Nicholson Road, Khandallah, 6035',
    '96 Cashmere Avenue, Khandallah, 6035',
    '65 Delhi Crescent, Khandallah, 6035',
    '11 Elgin Way, Khandallah, 6035',
    '18b Benares Street, Khandallah, 6035',
    '50 Amapur Drive, Khandallah, 6035',
    '2/50 Mandalay Terrace, Khandallah, 6035',
    '43a Mandalay Terrace, Khandallah, 6035',
    '29 Ranui Crescent, Khandallah, 6035',
    '100b Homebush Road, Khandallah, 6035',
    '120b Nicholson Road, Khandallah, 6035',
    '40 Calcutta Street, Khandallah, 6035',
    '16 Lochiel Road, Khandallah, 6035',
    '5a Chella Way, Khandallah, 6035',
    '4 Andaman Grove, Khandallah, 6035',
    '23 Everest Street, Khandallah, 6035',
    '12 Clutha Avenue, Khandallah, 6035',
    '6 Gurkha Crescent, Khandallah, 6035',
    '37c Gurkha Crescent, Khandallah, 6035',
    '11 Himalaya Crescent, Khandallah, 6035',
    '15 Homebush Road, Khandallah, 6035',
    '46 Madras Street, Khandallah, 6035',
    '6 Amapur Drive, Khandallah, 6035',
    '24a Mysore Street, Khandallah, 6035',
    '9 Amritsar Street, Khandallah, 6035',
    '44 Baroda Street, Khandallah, 6035',
    '45a Satara Crescent, Khandallah, 6035',
    '78a Jubilee Road, Khandallah, 6035',
    '14a Indira Place, Khandallah, 6035',
    '18 Omar Street, Khandallah, 6035',
    '5/10 Box Hill, Khandallah, 6035',
    '13 Kapil Grove, Khandallah, 6035',
    '17 Torwood Road, Khandallah, 6035',
    '65 Nicholson Road, Khandallah, 6035',
    '47a Satara Crescent, Khandallah, 6035',
    '22 Lucknow Terrace, Khandallah, 6035',
    '2 Harbour Park Terrace, Khandallah, 6035',
    '14 Waru Street, Khandallah, 6035',
    '2/23 Box Hill, Khandallah, 6035',
    '5b Ranikhet Way, Khandallah, 6035',
    '24a Gurkha Crescent, Khandallah, 6035',
    '70a Delhi Crescent, Khandallah, 6035',
    '22 Izard Road, Khandallah, 6035',
    '11a Kamla Way, Khandallah, 6035',
    '17 Onslow Road, Khandallah, 6035',
    '4 Ravi Street, Khandallah, 6035',
    '31 Gurkha Crescent, Khandallah, 6035',
    '28 Rama Crescent, Khandallah, 6035',
    '1 Elgin Way, Khandallah, 6035'
);




SELECT *
FROM properties
WHERE address LIKE '%8 Elgin Way%';



----Count the number of records with status = 'for sell':

SELECT COUNT(*) AS total_for_sell
FROM properties
WHERE status = 'for sell';


----Retrieve the details of records with status = 'for sell':

SELECT *
FROM properties
WHERE status = 'for sell';










-- 2. Query to Retrieve Data from Both Tables (Using JOIN)

SELECT p.id AS property_id, p.address, p.city, p.postcode,
       ph.event_type, ph.event_date, ph.event_price, ph.agent_info
FROM properties p
       INNER JOIN property_history ph ON p.id = ph.property_id;

SELECT p.id AS property_id, p.address_line1, p.city, p.postcode,
       ph.event_type, ph.event_date, ph.event_price, ph.agent_info
FROM properties p
       LEFT JOIN property_history ph ON p.id = ph.property_id;

-- 3. Filtered Query for Properties Currently Rented or Sold

SELECT p.id AS property_id, p.address_line1, p.city, p.postcode,
       p.is_currently_rented, p.is_currently_sold, ph.event_type, ph.event_date
FROM properties p
       LEFT JOIN property_history ph ON p.id = ph.property_id
WHERE p.is_currently_rented = TRUE OR p.is_currently_sold = TRUE;


-----------1. 查询重复的 address 数量
SELECT 
    address,
    COUNT(*) AS duplicate_count
FROM 
    real_estate_duplicate
GROUP BY 
    address
HAVING 
    COUNT(*) > 1;

---------2. 更新并删除重复的数据
DELETE FROM 
    real_estate_duplicate
WHERE 
    id NOT IN (
        SELECT MIN(id)
        FROM real_estate_duplicate
        GROUP BY address
    );








