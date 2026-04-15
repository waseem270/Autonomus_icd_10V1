-- ICD-10 Table Initialization Script
-- Purpose: Stores the master list of ICD-10-CM codes for lookup and full-text search.

-- 1. Drop existing table for clean reload
DROP TABLE IF EXISTS icd10_codes;

-- 2. Create the table
CREATE TABLE icd10_codes (
    id SERIAL PRIMARY KEY,
    code VARCHAR(10) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 6. Add CHECK constraint that code length >= 3
    CONSTRAINT chk_code_length CHECK (LENGTH(code) >= 3)
);

-- 5. Add table comment
COMMENT ON TABLE icd10_codes IS 'Master database of ICD-10-CM codes and descriptions for medical coding.';

-- 4. Create regular B-tree index on code for fast lookup
-- Note: code column already has a unique index due to the UNIQUE constraint, 
-- but we explicitly define a B-tree search index for consistency.
CREATE INDEX idx_icd10_code_lookup ON icd10_codes USING btree (code);

-- 3. Create GIN index for full-text search on description
CREATE INDEX idx_icd10_description_fts ON icd10_codes 
USING GIN (to_tsvector('english', description));

-- Grant permissions (Optional, adjust as per your DB environment)
-- GRANT ALL PRIVILEGES ON TABLE icd10_codes TO your_user;
