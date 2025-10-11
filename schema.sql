CREATE TABLE ddi_rules (
    id SERIAL PRIMARY KEY,
    drug_a_atc VARCHAR(10) NOT NULL,
    drug_b_atc VARCHAR(10) NOT NULL,
    severity VARCHAR(20) NOT NULL, -- 'Major', 'Moderate', 'Minor'
    mechanism TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ddxi_rules (
    id SERIAL PRIMARY KEY,
    drug_atc VARCHAR(10) NOT NULL,
    icd_code VARCHAR(10) NOT NULL,
    contraindication_level VARCHAR(20) NOT NULL, -- 'Absolute', 'Relative'
    statement TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE dose_rules (
    id SERIAL PRIMARY KEY,
    drug_atc VARCHAR(10) NOT NULL,
    population VARCHAR(50), -- 'adult', 'pediatric'
    route VARCHAR(50), -- 'oral', 'IV'
    dose_min NUMERIC,
    dose_max NUMERIC,
    dose_unit VARCHAR(50), -- 'mg/day', 'mg/kg/day'
    note TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE event_logs (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    list_atc TEXT[] NOT NULL,
    icd_list TEXT[] NOT NULL,
    alert_type VARCHAR(20) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    rule_id VARCHAR(50),
    doctor_action VARCHAR(20) NOT NULL,
    final_rx_outcome VARCHAR(20) NOT NULL,
    patient_age INT,
    patient_gender VARCHAR(10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);


CREATE INDEX idx_ddi_rules_drug_a ON ddi_rules(drug_a_atc);
CREATE INDEX idx_ddi_rules_drug_b ON ddi_rules(drug_b_atc);

CREATE INDEX idx_ddxi_rules_drug ON ddxi_rules(drug_atc);

CREATE INDEX idx_dose_rules_drug ON dose_rules(drug_atc);

CREATE INDEX idx_event_logs_created_at ON event_logs(created_at);