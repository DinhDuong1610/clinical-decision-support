import psycopg2
import redis
import json
from collections import defaultdict

POSTGRES_CONFIG = {
    "dbname": "cds_knowledge_base",
    "user": "cds_user",
    "password": "cds_password",
    "host": "localhost",
    "port": "5433"
}

REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0
}

def sync_ddi_rules(pg_cursor, redis_pipe):
    pg_cursor.execute("SELECT drug_a_atc, drug_b_atc, severity, mechanism FROM ddi_rules")

    for row in pg_cursor.fetchall():
        drug_a, drug_b, severity, mechanism = row
        interaction_data = json.dumps({"severity": severity, "mechanism": mechanism})

        redis_pipe.hset(f"ddi:{drug_a}", drug_b, interaction_data)
        redis_pipe.hset(f"ddi:{drug_b}", drug_a, interaction_data)

def sync_ddxi_rules(pg_cursor, redis_pipe):
    pg_cursor.execute("SELECT drug_atc, icd_code, contraindication_level, statement FROM ddxi_rules")

    rules_by_drug = defaultdict(dict)
    for row in pg_cursor.fetchall():
        drug_atc, icd_code, level, statement = row
        contraindication_data = json.dumps({"level": level, "statement": statement})
        rules_by_drug[drug_atc][icd_code] = contraindication_data

    for drug_atc, contraindications in rules_by_drug.items():
        redis_pipe.hset(f"ddxi:{drug_atc}", mapping=contraindications)

def sync_dose_rules(pg_cursor, redis_pipe):
    pg_cursor.execute("SELECT drug_atc, population, route, dose_min, dose_max, dose_unit, note FROM dose_rules")

    rules_by_drug = defaultdict(list)
    for row in pg_cursor.fetchall():
        drug_atc, pop, route, d_min, d_max, unit, note = row
        # Chuyển đổi Decimal sang float nếu cần
        dose_min = float(d_min) if d_min is not None else None
        dose_max = float(d_max) if d_max is not None else None

        dose_rule = {
            "population": pop, "route": route, "dose_min": dose_min,
            "dose_max": dose_max, "dose_unit": unit, "note": note
        }
        rules_by_drug[drug_atc].append(dose_rule)

    for drug_atc, rules in rules_by_drug.items():
        redis_pipe.set(f"dose:{drug_atc}", json.dumps(rules))

def main():
    try:
        pg_conn = psycopg2.connect(**POSTGRES_CONFIG)
        pg_cursor = pg_conn.cursor()

        redis_conn = redis.Redis(**REDIS_CONFIG, decode_responses=True)
        redis_conn.ping() # Kiểm tra kết nối

        redis_conn.flushdb()

        with redis_conn.pipeline() as pipe:
            sync_ddi_rules(pg_cursor, pipe)
            sync_ddxi_rules(pg_cursor, pipe)
            sync_dose_rules(pg_cursor, pipe)

            pipe.execute()

    except psycopg2.OperationalError as e:
        print(f"LỖI: Không thể kết nối tới PostgreSQL. Vui lòng kiểm tra lại thông tin kết nối và đảm bảo container đang chạy.")
        print(f"   Chi tiết: {e}")
    except redis.exceptions.ConnectionError as e:
        print(f"LỖI: Không thể kết nối tới Redis. Vui lòng kiểm tra lại thông tin kết nối và đảm bảo container đang chạy.")
        print(f"   Chi tiết: {e}")
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")
    finally:
        if 'pg_conn' in locals() and pg_conn:
            pg_cursor.close()
            pg_conn.close()

if __name__ == "__main__":
    main()