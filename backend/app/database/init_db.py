from app.core.config import settings


def init_db() -> None:
    if settings.USE_MOCK_MODE:
        return
    import psycopg2

    conn = psycopg2.connect(settings.DATABASE_URL)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS save_slots (
            id SERIAL PRIMARY KEY,
            slot_name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB,
            rag_context_ids INTEGER[],
            game_state JSONB
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS task_history (
            id SERIAL PRIMARY KEY,
            parent_id INTEGER REFERENCES task_history(id),
            task_type VARCHAR(50),
            prompt TEXT,
            response TEXT,
            state_delta JSONB,
            image_url VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    cur.close()
    conn.close()
