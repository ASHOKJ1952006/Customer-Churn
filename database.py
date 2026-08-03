"""
database.py — SQLite-backed persistent storage for churn predictions
Stores batch upload results and single predictions across app sessions.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

class ChurnDatabase:
    def __init__(self, db_path: str = "data/churn_predictions.db"):
        self.db_path = db_path
        self._init_db()
    
    def _get_connection(self):
        """Get database connection with row factory for dict access."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Single predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS single_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    churn_probability REAL,
                    risk_level TEXT,
                    top_factors TEXT,
                    recommended_action TEXT,
                    input_features TEXT
                )
            """)
            
            # Batch predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS batch_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT,
                    customer_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    churn_probability REAL,
                    risk_level TEXT,
                    top_factors TEXT,
                    recommended_action TEXT
                )
            """)
            
            # Batch metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS batch_metadata (
                    batch_id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_customers INTEGER,
                    high_risk_count INTEGER,
                    medium_risk_count INTEGER,
                    low_risk_count INTEGER,
                    avg_churn_probability REAL
                )
            """)
            
            # Risk queue table (high-risk customers for follow-up)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    churn_probability REAL,
                    risk_level TEXT,
                    top_factors TEXT,
                    recommended_action TEXT,
                    status TEXT DEFAULT 'pending',
                    follow_up_notes TEXT,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def add_single_prediction(
        self,
        customer_id: str,
        churn_probability: float,
        risk_level: str,
        top_factors: List[str],
        recommended_action: str,
        input_features: Dict
    ) -> int:
        """Add a single customer prediction to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO single_predictions 
                (customer_id, churn_probability, risk_level, top_factors, recommended_action, input_features)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                customer_id,
                churn_probability,
                risk_level,
                ", ".join(top_factors),
                recommended_action,
                str(input_features)
            ))
            
            # Add to risk queue if high risk
            if risk_level == "High":
                cursor.execute("""
                    INSERT INTO risk_queue
                    (customer_id, churn_probability, risk_level, top_factors, recommended_action)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    customer_id,
                    churn_probability,
                    risk_level,
                    ", ".join(top_factors),
                    recommended_action
                ))
            
            conn.commit()
            return cursor.lastrowid
    
    def add_batch_predictions(
        self,
        batch_id: str,
        predictions: List[Dict]
    ) -> None:
        """Add batch predictions to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            high_risk = 0
            medium_risk = 0
            low_risk = 0
            total_prob = 0
            
            for pred in predictions:
                cursor.execute("""
                    INSERT INTO batch_predictions
                    (batch_id, customer_id, churn_probability, risk_level, top_factors, recommended_action)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    batch_id,
                    pred['customer_id'],
                    pred['churn_probability'],
                    pred['risk_level'],
                    ", ".join(pred['top_factors']),
                    pred['recommended_action']
                ))
                
                # Count risk levels
                if pred['risk_level'] == 'High':
                    high_risk += 1
                    # Add to risk queue
                    cursor.execute("""
                        INSERT OR IGNORE INTO risk_queue
                        (customer_id, churn_probability, risk_level, top_factors, recommended_action)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        pred['customer_id'],
                        pred['churn_probability'],
                        pred['risk_level'],
                        ", ".join(pred['top_factors']),
                        pred['recommended_action']
                    ))
                elif pred['risk_level'] == 'Medium':
                    medium_risk += 1
                else:
                    low_risk += 1
                
                total_prob += pred['churn_probability']
            
            # Add batch metadata
            cursor.execute("""
                INSERT INTO batch_metadata
                (batch_id, total_customers, high_risk_count, medium_risk_count, low_risk_count, avg_churn_probability)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                batch_id,
                len(predictions),
                high_risk,
                medium_risk,
                low_risk,
                total_prob / len(predictions) if predictions else 0
            ))
            
            conn.commit()
    
    def get_single_predictions(self, limit: int = 100) -> pd.DataFrame:
        """Get recent single predictions."""
        with self._get_connection() as conn:
            query = """
                SELECT * FROM single_predictions 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            return pd.read_sql_query(query, conn, params=(limit,))
    
    def get_batch_predictions(self, batch_id: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        """Get batch predictions, optionally filtered by batch_id."""
        with self._get_connection() as conn:
            if batch_id:
                query = """
                    SELECT * FROM batch_predictions 
                    WHERE batch_id = ?
                    ORDER BY timestamp DESC
                """
                return pd.read_sql_query(query, conn, params=(batch_id,))
            else:
                query = """
                    SELECT * FROM batch_predictions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """
                return pd.read_sql_query(query, conn, params=(limit,))
    
    def get_batch_metadata(self, limit: int = 50) -> pd.DataFrame:
        """Get batch metadata."""
        with self._get_connection() as conn:
            query = """
                SELECT * FROM batch_metadata 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            return pd.read_sql_query(query, conn, params=(limit,))
    
    def get_risk_queue(self, status: str = 'pending', limit: int = 100) -> pd.DataFrame:
        """Get risk queue, optionally filtered by status."""
        with self._get_connection() as conn:
            if status:
                query = """
                    SELECT * FROM risk_queue 
                    WHERE status = ?
                    ORDER BY churn_probability DESC, timestamp DESC
                    LIMIT ?
                """
                return pd.read_sql_query(query, conn, params=(status, limit))
            else:
                query = """
                    SELECT * FROM risk_queue 
                    ORDER BY churn_probability DESC, timestamp DESC
                    LIMIT ?
                """
                return pd.read_sql_query(query, conn, params=(limit,))
    
    def update_risk_queue_status(
        self,
        customer_id: str,
        status: str,
        follow_up_notes: Optional[str] = None
    ) -> None:
        """Update risk queue status for a customer."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if follow_up_notes:
                cursor.execute("""
                    UPDATE risk_queue 
                    SET status = ?, follow_up_notes = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE customer_id = ?
                """, (status, follow_up_notes, customer_id))
            else:
                cursor.execute("""
                    UPDATE risk_queue 
                    SET status = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE customer_id = ?
                """, (status, customer_id))
            conn.commit()
    
    def get_dashboard_stats(self) -> Dict:
        """Get dashboard statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total predictions
            cursor.execute("SELECT COUNT(*) FROM single_predictions")
            total_single = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM batch_predictions")
            total_batch = cursor.fetchone()[0]
            
            # Risk queue stats
            cursor.execute("SELECT COUNT(*) FROM risk_queue WHERE status = 'pending'")
            pending_risks = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM risk_queue WHERE status = 'contacted'")
            contacted_risks = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM risk_queue WHERE status = 'resolved'")
            resolved_risks = cursor.fetchone()[0]
            
            # Recent high-risk count (last 7 days)
            cursor.execute("""
                SELECT COUNT(*) FROM risk_queue 
                WHERE status = 'pending' 
                AND timestamp >= datetime('now', '-7 days')
            """)
            recent_high_risk = cursor.fetchone()[0]
            
            return {
                'total_single_predictions': total_single,
                'total_batch_predictions': total_batch,
                'pending_risks': pending_risks,
                'contacted_risks': contacted_risks,
                'resolved_risks': resolved_risks,
                'recent_high_risk_7days': recent_high_risk
            }
    
    def clear_old_data(self, days: int = 30) -> None:
        """Clear data older than specified days."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM single_predictions 
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))
            
            cursor.execute("""
                DELETE FROM batch_predictions 
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))
            
            cursor.execute("""
                DELETE FROM batch_metadata 
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))
            
            cursor.execute("""
                DELETE FROM risk_queue 
                WHERE status = 'resolved' 
                AND last_updated < datetime('now', '-' || ? || ' days')
            """, (days,))
            
            conn.commit()
