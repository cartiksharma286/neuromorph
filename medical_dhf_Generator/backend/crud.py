from database import get_db_connection
from models import QMSDocument, DocumentVersion, Device
from typing import List, Optional
from datetime import datetime

def create_document(doc: QMSDocument):
    conn = get_db_connection()
    c = conn.cursor()
    
    try:
        c.execute('''
            INSERT INTO documents (id, title, version, author, approval_status, last_updated, content)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (doc.id, doc.title, doc.version, doc.author, doc.approval_status, doc.last_updated, doc.content))
        
        # Initial version entry
        c.execute('''
            INSERT INTO document_versions (doc_id, version, change_description, changed_by, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (doc.id, doc.version, "Initial Creation", doc.author, doc.last_updated))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error creating document: {e}")
        return False
    finally:
        conn.close()

def get_all_documents() -> List[QMSDocument]:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM documents')
    rows = c.fetchall()
    
    documents = []
    for row in rows:
        # Convert row to dict first to avoid issues with index access if schema changes
        # row_factory is sqlite3.Row so it behaves like a dict
        
        # Fetch versions
        # Ideally we'd do a join, but for simplicity/clarity we'll do sub-queries for now or leave history empty for list view
        documents.append(QMSDocument(
            id=row['id'],
            title=row['title'],
            version=row['version'],
            author=row['author'],
            approval_status=row['approval_status'],
            last_updated=row['last_updated'],
            content=row['content'] if row['content'] else "",
            approved_by=row['approved_by'],
            approval_date=row['approval_date']
        ))
    
    conn.close()
    return documents

def get_document_by_id(doc_id: str) -> Optional[QMSDocument]:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
    row = c.fetchone()
    
    if row:
        doc = QMSDocument(
            id=row['id'],
            title=row['title'],
            version=row['version'],
            author=row['author'],
            approval_status=row['approval_status'],
            last_updated=row['last_updated'],
            content=row['content'] if row['content'] else "",
            approved_by=row['approved_by'],
            approval_date=row['approval_date']
        )
        conn.close()
        return doc
    
    conn.close()
    return None

def approve_document(doc_id: str, approver: str):
    conn = get_db_connection()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    
    c.execute('''
        UPDATE documents 
        SET approval_status = ?, approved_by = ?, approval_date = ?
        WHERE id = ?
    ''', ('APPROVED', approver, now, doc_id))
    
    conn.commit()
    conn.close()
    return {"status": "APPROVED", "approved_by": approver, "timestamp": now}

def reject_document(doc_id: str, reason: str):
    conn = get_db_connection()
    c = conn.cursor()
    
    # Reset to DRAFT
    c.execute('''
        UPDATE documents 
        SET approval_status = ?, approved_by = NULL, approval_date = NULL
        WHERE id = ?
    ''', ('DRAFT', doc_id))
    
    # Ideally verify this reason is logged somewhere (e.g. versions or a separate audit log)
    # For now we'll just return it
    
    conn.commit()
    conn.close()
    return {"status": "DRAFT", "note": reason}
