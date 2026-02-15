from fastapi.testclient import TestClient
from main import app
import os
import database

# Use a test DB
database.DB_FILE = "test_med_dev_qms.db"

client = TestClient(app)

def test_startup():
    # Trigger startup event manually or trust client to do it (TestClient typically runs startup)
    # But we want to ensure clean DB
    if os.path.exists("test_med_dev_qms.db"):
        os.remove("test_med_dev_qms.db")
    database.init_db()

def test_create_document():
    doc_data = {
        "id": "TEST-001",
        "title": "Test Procedure",
        "version": "1.0",
        "author": "Tester",
        "approval_status": "DRAFT",
        "last_updated": "2025-01-01T12:00:00Z",
        "content": "This is a test."
    }
    response = client.post("/api/documents/create", json=doc_data)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_get_documents():
    response = client.get("/api/documents")
    assert response.status_code == 200
    docs = response.json()
    assert len(docs) > 0
    assert docs[0]["id"] == "TEST-001"

def test_approve_document():
    response = client.post("/api/documents/TEST-001/approve?approver=AdminQA")
    assert response.status_code == 200
    assert response.json()["status"] == "APPROVED"
    assert response.json()["approved_by"] == "AdminQA"

    # Verify persistent change
    response = client.get("/api/documents")
    docs = response.json()
    assert docs[0]["approval_status"] == "APPROVED"

def test_reject_document():
    # Create another doc to reject
    doc_data = {
        "id": "TEST-002",
        "title": "Bad Procedure",
        "version": "1.0",
        "author": "Tester",
        "approval_status": "DRAFT",
        "last_updated": "2025-01-01T12:00:00Z"
    }
    client.post("/api/documents/create", json=doc_data)
    
    response = client.post("/api/documents/TEST-002/reject?reason=Typo")
    assert response.status_code == 200
    assert response.json()["status"] == "DRAFT"
    
    # Check status (should be DRAFT, but arguably "REJECTED" or revert to DRAFT. implementation said "DRAFT" with note)
    response = client.get("/api/documents")
    # find doc
    doc = next(d for d in response.json() if d["id"] == "TEST-002")
    assert doc["approval_status"] == "DRAFT"

if __name__ == "__main__":
    test_startup()
    test_create_document()
    test_get_documents()
    test_approve_document()
    test_reject_document()
    print("ALL TESTS PASSED")
    
    # Cleanup
    if os.path.exists("test_med_dev_qms.db"):
        os.remove("test_med_dev_qms.db")
