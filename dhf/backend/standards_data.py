from typing import List
from models import IEC13485Clause

IEC_13485_CLAUSES = [
    IEC13485Clause(
        id="4",
        clause_number="4",
        title="Quality Management System",
        description="General requirements for the QMS, including documentation requirements and medical device file.",
        sub_clauses=[]
    ),
    IEC13485Clause(
        id="4.1",
        clause_number="4.1",
        title="General Requirements",
        description="The organization shall document a QMS and maintain its effectiveness.",
        parent_id="4",
        sub_clauses=[]
    ),
    IEC13485Clause(
        id="4.2",
        clause_number="4.2",
        title="Documentation Requirements",
        description="General, Quality Manual, Medical Device File, Control of Documents, Control of Records.",
        parent_id="4",
        sub_clauses=[]
    ),
    IEC13485Clause(
        id="5",
        clause_number="5",
        title="Management Responsibility",
        description="Management commitment, customer focus, quality policy, planning, responsibility authority and communication, management review.",
        sub_clauses=[]
    ),
    IEC13485Clause(
        id="6",
        clause_number="6",
        title="Resource Management",
        description="Provision of resources, human resources, infrastructure, work environment and contamination control.",
        sub_clauses=[]
    ),
    IEC13485Clause(
        id="7",
        clause_number="7",
        title="Product Realization",
        description="Planning of product realization, customer-related processes, design and development, purchasing, production and service provision, control of monitoring and measuring equipment.",
        sub_clauses=[]
    ),
    IEC13485Clause(
        id="7.1",
        clause_number="7.1",
        title="Planning of Product Realization",
        description="The organization shall plan and develop the processes needed for product realization.",
        parent_id="7",
        sub_clauses=[]
    ),
    IEC13485Clause(
        id="7.3",
        clause_number="7.3",
        title="Design and Development",
        description="Design and development planning, inputs, outputs, review, verification, validation, transfer, control of changes.",
        parent_id="7",
        sub_clauses=[]
    ),
    IEC13485Clause(
        id="8",
        clause_number="8",
        title="Measurement, Analysis and Improvement",
        description="General, monitoring and measurement, control of nonconforming product, analysis of data, improvement.",
        sub_clauses=[]
    )
]
