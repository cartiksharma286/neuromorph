from typing import List
from models import IEC13485Clause, IEC60601Clause

IEC_13485_CLAUSES = [
    IEC13485Clause(
        id="4",
        clause_number="4",
        title="Quality Management System",
        description="General requirements for the QMS, including documentation requirements and medical device file.",
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

IEC_60601_CLAUSES = [
    IEC60601Clause(id="60601-5.1", clause_number="5.1", title="Protection against electrical hazards", description="Equipment must provide protection against electric shock.", category="Electrical"),
    IEC60601Clause(id="60601-6.3", clause_number="6.3", title="Protection against mechanical hazards", description="Moving parts must be enclosed or guarded.", category="Mechanical"),
    IEC60601Clause(id="60601-7.2", clause_number="7.2", title="Radiation protection", description="Minimizing exposure to unwanted radiation.", category="Radiation"),
    IEC60601Clause(id="60601-8.1", clause_number="8.1", title="Excessive temperatures", description="Limits on surface temperatures of applied parts.", category="Thermal"),
    IEC60601Clause(id="60601-9.4", clause_number="9.4", title="Instability hazards", description="Prevention of tipping or falling.", category="Mechanical"),
]
