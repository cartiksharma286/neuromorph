# Example clinical protocols for jet lag treatment
CLINICAL_PROTOCOLS = [
    {
        "name": "Standard Light Therapy",
        "description": "Timed exposure to bright light to shift circadian rhythms. Morning light for eastward travel, evening light for westward.",
        "steps": [
            "Begin light therapy 3 days before travel.",
            "Expose to 10,000 lux light for 30 minutes each morning.",
            "Continue for 5 days after arrival.",
        ]
    },
    {
        "name": "Melatonin Supplementation",
        "description": "Use of melatonin to advance or delay sleep phase.",
        "steps": [
            "Take 0.5-3 mg melatonin 1 hour before desired bedtime.",
            "Start 2 days before travel and continue for 4 days after arrival.",
        ]
    },
    {
        "name": "DBS Experimental Protocol",
        "description": "Deep brain stimulation targeting the suprachiasmatic nucleus (SCN) to modulate circadian phase.",
        "steps": [
            "Implant DBS electrode in SCN region.",
            "Deliver stimulation at 130 Hz, 60 μs pulse width, 2.5 V amplitude for 30 minutes at local morning time.",
            "Repeat daily for 5 days post-arrival.",
        ]
    },
]

def get_protocols():
    return CLINICAL_PROTOCOLS
