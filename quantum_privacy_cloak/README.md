# Northstar Cloak

A local Flask research prototype for exploring a quantum-invisibility-cloak field model, recurrent-prime session scheduling, and post-quantum privacy workflows for Canadian data governance.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 app.py
```

Open <http://127.0.0.1:7900>.

The cloak and QML calculations are simulations. The session endpoint exposes ML-KEM and ML-DSA as integration labels and uses a standard-library SHA-3/HMAC transcript commitment for the demo. Production systems must use a reviewed PQC implementation, threat model, key lifecycle, access controls, and applicable PIPEDA/provincial privacy review.
