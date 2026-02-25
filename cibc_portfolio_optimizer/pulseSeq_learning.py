import os, re, json, time
from pathlib import Path
import pandas as pd
from qiskit_aer import AerSimulator
import google.generativeai as genai
import numpy as np
from qiskit import QuantumCircuit, transpile
#from qiskit.providers.aer import AerSimulator
from qiskit.pulse import Schedule, DriveChannel, Play, Gaussian

# -------------------------------------------------
# 0️⃣  CONFIGURATION
# -------------------------------------------------
SEQ_DIR          = Path('my_seq_folder')
API_KEY_ENV      = "GOOGLE_API_KEY"
MAX_ITERATIONS   = 8
#GOOGLE_API_KEY=AIzaSyBj_2HZwkIKCFxXbxmWSRfrpAOvc23IV0Y

#API_KEY_ENV=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
genai.configure(api_key=os.getenv(GOOGLE_API_KEY))

# -------------------------------------------------
# 2️⃣  Parser (same as Section 2)
# -------------------------------------------------
def parse_seq(file_path: Path) -> dict:
    # … (copy the parse_seq implementation from Section 2) …
    # (omitted for brevity – paste the full function here)
    pass

def build_dataset(seq_dir: Path) -> pd.DataFrame:
    rows = []
    for fp in seq_dir.glob('*.seq'):
        parsed = parse_seq(fp)
        rows.append({
            'seq_id'   : parsed['name'],
            'seq_text' : parsed['raw'],
            'target_fid': float(parsed['metadata'].get('target_fidelity', 0.0))
        })
    return pd.DataFrame(rows)

# -------------------------------------------------
# 3️⃣  Gemini prompt helper
# -------------------------------------------------
few_shot_examples = """
Example #1:
Input:
```
# target_fidelity: 0.985
d0.play(gaussian, 40)
d1.play(gaussian, 40)
```
Output:
```
# target_fidelity: 0.998
d0.play(gaussian, 35)
d1.play(gaussian, 35)
```
"""

def ask_gemini(prompt: str, temperature: float = TEMPERATURE) -> str:
    resp = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=1024,
            top_p=0.95,
        )
    )
    return resp.text.strip()

def optimise_seq(seq_text: str) -> str:
    prompt = f"""
You are an expert quantum‑control engineer. 
Given a pulse‑program description written in the Qiskit‑Pulse DSL, suggest a modified version that 
maximizes two‑qubit gate fidelity while keeping total duration ≤ 150 ns.

{few_shot_examples}
Now, generate an improved sequence for the following input:
```
{seq_text}
```
"""
    return ask_gemini(prompt)

# -------------------------------------------------
# 4️⃣  Simulation utilities (Section 5)
# -------------------------------------------------
def seq_to_schedule(seq_text: str) -> Schedule:
    sched = Schedule(name='generated')
    for line in seq_text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = re.search(r'(?P<chan>\w+)\.play\(\s*(?P<pulse>\w+)\s*,\s*(?P<dur>\d+)\s*\)', line)
        if not m: continue
        chan = DriveChannel(int(m.group('chan')[1:]))
        dur  = int(m.group('dur'))
        if m.group('pulse').lower() == 'gaussian':
            pulse = Gaussian(duration=dur, amp=0.5, sigma=dur/4)
        else:
            pulse = Gaussian(duration=dur, amp=0.5, sigma=dur/2)
        sched += Play(pulse, chan)
    return sched

def simulate_fidelity(schedule: Schedule) -> float:
    backend = AerSimulator(method='density_matrix')
    qc = QuantumCircuit(2)
    qc.append(schedule, [0, 1])          # custom “instruction” – works for simple cases
    ideal = QuantumCircuit(2)
    ideal.cz(0, 1)

    # run noisy
    noisy_res = backend.run(transpile(qc, backend)).result()
    noisy_dm  = noisy_res.data(0)['density_matrix']

    # run ideal
    ideal_res = backend.run(transpile(ideal, backend)).result()
    ideal_dm  = ideal_res.data(0)['density_matrix']

    fid = np.real(np.trace(np.dot(noisy_dm.conj().T, ideal_dm))) / 4.0
    return fid

# -------------------------------------------------
# 5️⃣  Closed‑loop optimiser
# -------------------------------------------------
def optimise_loop(original_seq: str, max_iters: int = MAX_ITERATIONS):
    best_seq = original_seq
    best_fid = simulate_fidelity(seq_to_schedule(best_seq))
    print(" Starting fidelity")
    print(best_fid)

    for i in range(max_iters):
        cand_seq = optimise_seq(best_seq)
        # sanity‑check: make sure Gemini didn't output anything non‑textual
        if not cand_seq.strip():
            print(" Gemini returned empty output – skipping.")
            continue

        try:
            cand_fid = simulate_fidelity(seq_to_schedule(cand_seq))
        except Exception as e:
            print(f" Invalid candidate ({e}) – keep old.")
            continue



        if cand_fid > best_fid:
            best_fid, best_seq = cand_fid, cand_seq
            print(" New best found!")

        # small sleep to respect API rate limits
        time.sleep(0.4)

    return best_seq, best_fid

# -------------------------------------------------
# 6️⃣  Main driver
# -------------------------------------------------
if __name__ == "__main__":
    df = build_dataset(SEQ_DIR)
    # Pick a random entry (or you can loop over the whole set)
    seed_seq = df.iloc[0]['seq_text']

    opt_seq, opt_fid = optimise_loop(seed_seq)
    print("\n=== FINAL OPTIMIZED SEQUENCE ===")
    print(opt_seq)
    print(f"Achieved fidelity: {opt_fid:.6f}")

    # OPTIONAL: save the result
    out_path = Path('optimized') / f"opt_{int(time.time())}.seq"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(opt_seq)
    print(" Saved to {out_path}")






