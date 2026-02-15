import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pulse_generator import PulseGenerator
import threading

class QuantumPulseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Pulse Architect")
        self.root.geometry("500x450")
        self.root.configure(bg="#f0f0f0")

        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Generator Instance
        self.generator = PulseGenerator()

        # UI Components
        self.create_widgets()

    def create_widgets(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50")
        header_frame.pack(fill="x", padx=0, pady=0)
        
        lbl_title = tk.Label(header_frame, text="Quantum Pulse Generator", font=("Helvetica", 16, "bold"), fg="white", bg="#2c3e50", pady=10)
        lbl_title.pack()

        # Main Form
        form_frame = tk.Frame(self.root, bg="#f0f0f0")
        form_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Sequence Type
        tk.Label(form_frame, text="Sequence Type:", bg="#f0f0f0", font=("Arial", 10)).grid(row=0, column=0, sticky="w", pady=5)
        self.seq_type_var = tk.StringVar(value="GRE")
        type_cb = ttk.Combobox(form_frame, textvariable=self.seq_type_var, values=["GRE", "SE"], state="readonly")
        type_cb.grid(row=0, column=1, sticky="ew", pady=5)
        type_cb.bind("<<ComboboxSelected>>", self.update_fields)

        # TE
        tk.Label(form_frame, text="Echo Time (TE) [ms]:", bg="#f0f0f0", font=("Arial", 10)).grid(row=1, column=0, sticky="w", pady=5)
        self.te_var = tk.DoubleVar(value=20.0)
        tk.Entry(form_frame, textvariable=self.te_var).grid(row=1, column=1, sticky="ew", pady=5)

        # TR
        tk.Label(form_frame, text="Repetition Time (TR) [ms]:", bg="#f0f0f0", font=("Arial", 10)).grid(row=2, column=0, sticky="w", pady=5)
        self.tr_var = tk.DoubleVar(value=100.0)
        tk.Entry(form_frame, textvariable=self.tr_var).grid(row=2, column=1, sticky="ew", pady=5)

        # Flip Angle (GRE only)
        self.lbl_fa = tk.Label(form_frame, text="Flip Angle [deg]:", bg="#f0f0f0", font=("Arial", 10))
        self.lbl_fa.grid(row=3, column=0, sticky="w", pady=5)
        self.fa_var = tk.DoubleVar(value=90.0)
        self.ent_fa = tk.Entry(form_frame, textvariable=self.fa_var)
        self.ent_fa.grid(row=3, column=1, sticky="ew", pady=5)

        # Optimization Toggle
        self.opt_var = tk.BooleanVar(value=False)
        tk.Checkbutton(form_frame, text="Enable Quantum Optimization", variable=self.opt_var, bg="#f0f0f0", font=("Arial", 10)).grid(row=4, column=0, columnspan=2, sticky="w", pady=10)

        # Generate Button
        btn_gen = tk.Button(self.root, text="GENERATE SEQUENCE", command=self.generate_sequence, bg="#27ae60", fg="white", font=("Arial", 11, "bold"), relief="flat", padx=20, pady=10)
        btn_gen.pack(pady=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var, bg="#f0f0f0", fg="#7f8c8d", font=("Arial", 9)).pack(side="bottom", pady=5)

    def update_fields(self, event=None):
        st = self.seq_type_var.get()
        if st == "SE":
            self.lbl_fa.grid_remove()
            self.ent_fa.grid_remove()
        else:
            self.lbl_fa.grid()
            self.ent_fa.grid()

    def generate_sequence(self):
        # file dialog
        filename = filedialog.asksaveasfilename(defaultextension=".seq", filetypes=[("Pulseq File", "*.seq")])
        if not filename:
            return

        self.status_var.set("Generating...")
        self.root.update()
        
        # Run in thread to not freeze UI
        threading.Thread(target=self._run_generation, args=(filename,)).start()

    def _run_generation(self, filename):
        try:
            st = self.seq_type_var.get()
            te = self.te_var.get()
            tr = self.tr_var.get()
            opt = self.opt_var.get()
            
            params = {
                "te_ms": te,
                "tr_ms": tr,
                "fov_mm": 200,
                "matrix_size": 128,
                "optimize": opt
            }

            seq_data = None
            if st == "GRE":
                fa = self.fa_var.get()
                seq_data = self.generator.generate_gre(flip_angle_deg=fa, **params)
            else:
                seq_data = self.generator.generate_se(**params)
            
            content = self.generator.export_to_seq(seq_data, filename=filename)
            
            with open(filename, "w") as f:
                f.write(content)
                
            self.status_var.set(f"Saved to {filename}")
            messagebox.showinfo("Success", f"Sequence successfully generated and saved to:\n{filename}")
            
            if opt and 'optimization_metadata' in seq_data:
                meta = seq_data['optimization_metadata']
                messagebox.showinfo("Optimization Results", f"Method: {meta['method']}\nBest Value: {meta['optimal_value']:.2f}\nMetric: {meta['metric_value']:.4f}")

        except Exception as e:
            self.status_var.set("Error Occurred")
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = QuantumPulseApp(root)
    root.mainloop()
