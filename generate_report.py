from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Technical Report: MR Computing Network Infrastructure', new_x="LMARGIN", new_y="NEXT", align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', new_x="LMARGIN", new_y="NEXT", align='C')

    def chapter_title(self, title):
        self.set_font('helvetica', 'B', 12)
        self.multi_cell(0, 10, title)
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('helvetica', '', 11)
        self.multi_cell(0, 7, body)
        self.ln(5)

pdf = PDF()
pdf.add_page()

# 1. Executive Summary
pdf.chapter_title('1. Executive Summary')
body1 = (
    "This technical report outlines the advanced cloud cluster topology, computing infrastructure, "
    "and network specifications designed to support efficient Magnetic Resonance (MR) image signal "
    "reconstruction and bleeding-edge imaging research. The architecture integrates on-premise "
    "Edge Nodes with massive, highly parallelized Cloud HPC instances and national supercomputing "
    "installations (Compute Canada) over specialized optical networks."
)
pdf.chapter_body(body1)

# 2. Network Topology & MR Signal Reconstruction
pdf.chapter_title('2. Network Topology and Computing Infrastructure')
body2 = (
    "Architecture for efficient MR image signal reconstruction and imaging:\n\n"
    "- Edge Nodes: Localized scanner gateways dedicated to immediate, real-time Fourier Transform approximations to offload primary scanner workloads.\n\n"
    "- Compute Canada Integration: Seamless bursting capabilities to national Compute Canada HPC clusters (e.g., Niagara, Cedar GPU, Beluga GPU). "
    "These clusters handle heavy non-Cartesian gridding, complex iterative reconstructions, and deep learning AI inference logic.\n\n"
    "- Network Path: Multi-tiered routing characterized by high-bandwidth and low-latency direct connections between "
    "DICOM nodes, local Edge clusters, and external research HPCs."
)
pdf.chapter_body(body2)

# 3. MR Research Computing & Cloud Topology
pdf.chapter_title('3. MR Research Computing & Cloud Topology')
body3 = (
    "Advanced computing framework specifications targeted at pure MRI research:\n\n"
    "- Hybrid Core: Secure on-premises edge nodes handling acute clinical data interoperating rapidly with multi-cloud resources without breaching PHI compliance.\n\n"
    "- AWS/GCP Extensibility: Orchestrated hyper-scale Spot instance fleets configured specifically for massive parallelized parameter grid searches, "
    "radiomics feature extraction, and large-cohort foundation model AI training.\n\n"
    "- Quantum Accelerators: Integration of experimental NVIDIA Quantum nodes for the bleeding-edge simulation of macroscopic spin-dynamics "
    "and quantum-assisted pulse sequences."
)
pdf.chapter_body(body3)

# 4. Network Specifications
pdf.chapter_title('4. Detailed Network Specifications')
body4 = (
    "To facilitate the optimal transfer of enormous 4D MR datasets and raw k-space arrays, the network layer specifies:\n\n"
    "- Protocol: TCP/IP optimized meticulously for jumbo frames (MTU 9000). Extensive use of RDMA over Converged Ethernet (RoCE) for cluster communications.\n"
    "- Bandwidth: Persistent 100 Gbps dedicated optical links natively routed through the CANARIE network (and regional optical pipelines).\n"
    "- Latency: Under 5ms RTT to regional foundational Compute Canada nodes to mimic local disk behavior over SANs.\n"
    "- Security: Complete end-to-end AES-256-GCM point-to-point encryption, augmented with quantum-safe key exchanges across all boundaries.\n"
    "- Reliability: 99.999% guaranteed uptime mediated by automated dual-redundant backbone paths routing over BGP."
)
pdf.chapter_body(body4)

# 5. Hardware Specifications
pdf.chapter_title('5. Core Computing Infrastructure Hardware Specs')
body5 = (
    "- Storage: Massively distributed Ceph File System utilizing a 10 PB high-IOPS NVMe tier specifically designated for active/hot clinical datasets.\n"
    "- Compute Nodes: Fleet-scale compute-optimized instances sporting 128 vCPUs and at least 1TB RAM to digest traditional legacy workflows (FSL, FreeSurfer).\n"
    "- GPU Acceleration: Dedicated enclosures of A100 and H100 instances exclusively serving dense multi-dimensional tensor operations and AI neural network reconstruction.\n"
    "- Network Fabric: InfoBand 400Gbps switches guaranteeing unbounded internal cluster inter-node topology communications."
)
pdf.chapter_body(body5)

pdf.output('MR_Network_Infrastructure_Report.pdf')
print('PDF generated successfully.')
