
import matplotlib.pyplot as plt
import numpy as np
from dementia_neural_model import DementiaNeuralModel

def generate_dbs_plots():
    print("Running DBS Dementia Simulation for Plots...")
    
    # Initialize Model (Moderate Dementia)
    model = DementiaNeuralModel(disease_duration_years=3.0)
    
    # Predict 12-month treatment response
    print("Simulating 12-month treatment...")
    prediction = model.predict_treatment_response(
        target_region='nucleus_basalis',
        amplitude_ma=3.0,
        frequency_hz=20,
        pulse_width_us=90,
        treatment_months=12
    )
    
    months = [d['month'] for d in prediction['monthly_progression']]
    mmse_scores = [d['mmse'] for d in prediction['monthly_progression']]
    moca_scores = [d['moca'] for d in prediction['monthly_progression']]
    
    # PLOT 1: Cognitive Score Progression
    plt.figure(figsize=(10, 6))
    plt.plot(months, mmse_scores, 'b-o', label='MMSE Score', linewidth=2)
    plt.plot(months, moca_scores, 'r-s', label='MoCA Score', linewidth=2)
    
    # Add baseline and healthy reference lines
    plt.axhline(y=24, color='g', linestyle='--', alpha=0.7, label='MCI Threshold (24)')
    plt.axhline(y=10, color='k', linestyle=':', alpha=0.5, label='Severe Threshold (10)')
    
    plt.title('Cognitive Recovery under Cholinergic DBS Therapy', fontsize=14)
    plt.xlabel('Treatment Duration (Months)', fontsize=12)
    plt.ylabel('Cognitive Score (0-30)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('dbs_treatment_outcome.png')
    print("Saved dbs_treatment_outcome.png")
    plt.close()
    
    # PLOT 2: Neural Activity Histogram (Pre vs Post)
    regions = list(model.regions.keys())
    # Create a fresh model for pre-state to ensure clean comparison
    pre_model = DementiaNeuralModel(disease_duration_years=3.0)
    pre_activity = [pre_model.activity[r] for r in regions]
    
    # Post state from the treated model
    post_activity = [prediction['monthly_progression'][-1]['activity'][r] for r in regions]
    
    x = np.arange(len(regions))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, pre_activity, width, label='Pre-Treatment (3 Years)', color='gray')
    plt.bar(x + width/2, post_activity, width, label='Post-Treatment (12 Months)', color='#00d4ff')
    
    plt.ylabel('Neural Activity (Normalized)', fontsize=12)
    plt.title('Regional Neural Reactivation', fontsize=14)
    plt.xticks(x, [r.replace('_', '\n').title() for r in regions])
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig('dbs_neural_reactivation.png')
    print("Saved dbs_neural_reactivation.png")
    plt.close()

if __name__ == "__main__":
    generate_dbs_plots()
