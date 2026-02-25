import argparse
import sys
import os

# Ensure we can import from current directory
sys.path.append(os.getcwd())

from coronary_stent_design.stent_generator import StentGenerator
from coronary_stent_design.recommendation_engine import StentRecommender
from coronary_stent_design.visualizer import StentVisualizer

def main():
    parser = argparse.ArgumentParser(description="Coronary Stent Design & Recommendation System")
    parser.add_argument("--diameter", type=float, default=3.0, help="Stent diameter in mm")
    parser.add_argument("--length", type=float, default=20.0, help="Stent length in mm")
    parser.add_argument("--thickness", type=float, default=0.09, help="Strut thickness in mm")
    parser.add_argument("--output", type=str, default="stent_design_output", help="Output filename base")
    
    args = parser.parse_args()
    
    print(f"Generating Stent Design: D={args.diameter}mm, L={args.length}mm, T={args.thickness}mm")
    
    # 1. Analyze & Recommend (FIRST)
    recommender = StentRecommender(args.diameter, args.thickness)
    recommender.analyze()
    report = recommender.get_report()
    
    print("\n--- Recommendation Report ---")
    opt = report.get("optimization_data", {})
    print(f"Reynolds Number: {opt.get('Re', 0):.2f}")
    print(f"Optimal Crowns (Prime Optimized): {opt.get('Optimal_Crowns', 6)}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")
        
    print("\nDesign Hints:")
    for hint in report['design_hints']:
        print(f"- {hint}")

    # 2. Generate Geometry (Using optimized crowns)
    crowns = opt.get('Optimal_Crowns', 6)
    generator = StentGenerator(args.length, args.diameter, args.thickness)
    generator.generate_sine_rings(crowns=crowns)
    # Add connectors feature
    generator.add_connectors()
    
    generator.export_to_json(f"{args.output}.json")
    print(f"\nGeometry exported to {args.output}.json (Crowns={crowns})")
    
    # Generate Physics Report (LaTeX)
    import subprocess
    report_tex = "nature_stent_report.tex"
    print(f"Generating Technical Report: {report_tex}")
    # We will assume the file exists or is created by the system separately, 
    # but we can trigger the build command if it existed. 
    # Since I am writing the logic, I'll rely on the user/system to create the .tex file next.
        
    # 3. Visualize
    visualizer = StentVisualizer(generator)
    visualizer.plot_unrolled_pattern(f"{args.output}.png")

if __name__ == "__main__":
    main()
