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
    
    # 1. Generate Geometry
    generator = StentGenerator(args.length, args.diameter, args.thickness)
    generator.generate_sine_rings()
    generator.export_to_json(f"{args.output}.json")
    print(f"Geometry exported to {args.output}.json")
    
    # 2. Analyze & Recommend
    recommender = StentRecommender(args.diameter, args.thickness)
    recommender.analyze()
    report = recommender.get_report()
    
    print("\n--- Recommendation Report ---")
    print("Chamfer Recommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")
        
    print("\nDesign Hints:")
    for hint in report['design_hints']:
        print(f"- {hint}")
        
    # 3. Visualize
    visualizer = StentVisualizer(generator)
    visualizer.plot_unrolled_pattern(f"{args.output}.png")

if __name__ == "__main__":
    main()
