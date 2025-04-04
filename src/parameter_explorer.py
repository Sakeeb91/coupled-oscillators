import numpy as np
import matplotlib.pyplot as plt
from coupled_oscillators import (
    run_simulation, visualize_results, check_stability_heuristic
)
import argparse
import os

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Explore parameters for coupled Goodwin oscillators'
    )
    
    # Oscillator 1 parameters
    parser.add_argument('--a1', type=float, default=5.0, help='Max mRNA synthesis rate for oscillator 1')
    parser.add_argument('--A1', type=float, default=1.0, help='Basal repression for oscillator 1')
    parser.add_argument('--b1', type=float, default=0.1, help='mRNA degradation rate for oscillator 1')
    parser.add_argument('--alpha1', type=float, default=0.2, help='Protein synthesis rate for oscillator 1')
    parser.add_argument('--beta1', type=float, default=0.02, help='Protein degradation rate for oscillator 1')
    parser.add_argument('--k11', type=float, default=0.05, help='Self-repression coefficient for oscillator 1')
    
    # Oscillator 2 parameters
    parser.add_argument('--a2', type=float, default=5.0, help='Max mRNA synthesis rate for oscillator 2')
    parser.add_argument('--A2', type=float, default=1.0, help='Basal repression for oscillator 2')
    parser.add_argument('--b2', type=float, default=0.1, help='mRNA degradation rate for oscillator 2')
    parser.add_argument('--alpha2', type=float, default=0.2, help='Protein synthesis rate for oscillator 2')
    parser.add_argument('--beta2', type=float, default=0.02, help='Protein degradation rate for oscillator 2')
    parser.add_argument('--k22', type=float, default=0.05, help='Self-repression coefficient for oscillator 2')
    
    # Coupling parameters
    parser.add_argument('--k12', type=float, default=0.01, help='Repression of osc1 by osc2')
    parser.add_argument('--k21', type=float, default=0.01, help='Repression of osc2 by osc1')
    
    # Simulation parameters
    parser.add_argument('--t_end', type=float, default=2000.0, help='End time for simulation')
    parser.add_argument('--n_points', type=int, default=2000, help='Number of time points')
    parser.add_argument('--output', type=str, default='custom', help='Output filename prefix')
    parser.add_argument('--description', type=str, default='', help='Description of this simulation run')
    
    return parser.parse_args()

def main():
    """Main function to run parameter exploration"""
    args = parse_arguments()
    
    # Create base results directory
    os.makedirs("results", exist_ok=True)
    
    # Set up oscillator 1 parameters
    params1 = {
        'a': args.a1,
        'A': args.A1,
        'b': args.b1,
        'alpha': args.alpha1,
        'beta': args.beta1,
        'k_self': args.k11
    }
    
    # Set up oscillator 2 parameters
    params2 = {
        'a': args.a2,
        'A': args.A2,
        'b': args.b2,
        'alpha': args.alpha2,
        'beta': args.beta2,
        'k_self': args.k22
    }
    
    # Set up coupling parameters
    coupling_params = {
        'k12': args.k12,
        'k21': args.k21
    }
    
    # Generate a description if not provided
    run_description = args.description
    if not run_description:
        stable = check_stability_heuristic(args.k11, args.k22, args.k12, args.k21)
        stability_status = "stable" if stable else "unstable"
        coupling_type = "symmetric" if args.k12 == args.k21 else "asymmetric"
        run_description = f"Custom {coupling_type} coupling with k12={args.k12} and k21={args.k21}. "
        run_description += f"System predicted to be {stability_status} based on Goodwin's criterion."
    
    # Print parameter summary
    print("\nParameter Summary:")
    print("------------------")
    print(f"Oscillator 1: a={args.a1}, A={args.A1}, b={args.b1}, alpha={args.alpha1}, beta={args.beta1}, k_self={args.k11}")
    print(f"Oscillator 2: a={args.a2}, A={args.A2}, b={args.b2}, alpha={args.alpha2}, beta={args.beta2}, k_self={args.k22}")
    print(f"Coupling: k12={args.k12}, k21={args.k21}")
    
    # Check stability condition
    stable = check_stability_heuristic(args.k11, args.k22, args.k12, args.k21)
    stability_indicator = (args.k11 * args.k22 - args.k12 * args.k21)
    print(f"\nStability Analysis:")
    print(f"k11*k22 - k12*k21 = {stability_indicator:.6f}")
    print(f"Predicted stability: {'Stable' if stable else 'Unstable'}")
    
    # Run simulation
    print("\nRunning simulation...")
    t, X1, Y1, X2, Y2, results = run_simulation(
        params1, params2, coupling_params, 
        t_end=args.t_end, n_points=args.n_points,
        run_description=run_description
    )
    
    # Create visualizations and save to a uniquely named folder
    print("Generating visualizations...")
    run_folder = visualize_results(
        t, X1, Y1, X2, Y2, params1, params2, coupling_params,
        filename_prefix=args.output,
        run_description=run_description
    )
    
    # Open the dashboard
    print(f"Run complete! Results saved to: {run_folder}")
    print(f"Dashboard available at: {run_folder}/dashboard.html")
    
    return 0

if __name__ == "__main__":
    main() 