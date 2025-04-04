import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import hilbert
import os
import datetime
import json

def calculate_steady_state_single(params):
    """Calculate steady state for a single uncoupled oscillator"""
    a, A, b, alpha, beta, k_self = params
    
    # For steady state, we need dX/dt = 0 and dY/dt = 0
    # From dY/dt = 0, we get X = (beta/alpha) * Y
    # From dX/dt = 0, we get a/(A + k*Y) = b
    # Solving for Y: a/(A + k*Y) = b => a = b*(A + k*Y) => a/b = A + k*Y
    # Therefore Y = (a/b - A)/k
    
    q = (a/b - A)/k_self  # Y steady state
    if q <= 0:
        raise ValueError(f"Invalid steady state: a/b ({a/b}) must be > A ({A}) for positive steady state")
    
    p = (beta/alpha) * q  # X steady state
    
    return p, q

def coupled_goodwin_ode(t, state, params):
    """ODE function for coupled Goodwin oscillators"""
    X1, Y1, X2, Y2 = state
    p1_dict, p2_dict, coupling_dict = params  # Pass params in dictionaries

    # Oscillator 1 params
    a1, A1, b1, alpha1, beta1, k11 = (
        p1_dict['a'], p1_dict['A'], p1_dict['b'], 
        p1_dict['alpha'], p1_dict['beta'], p1_dict['k_self']
    )
    
    # Oscillator 2 params
    a2, A2, b2, alpha2, beta2, k22 = (
        p2_dict['a'], p2_dict['A'], p2_dict['b'], 
        p2_dict['alpha'], p2_dict['beta'], p2_dict['k_self']
    )
    
    # Coupling params
    k12, k21 = coupling_dict['k12'], coupling_dict['k21']

    # Denominators - ensure positivity
    denom1 = A1 + k11 * Y1 + k12 * Y2
    denom2 = A2 + k21 * Y1 + k22 * Y2
    denom1 = max(denom1, 1e-9)  # Avoid division by zero/negative
    denom2 = max(denom2, 1e-9)

    dX1dt = (a1 / denom1) - b1
    dY1dt = alpha1 * X1 - beta1 * Y1
    dX2dt = (a2 / denom2) - b2
    dY2dt = alpha2 * X2 - beta2 * Y2

    return [dX1dt, dY1dt, dX2dt, dY2dt]

def check_stability_heuristic(k11, k22, k12, k21):
    """Check stability based on Goodwin's analysis"""
    return (k11 * k22) > (k12 * k21)

def calculate_phase(signal):
    """Calculate instantaneous phase using Hilbert transform"""
    analytic_signal = hilbert(signal - np.mean(signal))
    phase = np.unwrap(np.angle(analytic_signal))
    return phase

def run_simulation(params1, params2, coupling_params, t_end=1000, n_points=2000, initial_state=None, run_description=""):
    """Run the simulation with given parameters and return results"""
    # Check stability (heuristic)
    k11, k22 = params1['k_self'], params2['k_self']
    k12, k21 = coupling_params['k12'], coupling_params['k21']
    stable = check_stability_heuristic(k11, k22, k12, k21)
    print(f"System stable (heuristic)? {stable}")
    
    # Define time span
    t_start = 0
    t_eval = np.linspace(t_start, t_end, n_points)
    
    # Calculate uncoupled steady states if initial state not provided
    if initial_state is None:
        try:
            p1_uncoupled, q1_uncoupled = calculate_steady_state_single(
                [params1['a'], params1['A'], params1['b'], 
                 params1['alpha'], params1['beta'], params1['k_self']]
            )
            p2_uncoupled, q2_uncoupled = calculate_steady_state_single(
                [params2['a'], params2['A'], params2['b'], 
                 params2['alpha'], params2['beta'], params2['k_self']]
            )
            
            # Start with slightly perturbed initial conditions (out of phase)
            initial_state = [
                p1_uncoupled * 1.1, q1_uncoupled * 0.9,
                p2_uncoupled * 0.9, q2_uncoupled * 1.1
            ]
            print(f"Calculated steady states - Osc1: ({p1_uncoupled:.4f}, {q1_uncoupled:.4f}), "
                  f"Osc2: ({p2_uncoupled:.4f}, {q2_uncoupled:.4f})")
            
        except ValueError as e:
            print(f"Warning: {e}")
            print("Using default initial values")
            # Fallback initial conditions if steady state calculation fails
            initial_state = [5.0, 5.0, 5.0, 5.0]
    
    # Combine parameters for passing to ODE function
    all_params = (params1, params2, coupling_params)
    
    # Solve ODE
    sol = solve_ivp(
        coupled_goodwin_ode,
        [t_start, t_end],
        initial_state,
        args=(all_params,),
        dense_output=True,
        t_eval=t_eval,
        method='LSODA'  # LSODA often performs well for stiff ODEs
    )
    
    # Check if integration successful
    if not sol.success:
        print(f"Warning: Integration failed with message: {sol.message}")
    
    # Extract results
    t = sol.t
    X1_sol, Y1_sol, X2_sol, Y2_sol = sol.y
    
    # Create a simulation results object with metadata
    simulation_results = {
        "params1": params1,
        "params2": params2,
        "coupling": coupling_params,
        "t_end": t_end,
        "n_points": n_points,
        "initial_state": initial_state,
        "description": run_description,
        "stable_prediction": stable,
        "stability_indicator": (k11 * k22 - k12 * k21)
    }
    
    return t, X1_sol, Y1_sol, X2_sol, Y2_sol, simulation_results

def visualize_results(t, X1_sol, Y1_sol, X2_sol, Y2_sol, params1, params2, coupling_params, 
                      filename_prefix='', run_description=""):
    """Visualize simulation results with multiple plots and save to a unique folder with documentation"""
    # Create a unique folder for this run based on date and time
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if filename_prefix:
        run_folder = f"results/{filename_prefix}_{timestamp}"
    else:
        run_folder = f"results/{timestamp}"
    
    os.makedirs(run_folder, exist_ok=True)
    
    k11, k22 = params1['k_self'], params2['k_self']
    k12, k21 = coupling_params['k12'], coupling_params['k21']
    
    # Get coupling type for plot titles
    stability = "Stable" if check_stability_heuristic(k11, k22, k12, k21) else "Unstable"
    coupling_desc = f"Coupling: k12={k12:.3f}, k21={k21:.3f} ({stability})"
    
    # Dictionary to store plot metadata for dashboard
    plots_metadata = {}
    
    # Plot 1: Time series of mRNA and protein concentrations
    plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 2)
    
    # mRNA time series
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(t, X1_sol, label='mRNA 1', color='darkblue')
    ax1.plot(t, X2_sol, label='mRNA 2', color='darkred')
    ax1.set_title(f'mRNA Time Series - {coupling_desc}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Concentration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Protein time series
    ax2 = plt.subplot(gs[1, :])
    ax2.plot(t, Y1_sol, label='Protein 1', color='blue')
    ax2.plot(t, Y2_sol, label='Protein 2', color='red')
    ax2.set_title('Protein Time Series')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Concentration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Phase portraits
    ax3 = plt.subplot(gs[2, 0])
    ax3.plot(X1_sol, Y1_sol, color='blue')
    ax3.set_title('Phase Portrait: Oscillator 1')
    ax3.set_xlabel('mRNA 1')
    ax3.set_ylabel('Protein 1')
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(gs[2, 1])
    ax4.plot(X2_sol, Y2_sol, color='red')
    ax4.set_title('Phase Portrait: Oscillator 2')
    ax4.set_xlabel('mRNA 2')
    ax4.set_ylabel('Protein 2')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    time_series_plot = f"time_series.png"
    plt.savefig(f'{run_folder}/{time_series_plot}', dpi=300)
    plots_metadata[time_series_plot] = {
        "title": "Time Series and Phase Portraits",
        "description": "Top: mRNA concentrations over time. Middle: Protein concentrations over time. Bottom: Phase portraits of both oscillators showing the relationship between mRNA and protein concentrations for each oscillator."
    }
    
    # Plot 2: Comparative phase portraits
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(X1_sol, X2_sol, color='purple')
    plt.title('mRNA1 vs mRNA2')
    plt.xlabel('mRNA 1')
    plt.ylabel('mRNA 2')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(Y1_sol, Y2_sol, color='orange')
    plt.title('Protein1 vs Protein2')
    plt.xlabel('Protein 1')
    plt.ylabel('Protein 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    phase_comparison_plot = f"phase_comparison.png"
    plt.savefig(f'{run_folder}/{phase_comparison_plot}', dpi=300)
    plots_metadata[phase_comparison_plot] = {
        "title": "Comparative Phase Portraits",
        "description": "Left: Relationship between mRNA levels of both oscillators. Right: Relationship between protein levels of both oscillators. These plots help visualize phase relationships and synchronization between the oscillators."
    }
    
    # Plot 3: Phase difference plot (if oscillations are present)
    phase_diff_plot = None
    try:
        # Calculate phases for both oscillators
        # Use second half of data to avoid transients
        mid_point = len(t) // 2
        phase1 = calculate_phase(X1_sol[mid_point:])
        phase2 = calculate_phase(X2_sol[mid_point:])
        phase_diff = phase1 - phase2
        
        plt.figure(figsize=(10, 4))
        plt.plot(t[mid_point:], phase_diff, color='green')
        plt.title('Phase Difference (mRNA1 - mRNA2)')
        plt.xlabel('Time')
        plt.ylabel('Phase Difference (radians)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        phase_diff_plot = f"phase_difference.png"
        plt.savefig(f'{run_folder}/{phase_diff_plot}', dpi=300)
        plots_metadata[phase_diff_plot] = {
            "title": "Phase Difference",
            "description": "The difference in phase between the two oscillators over time. A constant phase difference indicates phase locking, while a changing difference indicates phase drift."
        }
    except Exception as e:
        print(f"Could not calculate phase differences - possibly non-oscillatory behavior: {e}")
    
    plt.close('all')  # Close all figures to free memory
    
    # Create metadata for the simulation
    simulation_metadata = {
        "timestamp": timestamp,
        "oscillator1_params": params1,
        "oscillator2_params": params2,
        "coupling_params": coupling_params,
        "stability": stability,
        "stability_indicator": k11 * k22 - k12 * k21,
        "description": run_description,
        "plots": plots_metadata
    }
    
    # Save metadata as JSON
    with open(f'{run_folder}/metadata.json', 'w') as f:
        json.dump(simulation_metadata, f, indent=2)
    
    # Create markdown file with documentation
    with open(f'{run_folder}/README.md', 'w') as f:
        f.write(f"# Simulation Results: {filename_prefix}\n\n")
        f.write(f"**Run Date and Time:** {timestamp}\n\n")
        
        if run_description:
            f.write(f"**Description:** {run_description}\n\n")
        
        f.write("## Parameters\n\n")
        f.write("### Oscillator 1\n")
        for param, value in params1.items():
            f.write(f"- {param}: {value}\n")
        
        f.write("\n### Oscillator 2\n")
        for param, value in params2.items():
            f.write(f"- {param}: {value}\n")
        
        f.write("\n### Coupling\n")
        f.write(f"- k12 (Repression of Osc1 by Osc2): {k12}\n")
        f.write(f"- k21 (Repression of Osc2 by Osc1): {k21}\n\n")
        
        f.write("## Stability Analysis\n\n")
        f.write(f"- k11*k22 - k12*k21 = {k11*k22 - k12*k21:.6f}\n")
        f.write(f"- Predicted stability: {stability}\n\n")
        
        f.write("## Results\n\n")
        
        # Time Series
        f.write("### Time Series and Phase Portraits\n\n")
        f.write(f"![Time Series and Phase Portraits]({time_series_plot})\n\n")
        f.write(plots_metadata[time_series_plot]["description"] + "\n\n")
        
        # Phase Comparison
        f.write("### Comparative Phase Portraits\n\n")
        f.write(f"![Comparative Phase Portraits]({phase_comparison_plot})\n\n")
        f.write(plots_metadata[phase_comparison_plot]["description"] + "\n\n")
        
        # Phase Difference (if available)
        if phase_diff_plot:
            f.write("### Phase Difference\n\n")
            f.write(f"![Phase Difference]({phase_diff_plot})\n\n")
            f.write(plots_metadata[phase_diff_plot]["description"] + "\n\n")
    
    # Create a dashboard HTML file
    with open(f'{run_folder}/dashboard.html', 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Simulation Dashboard: {filename_prefix}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .plot-section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
        .plot-img {{ max-width: 100%; height: auto; margin-bottom: 10px; }}
        .plot-caption {{ font-style: italic; color: #666; }}
        .params {{ display: flex; flex-wrap: wrap; }}
        .param-block {{ flex: 1; min-width: 300px; margin-right: 20px; }}
        h1, h2, h3 {{ color: #333; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Simulation Dashboard: {filename_prefix}</h1>
            <p><strong>Run Date and Time:</strong> {timestamp}</p>
            """)
        
        if run_description:
            f.write(f"<p><strong>Description:</strong> {run_description}</p>\n")
        
        f.write(f"""
        </div>
        
        <h2>Parameters</h2>
        <div class="params">
            <div class="param-block">
                <h3>Oscillator 1</h3>
                <ul>
        """)
        
        for param, value in params1.items():
            f.write(f"<li>{param}: {value}</li>\n")
        
        f.write(f"""
                </ul>
            </div>
            
            <div class="param-block">
                <h3>Oscillator 2</h3>
                <ul>
        """)
        
        for param, value in params2.items():
            f.write(f"<li>{param}: {value}</li>\n")
        
        f.write(f"""
                </ul>
            </div>
            
            <div class="param-block">
                <h3>Coupling</h3>
                <ul>
                    <li>k12 (Repression of Osc1 by Osc2): {k12}</li>
                    <li>k21 (Repression of Osc2 by Osc1): {k21}</li>
                </ul>
                
                <h3>Stability Analysis</h3>
                <ul>
                    <li>k11*k22 - k12*k21 = {k11*k22 - k12*k21:.6f}</li>
                    <li>Predicted stability: {stability}</li>
                </ul>
            </div>
        </div>
        
        <h2>Results</h2>
        """)
        
        # Add each plot to the dashboard
        for plot_file, metadata in plots_metadata.items():
            f.write(f"""
        <div class="plot-section">
            <h3>{metadata['title']}</h3>
            <img src="{plot_file}" class="plot-img">
            <p class="plot-caption">{metadata['description']}</p>
        </div>
            """)
        
        f.write("""
    </div>
</body>
</html>
        """)
    
    print(f"Results saved to folder: {run_folder}")
    print(f"Dashboard available at: {run_folder}/dashboard.html")
    
    return run_folder

def main():
    """Main function to run different simulation scenarios"""
    # Create base results directory
    os.makedirs("results", exist_ok=True)
    
    # Base parameters for both oscillators (initially identical)
    base_params = {
        'a': 5.0, 'A': 1.0, 'b': 0.1, 
        'alpha': 0.2, 'beta': 0.02, 'k_self': 0.05
    }
    
    # Run each scenario and terminate after visualization
    
    # 1. Symmetric Weak Coupling
    print("\nScenario 1: Symmetric Weak Coupling")
    params1 = base_params.copy()
    params2 = base_params.copy()
    weak_coupling = {'k12': 0.01, 'k21': 0.01}
    t, X1, Y1, X2, Y2, results = run_simulation(
        params1, params2, weak_coupling, t_end=2000,
        run_description="Symmetric weak coupling where both oscillators weakly inhibit each other. "
                        "Expected to show phase locking with potential phase difference."
    )
    visualize_results(
        t, X1, Y1, X2, Y2, params1, params2, weak_coupling, 
        filename_prefix='weak_symmetric',
        run_description=results["description"]
    )
    print("Simulation 1 complete. Press Enter to continue to the next simulation...")
    input()
    
    # 2. Symmetric Strong Coupling (Stable)
    print("\nScenario 2: Symmetric Strong Coupling (Stable)")
    strong_stable = {'k12': 0.04, 'k21': 0.04}  # k11*k22 > k12*k21
    t, X1, Y1, X2, Y2, results = run_simulation(
        params1, params2, strong_stable, t_end=2000,
        run_description="Symmetric strong coupling that satisfies the stability condition (k11*k22 > k12*k21). "
                        "Expected to show stronger phase locking and potential synchronization."
    )
    visualize_results(
        t, X1, Y1, X2, Y2, params1, params2, strong_stable,
        filename_prefix='strong_symmetric_stable',
        run_description=results["description"]
    )
    print("Simulation 2 complete. Press Enter to continue to the next simulation...")
    input()
    
    # 3. Symmetric Strong Coupling (Unstable)
    print("\nScenario 3: Symmetric Strong Coupling (Unstable)")
    strong_unstable = {'k12': 0.06, 'k21': 0.06}  # k11*k22 < k12*k21
    t, X1, Y1, X2, Y2, results = run_simulation(
        params1, params2, strong_unstable, t_end=2000,
        run_description="Symmetric strong coupling that violates the stability condition (k11*k22 < k12*k21). "
                        "Expected to show potential competitive exclusion where one oscillator dominates."
    )
    visualize_results(
        t, X1, Y1, X2, Y2, params1, params2, strong_unstable,
        filename_prefix='strong_symmetric_unstable',
        run_description=results["description"]
    )
    print("Simulation 3 complete. Press Enter to continue to the next simulation...")
    input()
    
    # 4. Asymmetric Coupling
    print("\nScenario 4: Asymmetric Coupling")
    asymmetric = {'k12': 0.03, 'k21': 0.01}
    t, X1, Y1, X2, Y2, results = run_simulation(
        params1, params2, asymmetric, t_end=2000,
        run_description="Asymmetric coupling where one oscillator inhibits the other more strongly. "
                        "Expected to show more complex phase relationships and potential driving behavior."
    )
    visualize_results(
        t, X1, Y1, X2, Y2, params1, params2, asymmetric,
        filename_prefix='asymmetric',
        run_description=results["description"]
    )
    print("Simulation 4 complete. Press Enter to continue to the next simulation...")
    input()
    
    # 5. Different Initial Phases - In Phase
    print("\nScenario 5a: Different Initial Phases - In Phase")
    p1, q1 = calculate_steady_state_single(
        [params1['a'], params1['A'], params1['b'], 
         params1['alpha'], params1['beta'], params1['k_self']]
    )
    p2, q2 = calculate_steady_state_single(
        [params2['a'], params2['A'], params2['b'], 
         params2['alpha'], params2['beta'], params2['k_self']]
    )
    
    # Start in-phase
    initial_in_phase = [p1 * 1.1, q1 * 1.1, p2 * 1.1, q2 * 1.1]
    t, X1, Y1, X2, Y2, results = run_simulation(
        params1, params2, weak_coupling, t_end=2000, initial_state=initial_in_phase,
        run_description="Starting with in-phase initial conditions (both oscillators perturbed in the same direction). "
                        "Tests how initial conditions affect the final phase relationship."
    )
    visualize_results(
        t, X1, Y1, X2, Y2, params1, params2, weak_coupling,
        filename_prefix='in_phase_initial',
        run_description=results["description"]
    )
    print("Simulation 5a complete. Press Enter to continue to the next simulation...")
    input()
    
    # 6. Different Initial Phases - Anti Phase
    print("\nScenario 5b: Different Initial Phases - Anti Phase")
    # Start anti-phase
    initial_anti_phase = [p1 * 1.2, q1 * 1.2, p2 * 0.8, q2 * 0.8]
    t, X1, Y1, X2, Y2, results = run_simulation(
        params1, params2, weak_coupling, t_end=2000, initial_state=initial_anti_phase,
        run_description="Starting with anti-phase initial conditions (oscillators perturbed in opposite directions). "
                        "Tests how initial conditions affect the final phase relationship."
    )
    visualize_results(
        t, X1, Y1, X2, Y2, params1, params2, weak_coupling,
        filename_prefix='anti_phase_initial',
        run_description=results["description"]
    )
    print("All simulations complete!")

if __name__ == "__main__":
    main() 