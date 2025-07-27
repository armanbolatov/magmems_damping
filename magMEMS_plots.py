import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Times New Roman', 'DejaVu Serif'],
    'axes.linewidth': 1.2,
    'grid.alpha': 0.7,
    'lines.linewidth': 2,
    'figure.figsize': (10, 6),
    'text.usetex': False,
    'mathtext.fontset': 'stix',
    'mathtext.default': 'it'
})

class MagMEMSAnalyzer:
    def __init__(self, xi=0.0):
        self.xi = xi
        self.K_star_approx = 0.19464  # Approximate pull-in threshold for xi=0
        
    def galerkin_coefficient_a(self, K):
        """Calculate Galerkin coefficient 'a' for given K (xi=0 case)"""
        if K >= self.K_star_approx or 10 - 6*K <= 0:
            return None
            
        sqrt_term = np.sqrt(10 - 6*K)
        cos_arg = (28 - 9*K) / ((10 - 6*K)**(3/2))
        
        if abs(cos_arg) > 1:
            return None
            
        a = (2*sqrt_term/3) * np.cos((1/3) * np.arccos(cos_arg) + 2*np.pi/3) + 4/3
        return a
    
    def undamped_frequency(self, a, K):
        """Calculate undamped frequency from Galerkin analysis"""
        if a is None or a >= 1:
            return None
        omega_sq = (1 - 2*a - K*a*self.xi**2 + K*self.xi**2 - K*self.xi) / (1 - a)
        return np.sqrt(max(0, omega_sq))
    
    def undamped_solution(self, t, a, omega):
        """Undamped Galerkin solution: y(t) = 2a*sin²(ωt/2)"""
        if a is None or omega is None:
            return np.zeros_like(t)
        return 2*a * np.sin(omega*t/2)**2
    
    def steady_state_solution(self, K, gamma):
        """Calculate steady-state solution y_ss from equilibrium equation"""
        def equilibrium_eq(y_ss):
            if y_ss >= 0.995 or y_ss <= 0:
                return 1e6
            return ((1 + K*self.xi**2/2)*y_ss - 
                   (K/(1-y_ss) - K*self.xi + K*self.xi**2/2))
        
        for y0 in [0.01, 0.05, 0.1, 0.2, 0.3]:
            try:
                y_ss = fsolve(equilibrium_eq, y0, xtol=1e-12)[0]
                if 0 < y_ss < 0.995 and abs(equilibrium_eq(y_ss)) < 1e-10:
                    return y_ss
            except:
                continue
        return 0.01
    
    def damped_galerkin_solution(self, t, K, gamma):
        """Improved Galerkin approximation for damped case"""
        a = self.galerkin_coefficient_a(K)
        omega_undamped = self.undamped_frequency(a, K)
        
        if a is None or omega_undamped is None:
            return np.zeros_like(t)
        
        y_ss = self.steady_state_solution(K, gamma)
        alpha = gamma / 2
        omega_d = omega_undamped
        decay_envelope = np.exp(-alpha * t)
        oscillatory_part = (1 - np.cos(omega_d * t))
        
        return y_ss * (1 - decay_envelope) + a * decay_envelope * oscillatory_part
    
    def damped_ode_system(self, t, state, K, gamma):
        """ODE system for damped case: [y, y']"""
        y, y_dot = state
        
        if y >= 0.99:
            y = 0.99
            
        y_ddot = (K/(1-y) - K*self.xi + K*self.xi**2/2 - 
                 (1 + K*self.xi**2/2)*y - gamma*y_dot)
        
        return [y_dot, y_ddot]
    
    def solve_damped(self, t_span, K, gamma):
        """Solve damped ODE numerically"""
        y0 = [0.0, 0.0]
        sol = solve_ivp(self.damped_ode_system, [0, t_span[-1]], y0, 
                       t_eval=t_span, args=(K, gamma), 
                       method='RK45', rtol=1e-8, atol=1e-10)
        
        return sol.y[0] if sol.success else np.zeros_like(t_span)

def create_detailed_comparison():
    """Create detailed comparison plot"""
    analyzer = MagMEMSAnalyzer(xi=0.0)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    t = np.linspace(0, 25, 1000)
    
    K_fractions = [0.5, 0.7, 0.9]
    K_values = [f * analyzer.K_star_approx for f in K_fractions]
    colors = ['#0066CC', '#000000', '#FF1493']
    
    for i, (K, K_frac) in enumerate(zip(K_values, K_fractions)):
        a = analyzer.galerkin_coefficient_a(K)
        omega = analyzer.undamped_frequency(a, K)
        
        if a is not None and omega is not None:
            y_undamped = analyzer.undamped_solution(t, a, omega)
            ax.plot(t, y_undamped, color=colors[i], linestyle='-', linewidth=2.5,
                   label=r'$\tilde{y}(t)$ for $K = %s K_0^*$' % K_frac)
            
            for j, gamma in enumerate([0.02, 0.05]):
                y_damped = analyzer.solve_damped(t, K, gamma)
                linestyle = '--' if j == 0 else ':'
                alpha = 0.8 if j == 0 else 0.6
                ax.plot(t, y_damped, color=colors[i], linestyle=linestyle, 
                       linewidth=2, alpha=alpha,
                       label=r'damped $\gamma=%s$ for $K = %s K_0^*$' % (gamma, K_frac))
    
    ax.set_xlabel(r'$t$ [-]', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'displacement [-]', fontsize=13, fontweight='bold')
    ax.set_title(r'$\xi = 0$, $K_0^* = 0.203632188...$', fontsize=13, fontweight='bold')
    
    ax.grid(True, alpha=0.7, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 0.7)
    ax.set_xticks(np.arange(0, 26, 5))
    ax.set_yticks(np.arange(0, 0.8, 0.1))
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10,
             frameon=True, fancybox=True, shadow=True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    plt.savefig('figs/magMEMS_detailed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_amplitude_vs_excitation():
    """Create amplitude vs excitation parameter plot"""
    analyzer = MagMEMSAnalyzer(xi=0.0)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    K_range = np.linspace(0.1, 0.95, 20) * analyzer.K_star_approx
    max_amplitudes_undamped = []
    max_amplitudes_damped = []
    
    for K in K_range:
        a = analyzer.galerkin_coefficient_a(K)
        if a is not None:
            max_amplitudes_undamped.append(2*a)
            y_damped = analyzer.solve_damped(np.linspace(0, 20, 800), K, 0.05)
            max_amplitudes_damped.append(np.max(y_damped) if len(y_damped) > 0 else 0)
        else:
            max_amplitudes_undamped.append(0)
            max_amplitudes_damped.append(0)
    
    ax.plot(K_range/analyzer.K_star_approx, max_amplitudes_undamped, 'k-', 
            linewidth=2, label=r'Undamped max amplitude')
    ax.plot(K_range/analyzer.K_star_approx, max_amplitudes_damped, 'r--', 
            linewidth=2, label=r'Damped max amplitude ($\gamma=0.05$)')
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, label=r'Pull-in threshold')
    
    ax.set_xlabel(r'$K/K_0^*$ [-]', fontsize=12)
    ax.set_ylabel(r'Maximum Amplitude [-]', fontsize=12)
    ax.set_title(r'Amplitude vs Excitation Parameter', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.savefig('figs/amplitude_vs_excitation.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_ode_vs_galerkin_damped():
    """Create ODE vs Galerkin comparison for damped case"""
    analyzer = MagMEMSAnalyzer(xi=0.0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    t_comp = np.linspace(0, 100, 3000)
    
    test_cases = [
        {'K': 0.5 * analyzer.K_star_approx, 'gamma': 0.02, 'color': 'blue'},
        {'K': 0.5 * analyzer.K_star_approx, 'gamma': 0.05, 'color': 'green'},
        {'K': 0.7 * analyzer.K_star_approx, 'gamma': 0.02, 'color': 'red'},
        {'K': 0.7 * analyzer.K_star_approx, 'gamma': 0.05, 'color': 'purple'}
    ]
    
    for case in test_cases:
        K, gamma = case['K'], case['gamma']
        color = case['color']
        
        y_ode = analyzer.solve_damped(t_comp, K, gamma)
        y_gal = analyzer.damped_galerkin_solution(t_comp, K, gamma)
        
        ax1.plot(t_comp, y_ode, '--', color=color, linewidth=2,
                label=r'ODE: $K=%.3fK_0^*$, $\gamma=%.2f$' % (K/analyzer.K_star_approx, gamma))
        ax1.plot(t_comp, y_gal, '-', color=color, linewidth=2, alpha=0.7,
                label=r'Galerkin: $K=%.3fK_0^*$, $\gamma=%.2f$' % (K/analyzer.K_star_approx, gamma))
        
        error = np.abs(y_ode - y_gal)
        ax2.plot(t_comp, error, '-', color=color, linewidth=2,
                label=r'$K=%.3fK_0^*$, $\gamma=%.2f$' % (K/analyzer.K_star_approx, gamma))
    
    ax1.set_xlabel(r'$t$ [-]', fontsize=12)
    ax1.set_ylabel(r'$y(t)$ [-]', fontsize=12)
    ax1.set_title(r'ODE vs Galerkin Solutions for Damped Case', fontsize=12)
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.7)
    ax1.set_xlim(0, 100)
    
    ax2.set_xlabel(r'$t$ [-]', fontsize=12)
    ax2.set_ylabel(r'$|y_{ODE} - y_{Galerkin}|$ [-]', fontsize=12)
    ax2.set_title(r'Absolute Error Between ODE and Galerkin', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.7)
    ax2.set_xlim(0, 100)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('figs/ode_vs_galerkin_damped.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Creating magMEMS damping analysis plots...")
    
    create_detailed_comparison()
    create_amplitude_vs_excitation()
    create_ode_vs_galerkin_damped()