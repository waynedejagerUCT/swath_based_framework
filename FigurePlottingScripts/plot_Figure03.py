

#!/usr/bin/env python3
"""
Script to plot frequency distributions of brightness temperatures and gradient ratios
for different Antarctic sea-ice types using AMSR2 data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = '/home/waynedj/Projects/ecice/processing'

# Input files for different ice types
ICE_TYPE_FILES = {
    'ow':  'AMSR2-Dist-OW-AQ2_07_08.txt',    # Open Water
    'fyi': 'AMSR2-Dist-FYI-AQ2_04_06.txt',   # First-Year Ice
    'yi':  'AMSR2-Dist-YI-WRSplnnew.txt',    # Young Ice
    'myi': 'AMSR2-Dist-MYI-AQ2_01_03.txt'    # Multi-Year Ice
}

# Parameters to analyze
TB_PARAMETERS = ['BRT37H', 'BRT37V', 'BRT19H', 'GD37v19v']

# Parameter label mapping for plotting
PARAM_LABELS = {
    'BRT37H': 'T$_{b}$ (37H)',
    'BRT37V': 'T$_{b}$ (37V)',
    'BRT19H': 'T$_{b}$ (19H)',
    'GD37v19v': 'GR(37V,19V)'
}

# Plot configuration
PLOT_CONFIG = {
    'colors': {
        'ow': '#0072B2',    # Open Water
        'fyi': '#E69F00',  # First-Year Ice
        'yi': '#009E73',    # Young Ice
        'myi': "#FF0000"     # Multi-Year Ice
    },
    'y_limits': {
        'BRT37H': (0, 0.04),
        'BRT37V': (0, 0.08),
        'BRT19H': (0, 0.05),
        'GD37v19v': (0, 0.13)
    },
    'x_limits': {
        'BRT37H': (75, 275),
        'BRT37V': (75, 275),
        'BRT19H': (75, 275),
        'GD37v19v': (-0.12, 0.02)
    }
}

def load_data():
    """Load and preprocess data for all ice types."""
    dataframes = {}
    for ice_type, filename in ICE_TYPE_FILES.items():
        file_path = os.path.join(DATA_DIR, filename)
        # Read data and replace zeros with NaN
        df = pd.read_csv(file_path, delim_whitespace=True)[TB_PARAMETERS]
        df = df.replace(0, np.nan)  # Replace zero values with NaN
        dataframes[ice_type] = df
    return dataframes

def setup_figure():
    """Create and configure the figure and subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    #fig.suptitle('Frequency Distribution of Antarctic Sea-ice Type', fontsize=18)
    fig.patch.set_facecolor('white')
    
    # Configure all subplots
    for ax in axes.flatten():
        ax.grid(True, linestyle=':', color='k', alpha=0.5)
        ax.set_ylabel('Relative frequency', fontsize=18)
    
    return fig, axes.flatten()

def plot_distributions(dataframes):
    """Create distribution plots for all parameters and ice types."""
    fig, axes = setup_figure()
    
    # Plot each parameter
    for idx, column in enumerate(TB_PARAMETERS):
        ax = axes[idx]
        
        # Plot each ice type
        for ice_type, df in dataframes.items():
            x_range = np.linspace(df[column][0], df[column][1], len(df[column].iloc[3:]))
            ax.plot(x_range, df[column].iloc[3:], 
                   alpha=1, 
                   color=PLOT_CONFIG['colors'][ice_type], 
                   label=f'{ice_type.upper()}')
        
        # Configure subplot
        # Add units to x-label except for gradient ratio
        units = '' if column == 'GD37v19v' else ' [K]'
        ax.set_xlabel(f'{PARAM_LABELS[column]}{units}', fontsize=18)
        ax.legend(fontsize=14, loc='upper left')
        ax.text(0.96, 0.96, f'({chr(97+idx)})', 
                transform=ax.transAxes, fontsize=20, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=1, edgecolor='black'))
        
        ax.set_ylim(PLOT_CONFIG['y_limits'][column])
        ax.set_xlim(PLOT_CONFIG['x_limits'][column])
    
    return fig

def main():
    """Main function to execute the plotting workflow."""
    # Load data
    dataframes = load_data()
    
    # Create plots
    fig = plot_distributions(dataframes)
    
    # Final adjustments and save
    plt.tight_layout()
    
    # Save the figure first
    plt.savefig('/home/waynedj/Projects/publication/paper_03/figure01.png', 
                dpi=400, bbox_inches='tight', format='png')
    
    # Then display it
    plt.show()


# Execute if run as a script
if __name__ == '__main__':
    main()






