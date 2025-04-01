"""
Visualization functions
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import shapiro
from statsmodels.graphics.gofplots import qqplot


# Bar plot with standard deviation error bars
def box_plot_with_std_from_dataframe(dataframe, x_axis, y_axis, std_col, palette=None, title=None, x_label=None, y_label=None):
    """
    Generates and displays a bar plot with standard deviation error bars from a pandas DataFrame.
    Allows specifying a color palette for the bars.
    
    Args:
        dataframe: DataFrame containing the data
        x_axis: Column name for x-axis categories
        y_axis: Column name for y-axis values (means/averages)
        std_col: Column name for standard deviation values
        palette: Optional color palette for the bars (seaborn palette name, list of colors, or dict). 
                 If None, seaborn default palette is used.
        title: Plot title
        x_label: X-axis label (defaults to x_axis name if None)
        y_label: Y-axis label (defaults to y_axis name if None)
    """
    plt.figure(figsize=(4, 4))  # Set figure size
    
    # Responding to seaborn deprecation warning:
    # Pass x_axis as hue and set legend=False (similar effect to just using palette)
    ax = sns.barplot(
        data=dataframe, 
        x=x_axis, 
        y=y_axis, 
        hue=x_axis,  # Use x_axis as hue to apply different colors
        palette=palette,
        legend=False  # Don't show redundant legend
    )
    
    # Add error bars for standard deviations (simplified logic)
    for i, bar in enumerate(ax.patches):
        x_pos = bar.get_x() + bar.get_width()/2
        y_pos = bar.get_height()
        # Check if std_col exists before accessing
        std = dataframe.iloc[i][std_col]
        plt.errorbar(x=x_pos, y=y_pos, yerr=std, fmt='none', color='black', capsize=5)

    # Set title
    if title:
        plt.title(title)
    else:
        plt.title(f'{y_axis} by {x_axis}')

    # Set labels
    plt.xlabel(x_label if x_label else x_axis)
    plt.ylabel(y_label if y_label else y_axis)

    plt.tight_layout()
    plt.show()


def qq_plot(data, title=None, color='blue', line_color='red'):
    """
    Generates and displays a Q-Q plot to assess if data follows a normal distribution.
    
    Args:
        data: Series or array-like data to plot
        title: Plot title (defaults to 'Q-Q Plot' if None)
        color: Color for the data points
        line_color: Color for the reference line
    """
    # Clean data (remove NaNs)
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        data = data.dropna()
    else:
        data = np.array(data)
        data = data[~np.isnan(data)]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Create Q-Q plot with no additional parameters
    fig = qqplot(data, line='s', ax=ax)
    
    # Customize plot after creation
    # First point is the scatter plot, second is the line
    if len(ax.get_lines()) > 0:
        ax.get_lines()[0].set_color(color)
        ax.get_lines()[0].set_marker('.')
        ax.get_lines()[0].set_linestyle('none')
    
    # Change the color of the reference line (usually the second line)
    if len(ax.get_lines()) > 1:
        ax.get_lines()[1].set_color(line_color)
        ax.get_lines()[1].set_linewidth(2)
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Q-Q Plot')
    
    # Add Shapiro-Wilk test results if requested
    if len(data) >= 3:  # Shapiro test requires at least 3 data points
        stat, p = shapiro(data)
        text = f'Shapiro-Wilk Test\nStatistic: {stat:.4f}\np-value: {p:.4f}'
        if p < 0.05:
            text += '\nData is not normally distributed'
        else:
            text += '\nData appears normally distributed'
            
        ax.text(0.05, 0.95, text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return fig


def density_plot(data, title=None, color='black', fill=True, rug=True, xlim=None, mean_line=True, median_line=True):
    """
    Generates and displays a density plot for visualizing data distribution.
    
    Args:
        data: Series or array-like data to plot
        title: Plot title (defaults to 'Density Plot' if None)
        color: Color for the density curve
        fill: Whether to fill the area under the curve
        rug: Whether to show the rug plot (individual observations)
        xlim: Optional tuple for x-axis limits (min, max)
        mean_line: Whether to show the mean line
        median_line: Whether to show the median line
    
    Returns:
        The figure object
    """
    # Clean data (remove NaNs)
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        data = data.dropna()
    else:
        data = np.array(data)
        data = data[~np.isnan(data)]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Create density plot
    sns.kdeplot(
        data=data,
        color=color,
        fill=fill,
        ax=ax
    )
    
    # Add rug plot (individual observations at the bottom)
    if rug:
        sns.rugplot(data=data, color=color, alpha=0.5, ax=ax)
    
    # Add mean and median lines
    if mean_line:
        mean_val = np.mean(data)
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    
    if median_line:
        median_val = np.median(data)
        ax.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.2f}')
    
    # Calculate skewness
    from scipy.stats import skew
    skewness = skew(data)
    
    # Set title with skewness value
    if title:
        ax.set_title(f"{title} (Skewness: {skewness:.4f})")
    else:
        ax.set_title(f"Density Plot (Skewness: {skewness:.4f})")
    
    # Set x-axis limits if provided
    if xlim:
        ax.set_xlim(xlim)
    
    # Add legend if we have mean or median lines
    if mean_line or median_line:
        ax.legend()
    
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return fig

