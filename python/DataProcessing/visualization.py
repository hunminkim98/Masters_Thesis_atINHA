"""
Visualization functions
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import shapiro
from statsmodels.graphics.gofplots import qqplot
import pingouin as pg


# Bar plot with standard deviation error bars
def bar_plot_with_std_from_dataframe(dataframe, x_axis, y_axis, std_col, palette=None, title=None, x_label=None, y_label=None):
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

def qq_plot_with_pingouin(data, dist='norm', sparams=(), confidence=0.95, square=True, ax=None, **kwargs):
    """
    Generate a Q-Q plot using Pingouin's qqplot function.

    Args:
        data: Series or array-like data to plot.
        dist: str or stats.distributions instance, optional
              Distribution or distribution function name. The default is 'norm'.
        sparams: tuple, optional
                 Distribution-specific shape parameters.
        confidence: float or False
                    Confidence level for point-wise confidence envelope (e.g., 0.95).
                    Set to False to disable.
        square: bool
                If True (default), ensure equal aspect ratio between X and Y axes.
        ax: matplotlib axes, optional
            Axis on which to draw the plot. If None, a new figure/axis is created.
        **kwargs: dict, optional
                  Optional argument(s) passed to matplotlib.pyplot.scatter().

    Returns:
        matplotlib.axes._axes.Axes: Matplotlib Axes instance.

    Notes:
        - Cleans NaN values from the input data before plotting.
        - See pingouin.qqplot documentation for more details.
    """
    # Clean data (remove NaNs)
    if isinstance(data, pd.Series):
        data = data.dropna()
    elif isinstance(data, pd.DataFrame):
        # If DataFrame, assume the first column is the data
        # Or raise an error/require specific column name
        data = data.iloc[:, 0].dropna() 
    else:
        data = np.array(data)
        data = data[~np.isnan(data)]

    # Create plot using pingouin.qqplot
    ax = pg.qqplot(
        x=data,
        dist=dist,
        sparams=sparams,
        confidence=confidence,
        square=square,
        ax=ax,
        **kwargs
    )

    # Add title if not provided through kwargs or existing ax
    if ax and not ax.get_title():
        ax.set_title(f'Q-Q Plot ({dist.capitalize()} Distribution)')

    plt.tight_layout()
    plt.show()

    return ax

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

def anova_boxplot(dataframe, x_axis, y_axis, palette=None, title=None, x_label=None, y_label=None):
    """
    Generates and displays a box plot suitable for visualizing ANOVA results.

    Shows the distribution of the dependent variable (y_axis) for each
    group defined by the independent variable (x_axis).

    Args:
        dataframe (pd.DataFrame): DataFrame containing the data.
        x_axis (str): Column name for the independent variable (grouping factor).
        y_axis (str): Column name for the dependent variable.
        palette (str, list, or dict, optional): Seaborn color palette for the boxes.
                                               Defaults to None (Seaborn default).
        title (str, optional): Plot title. Defaults to a generated title.
        x_label (str, optional): X-axis label. Defaults to x_axis name.
        y_label (str, optional): Y-axis label. Defaults to y_axis name.

    Returns:
        matplotlib.axes._axes.Axes: The Matplotlib Axes object with the plot.
    """
    plt.figure(figsize=(6, 5))  # Adjust figure size as needed

    ax = sns.boxplot(
        data=dataframe,
        x=x_axis,
        y=y_axis,
        palette=palette,
        hue=x_axis,  # Use x_axis as hue to apply different colors per group
        legend=False # Hide legend as it's redundant with x-axis ticks
    )

    # Set title
    if title:
        plt.title(title)
    else:
        plt.title(f'Distribution of {y_axis} by {x_axis}')

    # Set labels
    plt.xlabel(x_label if x_label else x_axis)
    plt.ylabel(y_label if y_label else y_axis)

    plt.grid(True, linestyle='--', alpha=0.6, axis='y') # Add horizontal grid lines
    plt.tight_layout()
    plt.show()

    return ax

def mixed_anova_interaction_plot(data, dv, between, within, subject, 
                                 error_bars='se', palette='Set2', 
                                 title=None, figsize=(8, 6), 
                                 connect_lines=True, markers=True, 
                                 show_p_values=True, aov_results=None):
    """
    Create an interaction plot for Mixed ANOVA results showing group means across time points.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Long-format data used for the Mixed ANOVA analysis
    dv : str
        Name of the dependent variable column
    between : str
        Name of the between-subjects factor column
    within : str
        Name of the within-subjects factor column
    subject : str
        Name of the subject identifier column
    error_bars : str, optional
        Type of error bars to display ('se' for standard error, 'ci' for 95% confidence interval, 
        'sd' for standard deviation, or None for no error bars)
    palette : str or list
        Color palette for different groups
    title : str, optional
        Plot title (if None, a default title is generated)
    figsize : tuple, optional
        Figure size (width, height) in inches
    connect_lines : bool, optional
        Whether to connect points with lines
    markers : bool, optional
        Whether to show markers for data points
    show_p_values : bool, optional
        Whether to annotate with p-values from ANOVA results
    aov_results : pandas.DataFrame, optional
        Mixed ANOVA results dataframe for annotation
        
    Returns
    -------
    matplotlib.axes._axes.Axes
        The axes object with the plot
    """
    # Calculate summary statistics by group and time
    grouped = data.groupby([between, within])[dv].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate standard error
    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
    grouped['ci95'] = 1.96 * grouped['se']
    
    # Create the plot
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    
    # Choose the error bar type
    err_column = None
    if error_bars == 'se':
        err_column = 'se'
        err_label = 'Standard Error'
    elif error_bars == 'ci':
        err_column = 'ci95'
        err_label = '95% Confidence Interval'
    elif error_bars == 'sd':
        err_column = 'std'
        err_label = 'Standard Deviation'
    
    # Create line plot with error bars
    groups = grouped[between].unique()
    linestyles = ['-', '--', '-.', ':'] * (len(groups) // 4 + 1)
    markers_list = ['o', 's', '^', 'D', 'v', '*', 'X'] * (len(groups) // 7 + 1)
    
    # Define time point order
    time_order = ['Pre', 'Post1', 'Post2']
    
    for i, group in enumerate(groups):
        group_data = grouped[grouped[between] == group].copy()
        
        # Sort by time points to ensure correct ordering
        if all(tp in time_order for tp in group_data[within]):
            # Create order mapping and sort
            time_map = {tp: idx for idx, tp in enumerate(time_order)}
            group_data['time_order'] = group_data[within].map(time_map)
            group_data = group_data.sort_values('time_order')
        
        # Extract yerr for just this group
        yerr = None if err_column is None else group_data[err_column].values
        
        # Plot with appropriate settings
        if connect_lines and markers:
            plt.errorbar(group_data[within], group_data['mean'], 
                        yerr=yerr,
                        label=group, color=sns.color_palette(palette)[i], 
                        linestyle=linestyles[i], marker=markers_list[i], 
                        capsize=4, markersize=8, linewidth=2)
        elif connect_lines:
            plt.errorbar(group_data[within], group_data['mean'], 
                        yerr=yerr,
                        label=group, color=sns.color_palette(palette)[i], 
                        linestyle=linestyles[i], marker=None, 
                        capsize=4, linewidth=2)
        else:
            plt.errorbar(group_data[within], group_data['mean'], 
                        yerr=yerr,
                        label=group, color=sns.color_palette(palette)[i], 
                        linestyle='none', marker=markers_list[i], 
                        capsize=4, markersize=8)
    
    # Set correct order for x-axis ticks if they match expected time points
    if hasattr(plt.gca(), 'set_xticks') and all(tp in time_order for tp in plt.gca().get_xticks()):
        plt.gca().set_xticks(range(len(time_order)))
        plt.gca().set_xticklabels(time_order)
    
    # Add p-values annotations if requested and results provided
    if show_p_values and aov_results is not None:
        try:
            # Extract p-values (adjust column names based on pingouin's output)
            between_row = aov_results[aov_results['Source'] == between]
            within_row = aov_results[aov_results['Source'] == within]
            interaction_row = aov_results[aov_results['Source'] == 'Interaction']
            
            p_group = between_row['p-unc'].values[0] if not between_row.empty else float('nan')
            p_time = within_row['p-unc'].values[0] if not within_row.empty else float('nan')
            p_interaction = interaction_row['p-unc'].values[0] if not interaction_row.empty else float('nan')
            
            # Format p-values with significance stars
            def format_p_with_stars(p):
                if np.isnan(p):
                    return "p = N/A"
                p_formatted = f"p = {p:.4f}"
                if p < 0.001:
                    return p_formatted + " ***"
                elif p < 0.01:
                    return p_formatted + " **"
                elif p < 0.05:
                    return p_formatted + " *"
                else:
                    return p_formatted
            
            # Create annotation text
            ann_text = f"{between}: {format_p_with_stars(p_group)}\n"
            ann_text += f"{within}: {format_p_with_stars(p_time)}\n"
            ann_text += f"{between} × {within}: {format_p_with_stars(p_interaction)}"
            
            # Add text box with p-values
            plt.annotate(ann_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                        va='top', ha='left', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        except Exception as e:
            print(f"Could not add p-values: {e}")
    
    # Set title and labels
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    else:
        plt.title(f'Interaction Plot: {between} × {within}', fontsize=14, fontweight='bold')
    
    plt.xlabel(within, fontsize=12)
    plt.ylabel(dv, fontsize=12)
    
    # Add legend, grid, and styling
    plt.legend(title=between, loc='best', frameon=True, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add error bar type to y-axis label if error bars are shown
    if err_column is not None:
        plt.ylabel(f"{dv}\n(Error bars: {err_label})", fontsize=12)
    
    plt.tight_layout()
    plt.show()

def mixed_anova_distribution_plot(data, dv, between, within, subject, 
                                 layout='facet', palette='Set2', 
                                 title=None, figsize=(12, 8), 
                                 box_violin=True, swarm=True,
                                 means=True):
    """
    Create a comprehensive visualization of data distributions for Mixed ANOVA.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Long-format data used for the Mixed ANOVA analysis
    dv : str
        Name of the dependent variable column
    between : str
        Name of the between-subjects factor column
    within : str
        Name of the within-subjects factor column
    subject : str
        Name of the subject identifier column
    layout : str, optional
        Plot layout: 'facet' for facetgrid by time, 'trellis' for trellis by group
    palette : str or list
        Color palette for different groups
    title : str, optional
        Plot title (if None, a default title is generated)
    figsize : tuple, optional
        Figure size (width, height) in inches
    box_violin : bool, optional
        Whether to show box plots (True) or violin plots (False)
    swarm : bool, optional
        Whether to overlay a swarm plot of individual data points
    means : bool, optional
        Whether to show group means with error bars
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object with the plots
    """
    plt.figure(figsize=figsize)
    
    # Get unique values for factors
    times = data[within].unique()
    groups = data[between].unique()
    
    if layout == 'facet':
        # Create a facet grid by time points
        fig, axes = plt.subplots(1, len(times), figsize=figsize, sharey=True)
        
        for i, time in enumerate(times):
            ax = axes[i] if len(times) > 1 else axes
            time_data = data[data[within] == time]
            
            # Box or violin plot
            if box_violin:
                sns.boxplot(x=between, y=dv, hue=between, data=time_data, palette=palette, legend=False, ax=ax)
            else:
                sns.violinplot(x=between, y=dv, hue=between, data=time_data, palette=palette, 
                              inner=None, legend=False, ax=ax)
            
            # Add swarm plot
            if swarm:
                sns.swarmplot(x=between, y=dv, data=time_data, color='black', 
                             size=4, alpha=0.6, ax=ax)
            
            # Add means with confidence intervals if requested
            if means:
                grouped = time_data.groupby(between)[dv].agg(['mean', 'std', 'count']).reset_index()
                grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
                grouped['ci95'] = 1.96 * grouped['se']
                
                for j, group in enumerate(grouped[between]):
                    group_idx = grouped[between] == group
                    ax.errorbar(j, grouped.loc[group_idx, 'mean'].values[0], 
                               yerr=grouped.loc[group_idx, 'ci95'].values[0],
                               fmt='o', color='red', markersize=8, capsize=5, 
                               elinewidth=2, markeredgecolor='black')
            
            # Set titles
            ax.set_title(f'{within}: {time}', fontsize=12)
            
            # Only show y-label on the first subplot
            if i > 0:
                ax.set_ylabel('')
            else:
                ax.set_ylabel(dv, fontsize=12)
        
    else:  # trellis layout by group
        fig, axes = plt.subplots(len(groups), 1, figsize=figsize, sharex=True)
        
        for i, group in enumerate(groups):
            ax = axes[i] if len(groups) > 1 else axes
            group_data = data[data[between] == group]
            
            # Box or violin plot
            if box_violin:
                sns.boxplot(x=between, y=dv, hue=between, data=group_data, palette=palette, legend=False, ax=ax)
            else:
                sns.violinplot(x=between, y=dv, hue=between, data=group_data, palette=palette, 
                              inner=None, legend=False, ax=ax)
            
            # Add swarm plot
            if swarm:
                sns.swarmplot(x=between, y=dv, data=group_data, color='black', 
                             size=4, alpha=0.6, ax=ax)
            
            # Add means with confidence intervals if requested
            if means:
                grouped = group_data.groupby(between)[dv].agg(['mean', 'std', 'count']).reset_index()
                grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
                grouped['ci95'] = 1.96 * grouped['se']
                
                # Sort by time points to ensure correct ordering
                time_order = {time: idx for idx, time in enumerate(times)}
                grouped['order'] = grouped[between].map(time_order)
                grouped = grouped.sort_values('order')
                
                for j, time in enumerate(times):
                    time_idx = grouped[between] == time
                    if any(time_idx):
                        ax.errorbar(j, grouped.loc[time_idx, 'mean'].values[0], 
                                  yerr=grouped.loc[time_idx, 'ci95'].values[0],
                                  fmt='o', color='red', markersize=8, capsize=5, 
                                  elinewidth=2, markeredgecolor='black')
            
            # Set titles
            ax.set_title(f'{between}: {group}', fontsize=12)
            
            # Only show x-label on the last subplot
            if i < len(groups) - 1:
                ax.set_xlabel('')
            else:
                ax.set_xlabel(between, fontsize=12)
                
            # Only show y-label on the middle subplot or the first if only one
            if len(groups) > 1 and i == len(groups) // 2 or i == 0:
                ax.set_ylabel(dv, fontsize=12)
            else:
                ax.set_ylabel('')
    
    # Set main title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    else:
        fig.suptitle(f'Distribution of {dv} by {between} and {within}', 
                    fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90 if title else 0.95)  # Adjust for suptitle
    plt.show()