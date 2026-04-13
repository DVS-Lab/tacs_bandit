"""
plotting_utils.py — Shared visualization utilities for tACS Bandit study

Provides consistent plotting configuration and helper functions used across modules.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import plotly.graph_objects as go

from config import (
    AGE_MIN,
    AGE_MAX,
    AGE_COLORSCALE,
    COLOR_SHAM,
    COLOR_ACTIVE,
    COLOR_GOLD,
    PLOTLY_TEMPLATE,
    FONT_FAMILY,
)


# =============================================================================
# Age Color Utilities
# =============================================================================

def _get_age_colormap():
    """Get matplotlib colormap for age gradient (blue → gold)."""
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list('blue_gold', ['#1565C0', '#FFB300'])


def age_to_rgb(age: float, age_min: float = AGE_MIN, age_max: float = AGE_MAX) -> str:
    """
    Convert age to RGB color string for Plotly.
    
    Parameters
    ----------
    age : float
        Age value
    age_min, age_max : float
        Age range for normalization
    
    Returns
    -------
    str : RGB color string, e.g., 'rgb(21, 101, 192)'
    """
    cmap = _get_age_colormap()
    norm = np.clip((age - age_min) / (age_max - age_min), 0, 1)
    rgba = cmap(norm)
    return f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'


def age_to_rgba(age: float, alpha: float = 1.0,
                age_min: float = AGE_MIN, age_max: float = AGE_MAX) -> str:
    """
    Convert age to RGBA color string for Plotly.
    
    Parameters
    ----------
    age : float
        Age value
    alpha : float
        Opacity (0-1)
    age_min, age_max : float
        Age range for normalization
    
    Returns
    -------
    str : RGBA color string
    """
    cmap = _get_age_colormap()
    norm = np.clip((age - age_min) / (age_max - age_min), 0, 1)
    rgba = cmap(norm)
    return f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{alpha})'


def get_age_colors(ages: List[float], age_min: float = AGE_MIN,
                   age_max: float = AGE_MAX) -> List[str]:
    """
    Convert list of ages to RGB color strings.
    
    Parameters
    ----------
    ages : list
        List of age values
    age_min, age_max : float
        Age range for normalization
    
    Returns
    -------
    list : RGB color strings
    """
    return [age_to_rgb(a, age_min, age_max) for a in ages]


# =============================================================================
# Plotly Configuration
# =============================================================================

def apply_standard_layout(
    fig: go.Figure,
    width: int = 800,
    height: int = 500,
    title: Optional[str] = None,
    show_legend: bool = True
) -> go.Figure:
    """
    Apply standard layout configuration to a Plotly figure.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure object
    width, height : int
        Figure dimensions
    title : str, optional
        Figure title
    show_legend : bool
        Whether to show legend
    
    Returns
    -------
    go.Figure : Modified figure
    """
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(family=FONT_FAMILY, size=13),
        width=width,
        height=height,
        showlegend=show_legend,
        margin=dict(l=60, r=60, t=60, b=60),
    )
    
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15)))
    
    return fig


def apply_standard_axes(
    fig: go.Figure,
    show_grid: bool = False,
    zero_line: bool = False
) -> go.Figure:
    """
    Apply standard axis styling to a Plotly figure.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure object
    show_grid : bool
        Whether to show grid lines
    zero_line : bool
        Whether to show zero line
    
    Returns
    -------
    go.Figure : Modified figure
    """
    fig.update_xaxes(
        showgrid=show_grid,
        zeroline=zero_line,
        showline=True,
        linewidth=1,
        linecolor='black'
    )
    
    fig.update_yaxes(
        showgrid=show_grid,
        zeroline=zero_line,
        showline=True,
        linewidth=1,
        linecolor='black'
    )
    
    return fig


# =============================================================================
# Colorbar Utilities
# =============================================================================

def add_age_colorbar(
    fig: go.Figure,
    x: float = 1.05,
    y: float = 0.5,
    length: float = 0.6,
    title: str = 'Age'
) -> go.Figure:
    """
    Add an age colorbar to a Plotly figure.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure object
    x, y : float
        Colorbar position
    length : float
        Colorbar length (fraction of plot)
    title : str
        Colorbar title
    
    Returns
    -------
    go.Figure : Modified figure with colorbar trace
    """
    dummy_ages = np.linspace(AGE_MIN, AGE_MAX, 50)
    
    fig.add_trace(go.Scatter(
        x=[None] * 50,
        y=[None] * 50,
        mode='markers',
        marker=dict(
            size=0.1,
            color=dummy_ages,
            colorscale=AGE_COLORSCALE,
            cmin=AGE_MIN,
            cmax=AGE_MAX,
            colorbar=dict(
                x=x,
                y=y,
                len=length,
                thickness=10,
                title=title,
                titleside='right',
                tickfont=dict(size=11)
            )
        ),
        showlegend=False,
        hoverinfo='skip',
    ))
    
    return fig


# =============================================================================
# Common Plot Components
# =============================================================================

def create_paired_line_trace(
    x_coords: List[float],
    y_coords: List[float],
    color: str,
    hover_text: str = '',
    opacity: float = 0.7,
    line_width: float = 1.5,
    marker_size: int = 8
) -> go.Scatter:
    """
    Create a paired-comparison line trace (for spaghetti plots).
    
    Parameters
    ----------
    x_coords : list
        X coordinates (typically [0, 1] for two conditions)
    y_coords : list
        Y coordinates (values at each condition)
    color : str
        Line/marker color
    hover_text : str
        Hover text
    opacity : float
        Trace opacity
    line_width : float
        Line width
    marker_size : int
        Marker size
    
    Returns
    -------
    go.Scatter : Plotly scatter trace
    """
    return go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='lines+markers',
        line=dict(color=color, width=line_width),
        marker=dict(size=marker_size, color=color,
                    line=dict(width=0.5, color='white')),
        opacity=opacity,
        text=hover_text,
        hoverinfo='text+y',
        showlegend=False,
    )


def create_mean_marker_trace(
    x: float,
    y: float,
    error: float,
    label: str,
    color: str = '#404040',
    marker_size: int = 14,
    symbol: str = 'diamond'
) -> go.Scatter:
    """
    Create a group mean marker with error bars.
    
    Parameters
    ----------
    x : float
        X coordinate
    y : float
        Mean value
    error : float
        Error bar size (typically SEM)
    label : str
        Condition label for hover
    color : str
        Marker color
    marker_size : int
        Marker size
    symbol : str
        Marker symbol
    
    Returns
    -------
    go.Scatter : Plotly scatter trace
    """
    return go.Scatter(
        x=[x],
        y=[y],
        mode='markers',
        marker=dict(size=marker_size, color=color, symbol=symbol,
                    line=dict(width=2, color='white')),
        error_y=dict(type='data', array=[error], visible=True,
                     color=color, thickness=2, width=8),
        hovertemplate=f'{label}<br>M = {y:.3f}<br>SEM = {error:.3f}<extra></extra>',
        showlegend=False,
    )


def add_stats_annotation(
    fig: go.Figure,
    text: str,
    x: float = 0.95,
    y: float = 0.95,
    row: int = 1,
    col: int = 1,
    font_size: int = 12
) -> go.Figure:
    """
    Add a stats annotation to a subplot.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure object
    text : str
        Annotation text
    x, y : float
        Position within subplot (0-1)
    row, col : int
        Subplot row and column (1-indexed)
    font_size : int
        Font size
    
    Returns
    -------
    go.Figure : Modified figure
    """
    xref = 'x domain' if col == 1 else f'x{col} domain'
    yref = 'y domain' if col == 1 else f'y{col} domain'
    
    fig.add_annotation(
        x=x, y=y,
        xref=xref, yref=yref,
        text=text,
        showarrow=False,
        font=dict(size=font_size, color='#404040'),
        xanchor='right', yanchor='top',
        bgcolor='rgba(255,255,255,0.8)'
    )
    
    return fig


# =============================================================================
# Color Palette Accessors
# =============================================================================

def get_condition_colors() -> dict:
    """Get condition color palette."""
    return {
        'sham': COLOR_SHAM,
        'active': COLOR_ACTIVE,
        'baseline': COLOR_GOLD,
    }


def get_significance_color(p_value: float) -> str:
    """
    Get color based on significance level.
    
    Parameters
    ----------
    p_value : float
        P-value
    
    Returns
    -------
    str : Color code
    """
    if p_value < 0.001:
        return '#1B5E20'  # Dark green
    elif p_value < 0.01:
        return '#388E3C'  # Green
    elif p_value < 0.05:
        return '#7CB342'  # Light green
    else:
        return '#9E9E9E'  # Gray


# =============================================================================
# Figure Export Helpers
# =============================================================================

def show_with_config(
    fig: go.Figure,
    filename: str = 'figure',
    renderer: Optional[str] = None
) -> None:
    """
    Show figure with standard export configuration.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure object
    filename : str
        Default filename for export
    renderer : str, optional
        Plotly renderer
    """
    config = dict(
        toImageButtonOptions=dict(
            filename=filename,
            format='png',
            scale=2
        )
    )
    
    fig.show(config=config, renderer=renderer)


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing plotting_utils module...")
    
    # Test age_to_rgb
    print(f"  age_to_rgb(25) = {age_to_rgb(25)}")
    print(f"  age_to_rgb(50) = {age_to_rgb(50)}")
    print(f"  age_to_rgb(75) = {age_to_rgb(75)}")
    
    print("Functions: age_to_rgb, get_age_colors, apply_standard_layout")
    print("           add_age_colorbar, create_paired_line_trace, etc.")
