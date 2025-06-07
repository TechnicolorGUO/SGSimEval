"""
Visualization module for SGSimEval.
Handles plotting of evaluation results and similarity metrics.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
import os

@dataclass
class PlotConfig:
    """Configuration for plotting settings."""
    save_path: Optional[str] = None
    style: str = 'seaborn'
    dpi: int = 300
    figsize: Tuple[int, int] = (10, 6)

def radar_factory(num_vars: int, frame: str = 'circle') -> Tuple[plt.Figure, plt.Axes]:
    """Create a radar chart factory."""
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(plt.PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            return path

    class RadarAxes(plt.PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(theta, *args, closed=closed, **kwargs)

        def plot(self, *args, **kwargs):
            return super().plot(theta, *args, **kwargs)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta[:-1]), labels)

        def _gen_axes_patch(self):
            return plt.Circle((0.5, 0.5), 0.5)

        def _gen_axes_spines(self):
            return plt.Rectangle((0, 0), 1, 1, fill=False)

    plt.register_projection(RadarAxes)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='radar')
    return fig, ax

class RadarPlotter:
    """Creates radar plots for evaluation metrics."""
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        plt.style.use(self.config.style)

    def plot_radar(
        self,
        metrics: List[str],
        values: List[float],
        title: str
    ) -> None:
        """Create a radar plot for metrics."""
        fig, ax = radar_factory(len(metrics))
        
        # Plot data
        ax.plot(metrics, values)
        ax.fill(metrics, values, alpha=0.25)
        
        # Set labels and title
        ax.set_varlabels(metrics)
        plt.title(title)
        
        # Save if path provided
        if self.config.save_path:
            plt.savefig(
                os.path.join(self.config.save_path, f"{title.lower().replace(' ', '_')}.png"),
                dpi=self.config.dpi,
                bbox_inches='tight'
            )
        plt.close()

    def plot_comparison_radar(
        self,
        metrics: List[str],
        system_values: List[float],
        human_values: List[float],
        title: str
    ) -> None:
        """Create a comparison radar plot."""
        fig, ax = radar_factory(len(metrics))
        
        # Plot data
        ax.plot(metrics, system_values, label='System')
        ax.fill(metrics, system_values, alpha=0.25)
        ax.plot(metrics, human_values, label='Human')
        ax.fill(metrics, human_values, alpha=0.25)
        
        # Set labels and title
        ax.set_varlabels(metrics)
        plt.title(title)
        plt.legend()
        
        # Save if path provided
        if self.config.save_path:
            plt.savefig(
                os.path.join(self.config.save_path, f"{title.lower().replace(' ', '_')}.png"),
                dpi=self.config.dpi,
                bbox_inches='tight'
            )
        plt.close()

class HeatmapPlotter:
    """Creates heatmaps for evaluation results."""
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        plt.style.use(self.config.style)

    def plot_heatmap(
        self,
        data: pd.DataFrame,
        title: str
    ) -> None:
        """Create a heatmap from evaluation data."""
        plt.figure(figsize=self.config.figsize)
        
        # Create heatmap
        sns.heatmap(
            data,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            center=data.mean().mean()
        )
        
        # Set labels and title
        plt.title(title)
        
        # Save if path provided
        if self.config.save_path:
            plt.savefig(
                os.path.join(self.config.save_path, f"{title.lower().replace(' ', '_')}.png"),
                dpi=self.config.dpi,
                bbox_inches='tight'
            )
        plt.close()

    def plot_density_heatmap(
        self,
        data: pd.DataFrame,
        title: str
    ) -> None:
        """Create a density heatmap."""
        plt.figure(figsize=self.config.figsize)
        
        # Create density heatmap
        sns.heatmap(
            data,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            center=data.mean().mean(),
            square=True
        )
        
        # Set labels and title
        plt.title(title)
        
        # Save if path provided
        if self.config.save_path:
            plt.savefig(
                os.path.join(self.config.save_path, f"{title.lower().replace(' ', '_')}_density.png"),
                dpi=self.config.dpi,
                bbox_inches='tight'
            )
        plt.close()

class ResultPlotter:
    """Main class for plotting evaluation results."""
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self.radar_plotter = RadarPlotter(config)
        self.heatmap_plotter = HeatmapPlotter(config)

    def plot_evaluation_results(
        self,
        results: Dict,
        system_name: str
    ) -> None:
        """Plot comprehensive evaluation results."""
        # Plot outline metrics
        if 'outline' in results:
            self.radar_plotter.plot_radar(
                metrics=['Structure', 'Coverage', 'Organization'],
                values=[
                    results['outline'].get('structure_score', 0),
                    results['outline'].get('coverage_score', 0),
                    results['outline'].get('organization_score', 0)
                ],
                title=f'{system_name} - Outline Evaluation'
            )

        # Plot content metrics
        if 'content' in results:
            self.radar_plotter.plot_radar(
                metrics=['Coverage', 'Structure', 'Relevance', 'Language', 'Criticalness'],
                values=[
                    results['content'].get('coverage_score', 0),
                    results['content'].get('structure_score', 0),
                    results['content'].get('relevance_score', 0),
                    results['content'].get('language_score', 0),
                    results['content'].get('criticalness_score', 0)
                ],
                title=f'{system_name} - Content Evaluation'
            )

        # Plot reference metrics
        if 'reference' in results:
            self.radar_plotter.plot_radar(
                metrics=['Quality', 'Supportiveness', 'Relevance'],
                values=[
                    results['reference'].get('quality_score', 0),
                    results['reference'].get('supportiveness_score', 0),
                    results['reference'].get('relevance_score', 0)
                ],
                title=f'{system_name} - Reference Evaluation'
            )

        # Plot similarity metrics if available
        if 'similarity' in results:
            self.radar_plotter.plot_radar(
                metrics=['Outline', 'Content', 'Reference', 'Overall'],
                values=[
                    results['similarity'].get('outline_similarity', 0),
                    results['similarity'].get('content_similarity', 0),
                    results['similarity'].get('reference_similarity', 0),
                    results['similarity'].get('overall_similarity', 0)
                ],
                title=f'{system_name} - Similarity Metrics'
            )

def plot_results(
    results: Dict,
    system_name: str,
    config: Optional[PlotConfig] = None
) -> None:
    """Convenience function to plot evaluation results."""
    plotter = ResultPlotter(config)
    plotter.plot_evaluation_results(results, system_name) 