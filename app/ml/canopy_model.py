"""Canopy Health Analysis Model using Strategy Pattern"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List
import logging

from .base import MLModelStrategy
from ..core.models import TreeAnalysis, TreeHealth

logger = logging.getLogger(__name__)


class CanopyHealthModel(MLModelStrategy):
    """
    Analyzes tree canopy health using clustering algorithms.
    Identifies patterns in tree health metrics over time.
    """
    
    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        self.scaler = StandardScaler()
    
    def train(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Train KMeans clustering on tree health data.
        
        Args:
            X: Feature matrix (tree health metrics)
            y: Not used for unsupervised learning
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train KMeans
        self.model = KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10
        )
        self.model.fit(X_scaled)
        self.is_trained = True
        logger.info("Canopy health model trained successfully")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict health cluster for tree data.
        
        Args:
            X: Feature matrix
        
        Returns:
            Cluster labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        KMeans doesn't provide feature importance.
        
        Returns:
            Empty dictionary
        """
        return {}
    
    def analyze_health_trend(
        self,
        analyses: List[TreeAnalysis]
    ) -> Dict[str, Any]:
        """
        Analyze trends in tree health over time.
        
        Args:
            analyses: List of tree analysis results over time
        
        Returns:
            Health trend analysis
        """
        if len(analyses) < 2:
            return {
                "trend": "insufficient_data",
                "health_change": 0,
                "coverage_change": 0,
                "recommendation": "Need more data points for trend analysis"
            }
        
        # Sort by timestamp
        sorted_analyses = sorted(analyses, key=lambda x: x.timestamp)
        
        # Calculate changes
        first = sorted_analyses[0]
        last = sorted_analyses[-1]
        
        health_change = last.tree_health.healthy - first.tree_health.healthy
        coverage_change = last.canopy_coverage_pct - first.canopy_coverage_pct
        
        # Determine trend
        if health_change > 0 and coverage_change > 0:
            trend = "improving"
            recommendation = "Tree health is improving. Continue current practices."
        elif health_change < 0 and coverage_change < 0:
            trend = "declining"
            recommendation = "Tree health is declining. Investigate causes and consider intervention."
        elif health_change == 0 and coverage_change == 0:
            trend = "stable"
            recommendation = "Tree health is stable. Monitor regularly."
        else:
            trend = "mixed"
            recommendation = "Mixed signals in tree health. Investigate specific factors."
        
        return {
            "trend": trend,
            "health_change": health_change,
            "coverage_change": coverage_change,
            "recommendation": recommendation,
            "period_days": (last.timestamp - first.timestamp).days
        }
    
    def calculate_health_score(self, analysis: TreeAnalysis) -> float:
        """
        Calculate overall health score from tree analysis.
        
        Args:
            analysis: Tree analysis data
        
        Returns:
            Health score (0-100)
        """
        total_trees = analysis.tree_health.healthy + analysis.tree_health.needs_care + analysis.tree_health.needs_replacement
        
        if total_trees == 0:
            return 0.0
        
        # Weighted score
        healthy_weight = 1.0
        care_weight = 0.5
        replacement_weight = 0.0
        
        score = (
            (analysis.tree_health.healthy * healthy_weight) +
            (analysis.tree_health.needs_care * care_weight) +
            (analysis.tree_health.needs_replacement * replacement_weight)
        ) / total_trees
        
        # Adjust by confidence and canopy coverage
        adjusted_score = score * analysis.confidence_score * (analysis.canopy_coverage_pct / 100)
        
        return min(100, max(0, adjusted_score * 100))
    
    def compare_with_benchmark(
        self,
        analysis: TreeAnalysis,
        benchmark_density: float
    ) -> Dict[str, Any]:
        """
        Compare tree density with benchmark.
        
        Args:
            analysis: Tree analysis data
            benchmark_density: Benchmark density (trees per acre)
        
        Returns:
            Comparison results
        """
        current_density = analysis.tree_density_per_acre
        difference = current_density - benchmark_density
        percent_difference = (difference / benchmark_density) * 100
        
        if percent_difference > 20:
            status = "above_benchmark"
            recommendation = "Density is above benchmark. Consider thinning if overcrowding is observed."
        elif percent_difference < -20:
            status = "below_benchmark"
            recommendation = "Density is below benchmark. Consider planting more trees for optimal yield."
        else:
            status = "within_benchmark"
            recommendation = "Density is within acceptable range of benchmark."
        
        return {
            "status": status,
            "current_density": current_density,
            "benchmark_density": benchmark_density,
            "difference": difference,
            "percent_difference": percent_difference,
            "recommendation": recommendation
        }
