"""
Report Generator - generates HTML/PDF evaluation reports with visualizations.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates evaluation reports with visualizations."""
    
    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_html_report(
        self,
        evaluation_report: Dict[str, Any],
        include_charts: bool = True
    ) -> str:
        """
        Generate HTML evaluation report.
        
        Args:
            evaluation_report: Evaluation report data
            include_charts: Whether to include charts
            
        Returns:
            Path to generated HTML file
        """
        report_id = evaluation_report.get("report_id", "unknown")
        filename = f"evaluation_report_{report_id}.html"
        filepath = self.output_dir / filename
        
        html_content = self._build_html_content(evaluation_report, include_charts)
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {filepath}")
        
        return str(filepath)
    
    def _build_html_content(
        self,
        report: Dict[str, Any],
        include_charts: bool
    ) -> str:
        """
        Build HTML content for report.
        
        Args:
            report: Report data
            include_charts: Whether to include charts
            
        Returns:
            HTML string
        """
        # Extract data
        report_id = report.get("report_id", "N/A")
        created_at = report.get("created_at", datetime.utcnow())
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        
        total_evals = report.get("total_evaluations", 0)
        aggregate_metrics = report.get("aggregate_metrics", {})
        recommendations = report.get("recommendations", [])
        performance_by_category = report.get("performance_by_category", {})
        
        # Build HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report - {report_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .header p {{
            margin: 5px 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: #667eea;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
        .metric-details {{
            margin-top: 10px;
            font-size: 12px;
            color: #666;
        }}
        .recommendations {{
            list-style: none;
            padding: 0;
        }}
        .recommendations li {{
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 6px;
            background: #f8f9fa;
        }}
        .recommendations li:before {{
            margin-right: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .chart-container {{
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Model Evaluation Report</h1>
        <p><strong>Report ID:</strong> {report_id}</p>
        <p><strong>Generated:</strong> {created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        <p><strong>Total Evaluations:</strong> {total_evals}</p>
    </div>
    
    <div class="section">
        <h2>📊 Aggregate Metrics</h2>
        <div class="metric-grid">
"""
        
        # Add metric cards
        for metric_name, metric_data in aggregate_metrics.items():
            if isinstance(metric_data, dict):
                mean_val = metric_data.get('mean', 0)
                html += f"""
            <div class="metric-card">
                <h3>{metric_name.replace('_', ' ').title()}</h3>
                <div class="metric-value">{mean_val:.3f}</div>
                <div class="metric-details">
                    Min: {metric_data.get('min', 0):.3f} | 
                    Max: {metric_data.get('max', 0):.3f} | 
                    Std: {metric_data.get('std', 0):.3f}
                </div>
            </div>
"""
        
        html += """
        </div>
    </div>
    
    <div class="section">
        <h2>📈 Performance by Category</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Count</th>
                    <th>Avg Rating</th>
                    <th>Avg Relevance</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Add category rows
        for category, data in performance_by_category.items():
            html += f"""
                <tr>
                    <td>{category.replace('_', ' ').title()}</td>
                    <td>{data.get('count', 0)}</td>
                    <td>{data.get('avg_rating', 0):.2f}</td>
                    <td>{data.get('avg_relevance', 0):.3f}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>💡 Recommendations</h2>
        <ul class="recommendations">
"""
        
        # Add recommendations
        for rec in recommendations:
            html += f"            <li>{rec}</li>\n"
        
        html += """
        </ul>
    </div>
    
</body>
</html>
"""
        
        return html
    
    def generate_json_report(
        self,
        evaluation_report: Dict[str, Any]
    ) -> str:
        """
        Generate JSON evaluation report.
        
        Args:
            evaluation_report: Evaluation report data
            
        Returns:
            Path to generated JSON file
        """
        report_id = evaluation_report.get("report_id", "unknown")
        filename = f"evaluation_report_{report_id}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        logger.info(f"Generated JSON report: {filepath}")
        
        return str(filepath)
    
    def generate_summary_text(
        self,
        evaluation_report: Dict[str, Any]
    ) -> str:
        """
        Generate a text summary of the evaluation.
        
        Args:
            evaluation_report: Evaluation report data
            
        Returns:
            Summary text
        """
        report_id = evaluation_report.get("report_id", "N/A")
        total_evals = evaluation_report.get("total_evaluations", 0)
        aggregate_metrics = evaluation_report.get("aggregate_metrics", {})
        recommendations = evaluation_report.get("recommendations", [])
        
        summary = f"""
EVALUATION REPORT SUMMARY
========================
Report ID: {report_id}
Total Evaluations: {total_evals}

KEY METRICS:
"""
        
        for metric_name, metric_data in aggregate_metrics.items():
            if isinstance(metric_data, dict):
                mean_val = metric_data.get('mean', 0)
                summary += f"  - {metric_name.replace('_', ' ').title()}: {mean_val:.3f}\n"
        
        summary += "\nRECOMMENDATIONS:\n"
        for i, rec in enumerate(recommendations, 1):
            summary += f"  {i}. {rec}\n"
        
        return summary
    
    def create_comparison_chart(
        self,
        comparison_data: Dict[str, Any],
        output_filename: str = "comparison_chart.png"
    ) -> str:
        """
        Create a comparison chart between two evaluations.
        
        Args:
            comparison_data: Comparison data
            output_filename: Output filename
            
        Returns:
            Path to generated chart
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            metric_changes = comparison_data.get("metric_changes", {})
            
            if not metric_changes:
                logger.warning("No metric changes to plot")
                return ""
            
            metrics = list(metric_changes.keys())
            before_values = [metric_changes[m]["before"] for m in metrics]
            after_values = [metric_changes[m]["after"] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x - width/2, before_values, width, label='Before', color='#764ba2')
            ax.bar(x + width/2, after_values, width, label='After', color='#667eea')
            
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Values')
            ax.set_title('Evaluation Comparison: Before vs After')
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            filepath = self.output_dir / output_filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Generated comparison chart: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating comparison chart: {e}")
            return ""


# Global instance
report_generator = ReportGenerator()
