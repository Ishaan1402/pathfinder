import os
from typing import List

def print_sparkline(study_name: str, scores: List[float], best_score: float, trial_number: int):
    """Prints a Unicode sparkline of study performance to stdout if HPO_SPARKLINES=1 is set."""
    if os.getenv("HPO_SPARKLINES") != "1":
        return
        
    if not scores:
        return
        
    ticks = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    
    min_s = min(scores)
    max_s = max(scores)
    range_s = max_s - min_s
    
    spark_str = ""
    for s in scores:
        if range_s == 0:
            idx = 0
        else:
            idx = int(((s - min_s) / range_s) * (len(ticks) - 1))
        # Ensure idx is within bounds
        idx = max(0, min(len(ticks) - 1, idx))
        spark_str += ticks[idx]
        
    print(f"\nStudy: {study_name} [{spark_str}] best={best_score:.4f} (Trial #{trial_number})\n")
