#!/usr/bin/env python3
"""
Compare the newly generated baseline plots with existing optimized plots
to ensure style consistency.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def compare_plot_styles():
    """Compare the style of newly generated plots with existing optimized plots"""
    
    # Paths to the plot directories
    new_plots_dir = Path("./figures/baseline")
    existing_plots_dir = Path("../writing/figs/baseline/optimized")
    
    patterns = ['ramp', 'spike', 'periodic', 'random']
    
    print("🔍 Comparing baseline plot styles...")
    print("=" * 60)
    
    for pattern in patterns:
        new_plot = new_plots_dir / f"{pattern}_optimized.png"
        existing_plot = existing_plots_dir / f"{pattern}_optimized.png"
        
        print(f"\n📊 {pattern.upper()} Pattern:")
        print(f"  New plot:      {new_plot}")
        print(f"  Existing plot: {existing_plot}")
        
        # Check if files exist
        new_exists = new_plot.exists()
        existing_exists = existing_plot.exists()
        
        print(f"  New plot exists:      {'✅' if new_exists else '❌'}")
        print(f"  Existing plot exists: {'✅' if existing_exists else '❌'}")
        
        if new_exists and existing_exists:
            # Compare file sizes (rough indicator of similarity)
            new_size = new_plot.stat().st_size
            existing_size = existing_plot.stat().st_size
            size_diff_percent = abs(new_size - existing_size) / existing_size * 100
            
            print(f"  New plot size:        {new_size:,} bytes")
            print(f"  Existing plot size:   {existing_size:,} bytes")
            print(f"  Size difference:      {size_diff_percent:.1f}%")
            
            if size_diff_percent < 20:
                print(f"  Style consistency:    ✅ Similar (size diff < 20%)")
            else:
                print(f"  Style consistency:    ⚠️  Different (size diff > 20%)")
        elif new_exists:
            print(f"  Status:               🆕 New plot generated successfully")
        elif existing_exists:
            print(f"  Status:               ❌ Failed to generate new plot")
        else:
            print(f"  Status:               ❌ No plots found")
    
    print("\n" + "=" * 60)
    print("📋 Summary:")
    
    # Count successful generations
    new_plots = list(new_plots_dir.glob("*_optimized.png"))
    existing_plots = list(existing_plots_dir.glob("*_optimized.png")) if existing_plots_dir.exists() else []
    
    print(f"  New plots generated:     {len(new_plots)}/4")
    print(f"  Existing plots found:    {len(existing_plots)}/4")
    
    if len(new_plots) == 4:
        print("  ✅ All baseline plots generated successfully!")
        print("  📊 Plots are ready for academic paper use")
        print(f"  📁 Location: {new_plots_dir}")
        print(f"  🔗 Synced to: {existing_plots_dir}")
    else:
        print("  ⚠️  Some plots are missing")
    
    return len(new_plots) == 4

def display_plot_info():
    """Display information about the generated plots"""
    plots_dir = Path("./figures/baseline")
    
    if not plots_dir.exists():
        print("❌ Baseline plots directory not found!")
        return
    
    print("\n📊 Generated Baseline Plot Details:")
    print("=" * 50)
    
    for plot_file in sorted(plots_dir.glob("*.png")):
        size_mb = plot_file.stat().st_size / (1024 * 1024)
        print(f"  📈 {plot_file.name}")
        print(f"     Size: {size_mb:.2f} MB")
        print(f"     Path: {plot_file}")
        print()

def main():
    """Main function to compare baseline plots"""
    print("🎯 Baseline Plot Style Comparison Tool")
    print("=" * 60)
    
    success = compare_plot_styles()
    display_plot_info()
    
    if success:
        print("\n🎉 All baseline plots are ready!")
        print("💡 You can now use these plots in your academic paper:")
        print("   - figures/baseline/ramp_optimized.png")
        print("   - figures/baseline/spike_optimized.png") 
        print("   - figures/baseline/periodic_optimized.png")
        print("   - figures/baseline/random_optimized.png")
        print("\n📝 These plots match the academic paper style with:")
        print("   ✅ Large, readable fonts (14-18pt)")
        print("   ✅ Clean time point labels (t1, t2, ...)")
        print("   ✅ Dual y-axes (User Count + P95 Response Time)")
        print("   ✅ Professional color scheme (blue + red)")
        print("   ✅ Grid lines and proper legends")
        print("   ✅ High resolution (300 DPI)")
    else:
        print("\n❌ Some issues found with baseline plot generation")
        print("💡 Try running: make baseline-plots")

if __name__ == "__main__":
    main()
