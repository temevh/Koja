# Strategy: Vesa 2 Strategy

**Owner**: Vesa
**Approach**: Adaptive Rule-Based Control with Swept Optimal Parameters & Night Setback

This strategy leverages the parameters found via a multi-dimensional sweep for optimal balance between energy costs and CO2/temperature penalties. 
In addition to CO2 demand-controlled flow and outdoor air compensation, this strategy adds a night setback logic that gently lowers heating setpoints during unoccupied hours.

## How to run
```bash
cd strategies/_template/
python run_idf.py
```

## Key idea
(Describe what makes your strategy different)

## Results
(Paste your total cost after running the simulation)
