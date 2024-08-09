# Characterizing Aquifer Diffusivity
> The application calculates the aquifer diffusivity using the simplified amplitude and time lag equations for groundwater wells that are tidally influenced

## Methodology Used
1.	Obtain groundwater and tide/hydrometric water level data 
2.	Choose optimal time frame of with a length of100 hours, based on most minimal groundwater level fluctuations
3.	Fit groundwater level data with a composite sine curve 
4.	Simplified Amplitude Analysis:
     * Calculate well/shore amplitudes
     * Pair each corresponding well/shore amplitudes together
     * Compute Hx/h0 for each corresponding well/shore amplitude pair
     * Compute T/S for each Hx/h0 value
     * Remove outliers
     * Compute mean T/S 
5.	Time Lag Analysis:
     * Identify time at peaks/troughs for both well/shore 
     * Compute time lags for peaks/troughs 
     * Compute T/S for each time lag 
     * Remove outliers
     * Compute mean T/S
