# Characterizing Aquifer Diffusivity
> The application calculates the aquifer diffusivity using the simplified amplitude and time lag equations given by (Ferris et al., 1962; Todd, 1980) for groundwater wells that are tidally influenced
>
> Visit streamlit site to use application [Streamlit Application](https://aquifer-diffusivity-calculator.streamlit.app/)

## Methodology Used
> ![image](https://github.com/user-attachments/assets/9d361204-7779-45e7-9a83-234390fa37ae)
1.	Obtain groundwater and tide/hydrometric water level data 
2.	Choose optimal time frame of with a length of h hours, based on most minimal groundwater level fluctuations
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



## References
Ferris, J.G., D.B. Knowles, R.H. Brown and R.W. Stallman, 1962. Theory of aquifer tests, U.S. Geol. Surv. 1536, 174

Fisheries and Oceans Canada. (2024). Tides, Currents and Water Levels. https://www.tides.gc.ca/en/stations

Province of British Columbia. (2024). Aquarius. https://aqrt.nrs.gov.bc.ca/Data 

Todd, D.K., 1980. Groundwater Hydrology, 2nd ed., John Wiley & Sons, 535
