# stochastic_subgrid_parameterization

In geophysical and engineering flows it is not possible to resolve all of the scales of motion, so one must resolve the large eddies are explicitly on a computational grid, and parameterise the interactions with the unresolved subgrid-scales. If these subgrid interactions are not properly parameterised, then an increase in resolution will not necessarily increase the accuracy of the resolved scales, hence the dependence on resolution. All simulation codes to date including the most sophisticated general circulation models suffer from this problem. This has wide ranging implications for geophysical research and operational activities, including weather / decadal / climate prediction. 

The code provided in this repo addresses the problem of resolution dependence problem via the development of data-driven stochastic subgrid turbulence parameterisations. Lower resolution simulations adopting these parameterization coefficients reproduce the statistical properties of the higher resolution version at each of the resolved scaled. This approach has been successfully applied to global atmospheres (Kitsios et. al. 2012, Kitsios & Frederiksen, 2019), global oceans (Ktisos et. al. 2013), and fully turbulent three-dimensional boundary layers (Kitsios et. al. 2017).

The process is completely general. It uses data-driven means to parameterize the all of the fundamental subgrid interactions between the large scale features resolved on a computational grid, and the small scale unresolved features in quadratically nonlinear systems. For flows with eddies, meanfields and topography (or bathymetry / orography / roughness) the fundamental classes of subgrid interactions are as follows:

* Eddy-eddy interactions are those between the subgrid and resolved eddies.

* Eddy-meanfield interactions are those between the subgrid and resolved components of the eddies and meanfield.

* Eddy-topographic  interactions are those between the subgrid and resolved components of the eddies and topography.

* The meanfield-topographic and meanfield-meanfield interactions together are referred to as the meanfield Jacobian terms, representing interactions between the resolved and subgrid components of the meanfield and topography.

The code provided calculates parameterizations for all of these interactions.

The analysis here is applied to the quasi-geostrophic simulation of an idealized global oceanic flow. Due to the size of the complete dataset, files have been included in the directory ./T504/ to calculate only the eddy-eddy coefficients. In order to calculate coefficients for the remaining interaction classes, additional files will need to be downloaded from an alternate source. However, these calculations have been undertaken beforehand with the result files provided in this repo. Using these result files one can reproduce all of the diagrams in the notebook. The diagrams are output to the files figure*.pdf in this repo.



__References:__

Kitsios, V. & Frederiksen, J.S., 2019, Subgrid parameterizations of eddy-eddy, eddy-meanfield, eddy-topographic, meanfield-meanfield and meanfield-topographic interactions in atmospheric models, Journal of the Atmospheric Sciences, Vol. 76, pp 457-477. https://doi.org/10.1175/JAS-D-18-0255.1

Kitsios, V., Frederiksen, J.S. & Zidikheri, M.J., 2012, Subgrid model with scaling laws for atmospheric simulations, Journal of the Atmospheric Sciences, 69, pp 1427-1445. https://doi.org/10.1175/JAS-D-11-0163.1

Kitsios, V., Frederiksen, J.S. & Zidikheri, M.J., 2013, Scaling laws for parameterisations of subgrid eddy-eddy interactions in simulations of oceanic circulations, Ocean Modelling, 68, pp 88-105. https://doi.org/10.1016/j.ocemod.2013.05.001

Kitsios, V., Sillero, J.A., Frederiksen, J.S. & Soria, J., 2017, Scale and Reynolds number dependence of stochastic subgrid energy transfer in turbulent channel flow, Computers and Fluids, Vol. 151, pp 132-143. https://doi.org/10.1016/j.compfluid.2016.08.003
