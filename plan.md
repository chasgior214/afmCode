Two main additions:
1) New method for determining volume above substrate, plus integration into automated well finding
	a) Fitting a plane to a flattened image's substrate
        - Needs to deal with edge cases where much of an image is the well
            - Initially, just don't use those images (don't use images with < 5 um in largest dimension)
            - How does Igor flatten in such cases?
		- Should handle the substrate being a different height than the graphene
            - First determine if this will make a significant difference in permeation coefficient. If not, put this down below with later section
            - Figure out which the deflection should measure from even though it makes almost no difference, need to have it stick to one vs the other
                - The substrate I think because the volume below that is included in the other part of the calculation?
                    - I won't have access to the substrate height for many images, though the step between the substrate and the top of the graphene could be calculated across all images of the sample given the pipeline automation in part 2 below like c in Hencky's solution will be, so could use that times the well area and add that to the volume
                        - If thickness times area is small compared to volume, leave out in first implementation
            - Can assume that the well is mostly surrounded by graphene as opposed to mostly surrounded by bare substrate
                - What if there's multiple step heights of graphene?
                    - Then determining the reference height used for the membrane bulge (before adding thickness to substrate times area) is the same as between the substrate and the first step: use whatever step is surrounding most of the well
                        - This means potentially using different planes for different areas of the same image
        - Fit plane again after well circles are found in step c below with all points in the circles not used in plane fitting (helps not use points in wells that are flat to surface)
	b) Mask of deviations from the substrate to identify the wells
		- Start with a simple few-nm deviation from substrate mask in either direction
            - Can start with something simple like 3-nm deviation. Later can move to estimating noise in substrate height to define the cutoff
        - Or should I use phase or amplitude, or a combination of all three?
            - Likely. Will want to identify wells that are basically level to the substrate, including those not in maps that are potentially only partially covered
	c) Fit circles of the proper area to the mask
		- Good check to add: use the spacing between wells to identify the max number of wells that an image could contain, and also what their relative positions should be (can account for the potential of the die not being aligned to the piezo's x and y axes)
            - Can do this for all wells whether they're on the map or not, and even if they're mostly out of the image
        - Note that the area used here isn't likely to be the DRIE-etched area, as the membrane can bulge up above some of the substrate just outside the well
            - And the area can differ between wells and even for a given well over time based on how inflated they are
                - Between wells, differences in tension/etc can mean a different area for two wells at a given pressure
	d) Have it take the area integral of the (signed) height above/below the substrate within the circle
		- How to deal with wells not fully in an image?
			- Don't count them at all for first implementation
            - After, if more than some given portion is in the image, could check if what is in view is symmetrical to a given degree, and if so use a mirror of the visible portion as a substitute for what's out of frame
                - Or with the full pipeline described in part 2 below, could be possible to cross-reference what the volume for that membrane is when the parts of the height that can be seen match the heights of that membrane in another image
        - When it's all working, try to see how estimated permeation coefficient changes at different height levels (or even when the membrane is just above or a bit below substrate level)
	e) Find the equivalent peak height for a symmetrical bulge with that volume
		- Start with Hencky's solution, and start with an assumption of c = 1.5e-4 kPa/nm^3
			- Perhaps just save volumes and as part of 2b below build the system that calculates the equivalent height/needed things for permeation
			- Try paraboloid later for comparison for curiosity
		- Ask Boutilier if using change in volume directly is beneficial instead of pi * R^2 * a * h (and maybe same for dV/dt)? Whether or not I do, compare what that gives to the volume above the substrate I get
	- Compare to results of current method when done. Especially for clean images of symmetrical wells, should get near-identical results
    - For each image it analyses, validate that it's base layer wasn't flattened and that the layer it uses is histo flattened. Else stop and raise to user
		- Added function for these checks to AFM image class
		    - Check they work on the ones I did ultra restore for and then histo flattened
    - How to deal with debris on red? If it makes < 2% difference, maybe don’t worry about it? Otherwise, can calculate volume of it and subtract that

2) Permeation calculations and pipelining:
    a) Finish membrane_calcs.py
        - Replicate against perm coeff calcs Boutilier sent
	b) Rework of pipeline from heights to slopes to permeation
		- It’d be nice to have it pick points for slopes in a deterministic way because then I could automate that part of data analysis
        - Does not need to be backwards-compatible with how deflation heights/slopes have been stored in the past
            - Can pick different definition of what I include in a well map (like always include wells that are at least partly covered) if it helps, or even different definition of a well map (more metadata, etc)
		- Try to take into account cases where a well is evacuated before being charged and there's air coming into it while the gas of interest is flowing out
            - We’ve used the first measurement for slow-deflating gasses so we use the height at the initial pressure of the gas of interest, though by the time it’s falling that’s changed a little. Could do some analysis on how much that changes things, but doesn't make a big difference at the end of the day
                - Given N2 and O2 permeation coefficients, could figure out how much of the total internal pressure (known by the height and Hencky’s solution) is air!
                    - Could expand to other mixtures of gasses, for example sample37's controls with SF6 and air and other gasses I'll be charging it with mainly to image red/blue
	c) Automatic identification of deflations, which gas goes with which deflation, pressures of gas leading to inflation
		- Needs to be backwards-compatible with both csv pressures and pressures I logged in Excel (though that can mean just copying those pressures to a new format)
		- Have it automatically go through all of each membrane's deflations of the slowest gasses and use them to fit the c term for hencky's solution to use in the permeation calculations
        - Look to add equilibirum height at t=0 for fast gasses as a point that can be used
	d) Also want improved plotting of permeation coefficients
		- Plot coefficinets from beyond my experiments (see Boutilier's email for file with values for both the below)
            - Boutilier's/Erfan's samples (maybe with automatic reanalysis of their images)
            - From other labs → all ones in hydrocarbon paper, maybe more

- Later
    - For extremely fast gasses, the height changes materially as I scan once through the membrane. Take into account, could actually be great:
        - Given I've imaged the membrane before for a slow gas, I can get many "points" on the h(t) curve from one scan. For a given line:
            1. Calculate the area above the substrate (integrate height across the line)
            2. Compare to a slow deflation: for the same position on the membrane, look up the membrane volume when the area above the line equals the area above the line for the fast deflation
            3. That volume is the volume of the membrane at the point in time the line was taken for the fast membrane
            4. Repeat with other lines later in the scan further down the membrane
        - If image quality is lacking, could average area of a few lines for the height at the time the central one was taken
        - If it's not symmetric but I know the height of a given area of it at a given pressure, same concept applies
        - Same could happen for when they pass below the substrate but aren't fully bottomed
        - If the image quality is good enough, this could be beneficial even for only a relatively quick gas like H2
        - To automate fully, will want it to be able to decide whether to do it this way or get one volume per image
            - HOW
    - A detailed uncertainty analysis could be great for the thesis. Ideas for ways to probe uncertainty:
        - Use measured V and dV/dt in q = dn/dt / P = 1/(R_u * T) * (V / P * dP/dt + dV/dt), see difference it makes
        - 