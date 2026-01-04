general code cleanup/some refactoring, documentation
- YAML config instead of settings all over the place (or at least keep centralizing in path_loader)
- make it simpler to find all the depressurizations, sorted/filtered by sample, gas, time, wells captured, etc
    - implement it into plot_recent_deflation_curve_slopes instead of the current workaround

# New Pressure Logger
- Make PressureSeries hold more metadata (what time period it was in which cell, gas, etc)
- Make functions to plot the pressures in a PressureSeries over time
- Test what happens if the Arduino is unplugged while logging
- predictions for what pressure will level off to when cell left attached to gas tank

# Well Locator Improvement Plan
## Next Steps
- Pick when to ask for user input. Don't want it to fail silently on skipping wells it should get points for
- Check that the ellipse described by the paraboloid fit is roughly circular
- Try using c term in paraboloid fit to reject fits with low rotational symmetry
- Play with fit window size
- Check accuracy by comparing results of both an H2 and a long one to what I got previously
- Add test for finding algorithm across a large series of images with known well locations, should find them all with tight tolerance
- look again at if fit window not including the vertex can be used to reject fits. Using it caused problems before

## Inadmissible solutions
- If there is no point in the image data within 5 nm of the vertex, reject it (root sum of squares → need to convert x/y to nm from um)
- Can check what the fit paraboloid would indicate about the well’s diameter (the circle at the intersection of the paraboloid and the surface). It’s not a perfect paraboloid, but saying it should be at least a 1 or 2 um diameter and less than 10 um diameter could be an amazing way to assess if fit is feasible
- Assign a min R^2 value (0.2?) for a fit → do this last, would want to play around after all the other stuff and see how bumpy of a near-cratered well I can get away with low R^2 for
- Well finding could look at if the disruption in the substrate plane is about cirular, and if not and no well expected there, it's debris - could also use in finding algorithm

## Better Finding Algorithm
- When multiple wells present in the same image, check that they are, within a tight margin (0.5 um to start), found to be within where they’d each predict the others to be
    - If not, use the one with the best R^2 to find the others
    - Consider that I can check what the drift is assuming each fit was correct, and compare those drifts. Outliers can indicate if one fit was bad. RANSAC (or just pick the median) to get a good drift vector, and use that from the old image to get the real position of the well that had a bad fit/go off of the ones with middle of the pack drift and map the outlier(s) to predicted positions based on where the other good ones are
- For images with one well, potentially skip over them and then get drift between image before and after it, divide that by two (or maybe time-weighted average of those two), and use that to get where the well in that image should be
- Play with paraboloid fit mask cutoff radius?
- If paraboloid fit deflection > 0 and there’s a position in the image data within 4um that gives > 10 nm taller compared to substrate at that position, move to that position and iterative paraboloid fit again
    - Likewise for below surface and negative
- Instead of updating well_positions individually, treat the well_map as a rigid constellation. Fit the entire constellation to the found points in the new image using a Least Squares Rigid Transformation (finding the best translation (dx,dy) that minimizes error for all points simultaneously)
- Could use the vertex of the paraboloid, the image, and the diameter of the well to:
    1. Find what part of the image is the substrate (within 5-10 nm of the height given by calculate_substrate_height), or by subtracting flattened height trace from flattened height retrace and looking for everywhere within a few nm of the mode of that difference map
    2. See if the vertex is centered in x and y over a non-substrate patch in the image
- could I do the fourier transform thing that the drift correction algorithm on the AFM uses to align all my images? Could feed it all the images from a depressurization and it could match the wells together even if the head were moved around
- Could train a model to notice strange selections based on combinations of things like deflection and paraboloid-substrate intersection area if it's hard to do with code
- Eventually, could add accounting for slight tilt of sample relative to x/y piezo in the well finder. Far off

## Better Feedback to User
Don’t hesitate to raise failures. I’ll learn from why they happened and either be able to automate it or it’ll just surface the strangest cases to me
- If it wanders more than (2um?) from its starting estimated point, have it have the user pick where it is → maybe showing other recently anchored points and the estimated position (and paraboloid fit path) on a stitched map to help me see where it should be
- Better visualizations to see why things go wrong: height map with predicted location, fit point, max and min within 3 um of each of those, maybe the path of iterative paraboloid fitting all marked on it, and any of the above that are relevant
- Overall, need to pick when it asks for user input

## Other Improvements
- Let it figure out which wells are which for itself. If it can find 5 wells in the first 10 images of sample37, it can figure out which is which based on relative positions and assign them to well IDs itself

# Manual Point Selection Improvements
- In paraboloid fit panel, show area of intersection with substrate
- In paraboloid fit panel, button to display the paraboloid on the image with make_heightmap_3d_surface
- Use mouse wheel for something
    - Zoom in/out? Centred on x,y of most recently selected point or centre of FOV or cursor? Maybe both, one active normally, another with shift + scroll, another with ctrl + scroll?
    - use horizontal scroll for something?
- Use left/right arrow keys for something
    - next/previous image in the collection or something else
- Display any other metadata I might want (drive amplitude/frequency, etc)
    - Maybe in another panel
    - Note that if I change some settings partway through an image, it tracks that in the metadata. Example, initial drive amplitude line looks like "DriveAmplitude: 0.089596", but later on if it was changed it would have a line that looks like "DriveAmplitude: 0.09@Line: 162"
    - Initial drive amplitude in for now, but make sure to have it take the one for the line later
    - Maybe imaging parameters on the right of the panel, separate from other data
- Plot of deflections so far visible? In other window on second screen?
- Clean up buttons?
- add a button "View 3D Heightmap", which renders a 3d heightmap in a new window. Can select points on the 3d heightmap which get added as selected points in the main window.
- cross section shows dotted lines of the line above and below the selected line?

# Substrate Detection Improvements
- Switch to using the mean of the heights in the mode bin instead of picking a data point closest to the mode bin center to reduce noise
- When looking for the "substrate" (what I really mean is the graphene height outside the well), if it's bimodal/multimodal, can have it pick the one that's closest to the paraboloid vertex. Fixes edge cases where more of the image line off the well is either on the substrate or on a different step height of graphene

# Excel Integration
- For chronological plot, let me just give a start date (last week’s ppt) and it automatically takes the ones since then and plots them (would need to do basic Excel access for getting the gas species but mostly could read from the slopeIDs)


# To Organize
- incorporate well mapping to stitching to account for drift

- ability to show movie of deflation over time (with both interpolated and just raw data as options)

Try using a denoising model to smooth images before paraboloid fitting?
    - https://careamics.github.io/0.1/
    - likely use either an algorithm of single images or pairs of images
    - give it cleaner photos only (not when the drive amplitude is too low, maybe only ones with good phase maps)
    - test on some particularly noisy deflations (ex H2 deflation images from Oct 17th), see how much the fits change

could try to use points from multiple depressurizations to get a better slope estimate

Have it make a map, then I select the regions that a well is in over the whole imaging session given drift, and it automatically shows me the same well over and over instead of navigating through images to find them (and could also have it automatically output a curve of using the highest point for that well, which I could compare to mine, and maybe if I get a denoising model to work well enough it could do basically everything automatically. Could also have it make timelapses of a 3d image of a single well changing over time given it would know how to center it)

Now that I'm saving pixel coordinates, try to have it read those back in and use them to get the deflection from the z-sensor data to compare to the height data. Can also make comparisons to curves made from max height within a few microns of the selected point (or the min of the 8 pixels surrounding it) vs the mode for the y value cross section that the max sits on