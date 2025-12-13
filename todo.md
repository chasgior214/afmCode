organize this list

general code cleanup/some refactoring, documentation
- YAML config instead of settings all over the place (or at least keep centralizing in path_loader)

- incorporate well mapping to stitching to account for drift

keep working on Excel integration
- For chronological plot, let me just give a start date (last week’s ppt) and it automatically takes the ones since then and plots them (would need to do basic Excel access for getting the gas species but mostly could read from the slopeIDs)

when looking for the "substrate" (what I really mean is the graphene height outside the well), if it's bimodal/multimodal, can have it pick the one that's closest to the paraboloid vertex. Fixes edge cases where more of the image line off the well is either on the substrate or on a different step height of graphene

make right click zoom and left click selections? Then can do away with button to enter/exit zoom mode and auto entering zoom mode on startup. Right drag could be box zoom and right click to zoom to 4x4 um square centred on cursor

cross section shows dotted lines of the line above and below the selected line?

another line, extremum location relative to neutral piezo in um (take position relative to where the middle of the scan would be, and add offset)

could I do the fourier transform thing that the drift thing uses to align all my images? Could feed it all the images from a depressurization and it could match the wells together, then I could pinpoint centeres and do completely automated data extraction, could be amazing with sample53

Try using a denoising model to smooth images before paraboloid fitting?
    - https://careamics.github.io/0.1/
    - likely use either an algorithm of single images or pairs of images
    - give it cleaner photos only (not when the drive amplitude is too low, maybe only ones with good phase maps)
    - test on some particularly noisy deflations (ex H2 deflation images from Oct 17th), see how much the fits change

Add filters to the list of images, could show only ones within a certain range of offsets to pick specific wells

could try to use points from multiple depressurizations to get a better slope estimate

Have it make a map, then I select the regions that a well is in over the whole imaging session given drift, and it automatically shows me the same well over and over instead of navigating through images to find them (and could also have it automatically output a curve of using the highest point for that well, which I could compare to mine, and maybe if I get a denoising model to work well enough it could do basically everything automatically. Could also have it make timelapses of a 3d image of a single well changing over time given it would know how to center it)

Now that I'm saving pixel coordinates, try to have it read those back in and use them to get the deflection from the z-sensor data to compare to the height data. Can also make comparisons to curves made from max height within a few microns of the selected point (or the min of the 8 pixels surrounding it) vs the mode for the y value cross section that the max sits on

Clean up buttons?

- Plot of deflections so far visible? In other window on second screen?

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

- add a button "View 3D Heightmap", which renders a 3d heightmap in a new window. Can select points on the 3d heightmap which get added as selected points in the main window.