# Summary

# Inadmissible solutions
- Set limits to what a deflection could possibly be (never more than +/- 350 nm)
- Check if the fitted vertex is within the x/y bounds of the image. If not, reject
- If there is no point in the data within 5 nm of the vertex, reject it (root sum of squares → need to convert x/y to nm from um)
- Can check what the fit paraboloid would indicate about the well’s diameter (the circle at the intersection of the paraboloid and the surface). It’s not a perfect paraboloid, but saying it should be at least a 1 or 2 um diameter and less than 10 um diameter could be an amazing way to assess if fit is feasible
- Assign a min R^2 value (0.2?) for a fit → do this last, would want to play around after all the other stuff and see how bumpy of a near-cratered well I can get away with low R^2 for
- Well finding could look at if the disruption in the substrate plane is about cirular, and if not and no well expected there, it's debris - could also use in finding algorithm

# Better Finding Algorithm
- When multiple wells present in the same image, check that they are, within a tight margin (0.5 um to start), found to be within where they’d each predict the others to be
    - If not, use the one with the best R^2 to find the others
    - Consider that I can check what the drift is assuming each fit was correct, and compare those drifts. Outliers can indicate if one fit was bad. RANSAC (or just pick the median) to get a good drift vector, and use that from the old image to get the real position of the well that had a bad fit/go off of the ones with middle of the pack drift and map the outlier(s) to predicted positions based on where the other good ones are
- For images with one well, potentially skip over them and then get drift between image before and after it, divide that by two (or maybe time-weighted average of those two), and use that to get where the well in that image should be
- Play with paraboloid fit mask cutoff radius?
- Have it draw a 4um square around the point it estimates and look for the max in that square as the position to start fitting a paraboloid to (or maybe the min if the previous time it saw that well it was negative)? Maybe just the next point instead
- If paraboloid fit deflection > 0 and there’s a position in the image data within 4um that gives > 10 nm taller compared to substrate at that position, move to that position and iterative paraboloid fit again
    - Likewise for below surface and negative
- Instead of updating well_positions individually, treat the well_map as a rigid constellation. Fit the entire constellation to the found points in the new image using a Least Squares Rigid Transformation (finding the best translation (dx,dy) that minimizes error for all points simultaneously)
- Could use the vertex of the paraboloid, the image, and the diameter of the well to:
    1. Find what part of the image is the substrate (within 5-10 nm of the height given by calculate_substrate_height)
    2. See if the vertex is centered in x and y over a non-substrate patch in the image

# Better Feedback to User
Don’t hesitate to raise failures. I’ll learn from why they happened and either be able to automate it or it’ll just surface the strangest cases to me
- If it wanders more than (2um?) from its starting estimated point, have it have the user pick where it is → maybe showing other recently anchored points and the estimated position (and paraboloid fit path) on a stitched map to help me see where it should be
- Better visualizations to see why things go wrong: height map with predicted location, fit point, max and min within 3 um of each of those, maybe the path of iterative paraboloid fitting all marked on it, and any of the above that are relevant
- Overall, need to pick when it asks for user input
- Ability to split it up into different days/sessions to save/load, reduces clutter on the absolute positions map
    - Goal to look at all images between which the head hasn't been moved (just piezos move) → if I keep tracking them as separate line items in the Excel tracker, I can get the groups of images to look at at a time from there
    - When 3+ hours between images, tracker starts fresh automatically? Otherwise I can set where to split it up

# Other Improvements
- Make export only overwrite data points corresponding to the time period between the start and end of the images that it got its points from instead of erasing the whole file and rewriting it
    - Use get_scan_start_datetime and get_scan_end_datetime, take some of the logic in manual_image_review.py's _build_initial_selections out of it and make it reusable for this and that file
- Let it figure out which wells are which for itself. If it can find 5 wells in the first 10 images of sample37, it can figure out which is which based on relative positions and assign them to well IDs itself