# Summary

# Inadmissible solutions
- Set limits to what a deflection could possibly be (never more than +/- 350 nm)
- Check if the fitted vertex is within the x/y bounds of the image. If not, reject
- If there is no point in the data within 5 nm of the vertex, reject it (root sum of squares)
- Can check what the fit paraboloid would indicate about the well’s diameter (the circle at the intersection of the paraboloid and the surface). It’s not a perfect paraboloid, but saying it should be at least a 1 or 2 um diameter and less than 10 um diameter could be an amazing way to assess fit
- Assign a min R^2 value (0.2?) for a fit

# Better Finding Algorithm
- When multiple wells present in the same image, check that they are, within a tight margin, found to be within where they’d each predict the others to be
    - If not, use the one with the best R^2 to find the others
    - Consider that I can check what the drift is assuming each fit was correct, and compare those drifts. Outliers can indicate if one fit was bad. RANSAC (or just pick the median) to get a good drift vector, and use that from the old image to get the real position of the well that had a bad fit
- For images with one well, potentially skip over them and then get drift between image before and after it, divide that by two, and use that to get where the well in that image should be
- Play with paraboloid fit mask cutoff radius?
- Have it draw a 4um square around the point it estimates and look for the max in that square as the position to start fitting a paraboloid to (or maybe the min if the previous time it saw that well it was negative)?
- If it wanders more than (2um?) from its starting estimated point, have it have the user pick where it is → maybe showing previous one and estimated position on a stitched map to help me see where it should be
- Instead of updating well_positions individually, treat the well_map as a rigid constellation. Fit the entire constellation to the found points in the new image using a Least Squares Rigid Transformation (finding the best translation (dx,dy) that minimizes error for all points simultaneously)
- If found deflection > 0 and there’s a position in the image data within 4um that gives > 10 nm taller, move to that position and iterative paraboloid fit again
    - Likewise for below surface and negative

# Better Feedback to User
- Don’t hesitate to raise failures. I’ll learn from why they happened and either be able to automate it or it’ll just surface them to me
- Better visualizations to see why things go wrong: height map with predicted location, fit point, max and min within 3 um of each of those, maybe the path of iterative paraboloid fitting all marked on it, and any of the above that are relevant
- Ability to split it up into different days/sessions to save/load, reduces clutter on the absolute positions map
    - Goal to look at all images between which the head hasn't been moved (just piezos move)
    - When 3+ hours between images, tracker starts fresh automatically? Otherwise I can set where to split it up
- When tracking tons of them, maybe do open circles of the full colour palette, then move to triangles, then squares (keep X and star for final of each/overall)