dir = getDirectory("Choose a folder with ND2 files");
list = getFileList(dir);

// --- Filter ND2 files ---
nd2List = "";
for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], ".nd2")) {
        nd2List += list[i] + "\n";
    }
}
nd2Array = split(nd2List, "\n");

nFiles = nd2Array.length;
if (nFiles == 0 || nd2Array[0] == "") {
    showMessage("No ND2 files found.");
    exit();
}

setBatchMode(true);

for (i = 0; i < nFiles; i++) {
    filename = nd2Array[i];
    if (filename == "") continue;

    showProgress(i, nFiles);
    print("Processing file " + (i+1) + " of " + nFiles + ": " + filename);

    name = replace(filename, ".nd2", "");

    // --- Open ND2 using Bio-Formats ---
    run("Bio-Formats Importer", "open=[" + dir + filename + "] autoscale color_mode=Default view=Hyperstack stack_order=XYCZT");
    wait(2000); // Wait for image to load
    imgTitle = getTitle(); // Get actual window title
    selectWindow(imgTitle);

    getDimensions(width, height, channels, slices, frames);  // XY, Z, T
    totalT = frames;
    totalZ = slices;

    projectionTitle = "MaxIP_Stack_" + name;
    newImage(projectionTitle, "32-bit black", width, height, totalT);
    selectWindow(projectionTitle);

    for (t = 1; t <= totalT; t++) {
        selectWindow(imgTitle);
        Stack.setPosition(1, 1, t); // C=1, Z=1, T=t

        // Duplicate Z-stack at time t
        run("Duplicate...", "title=ZStack_t" + t + " duplicate channels=1 slices=1-" + totalZ + " frames=" + t);
        selectWindow("ZStack_t" + t);

        // MaxIP over Z
        run("Z Project...", "projection=[Max Intensity]");
        wait(200);
        selectWindow("MAX_ZStack_t" + t);

        // Copy into final stack
        run("Copy");
        selectWindow(projectionTitle);
        setSlice(t);
        run("Paste");

        // Cleanup
        close("MAX_ZStack_t" + t);
        close("ZStack_t" + t);
    }

    // Align MaxIP stack
    selectWindow(projectionTitle);
    run("Enhance Contrast", "saturated=0.35");

    run("Linear Stack Alignment with SIFT", 
        "initial_gaussian_blur=1.60 steps_per_scale_octave=3 minimum_image_size=64 maximum_image_size=1024 " +
        "feature_descriptor_size=4 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 " +
        "maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=Rigid interpolate");

    run("Label...", "format=00:00:00 starting=0 interval=30 x=850 y=985 font=40 text=[] range=1-" + totalT + " use use_text");

    saveAs("Tiff", dir + name + "_Aligned_MaxIP_121frames.tif");
    close();  // Close MaxIP stack

    // Close original ND2 image
    selectWindow(imgTitle);
    close();
}

setBatchMode(false);
showMessage("Done", "Saved " + nFiles + " aligned MaxIP 121-frame TIFFs.");
