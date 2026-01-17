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
    wait(2000);
    imgTitle = getTitle();
    selectWindow(imgTitle);

    getDimensions(width, height, channels, slices, frames);  // XY, C, Z, T
    totalT = frames;
    totalZ = slices;
    totalC = channels;

    if (totalC != 2) {
        showMessage("File skipped", "Expected 2 channels, but found: " + totalC);
        close(imgTitle);
        continue;
    }

    // --- Generate MaxIP stacks for each channel ---
    for (c = 1; c <= 2; c++) {
        newImage("MaxIP_C" + c, "32-bit black", width, height, totalT);
    }

    for (t = 1; t <= totalT; t++) {
        for (c = 1; c <= 2; c++) {
            selectWindow(imgTitle);
            Stack.setPosition(c, 1, t);
            run("Duplicate...", "title=ZStack_C" + c + "_t" + t + " duplicate channels=" + c + " slices=1-" + totalZ + " frames=" + t);
            run("Z Project...", "projection=[Max Intensity]");
            selectWindow("MAX_ZStack_C" + c + "_t" + t);
            run("Copy");
            selectWindow("MaxIP_C" + c);
            setSlice(t);
            run("Paste");

            close("MAX_ZStack_C" + c + "_t" + t);
            close("ZStack_C" + c + "_t" + t);
        }
    }

    // --- Merge the two MaxIP stacks into composite ---
    selectWindow("MaxIP_C1");
    run("Grays");
    selectWindow("MaxIP_C2");
    run("Grays");

    imageCalculator("Combine...", "MaxIP_C1", "MaxIP_C2");
    rename("MaxIP_Composite");

    // --- Apply SIFT alignment (Multichannel) ---
    run("Linear Stack Alignment with SIFT (multichannel)",
        "initial_gaussian_blur=1.60 steps_per_scale_octave=3 minimum_image_size=64 maximum_image_size=1024 " +
        "feature_descriptor_size=4 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 " +
        "maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=Rigid interpolate");

    rename("Aligned_Composite");

    // --- Split channels back ---
    run("Split Channels");

    // Now two windows: "C1-Aligned_Composite", "C2-Aligned_Composite"
    rename("C1-Aligned_Composite", "Aligned_C1_" + name);
    rename("C2-Aligned_Composite", "Aligned_C2_" + name);

    // --- Add time label + save ---
    for (c = 1; c <= 2; c++) {
        selectWindow("Aligned_C" + c + "_" + name);
        run("Enhance Contrast", "saturated=0.35");
        run("Label...", "format=00:00:00 starting=0 interval=30 x=850 y=985 font=40 text=[] range=1-" + totalT + " use use_text");
        saveAs("Tiff", dir + name + "_C" + c + "_Aligned_MaxIP_" + totalT + "frames.tif");
        close();
    }

    // Clean up
    close("MaxIP_C1");
    close("MaxIP_C2");
    close("MaxIP_Composite");
    close(imgTitle);
}

setBatchMode(false);
showMessage("Done", "Saved " + nFiles + " ND2s with 2-channel aligned MaxIP projections.");
