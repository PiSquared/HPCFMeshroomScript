#!/usr/bin/env python
import re
import commands
import sys
import os
import collections

StepInfo = collections.namedtuple('StepInfo', "name partition qos time nodes tasks command")

def  SilentMkdir(theDir):
  if not os.path.isdir(theDir):
    os.makedirs(theDir)

def run_step(step):
  with open("{name}.slurm".format(**step), "w") as SLURM:
    SLURM.write("""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --output={name}.out
#SBATCH --error={name}.err
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --time={time}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={tasks}
#SBATCH --account=pi_donengel

{command}
""".format(**step))

def main():
    jobId = 0
    print("Prepping Scan, v2.")

    print(sys.argv)

    print(len(sys.argv))
    # modify for new arguments for photometrics
    if (len(sys.argv) != 6):
        print("usage: python run_alicevision.py <baseDir> <imgDir> <binDir> <numImages> <pmImgDir>")
        print("Must pass 6 arguments.")
        sys.exit(0)
    baseDir = sys.argv[1]
    srcImageDir = sys.argv[2]
    binDir = sys.argv[3]
    numImages = int(sys.argv[4])
    pmImgDir = sys.argv[5]

    print("Base dir  :            %s" % baseDir)
    print("Image dir :            %s" % srcImageDir)
    print("Bin dir   :            %s" % binDir)
    print("Num images:            %d" % numImages)
    print("Photometric image dir: %s" % pmImgDir)

    # start photometrics
    # iterator for each viewpoint
    viewIter = 0
    pmIdList = list()
    # Folder for the normal map output
    SilentMkdir(os.path.join(baseDir, "12_NormalMaps"))
    nmDir = os.path.join(baseDir, "12_NormalMaps/")

    # loop over each folder
    for curPMDir in os.listdir(srcImageDir):
        # Set variables for the run-rps script arguments
        # get mask
        mask = os.path.join(curPMDir, "mask.npy")
        # get lights
        lights = os.path.join(curPMDir, "lights.npy")
        # get image folder
        pmDir = os.path.join(curPMDir, "images")
        # decide where to put the final normal map
        output = os.path.join(nmDir, "normalMap%d" % viewIter)

        # create and run slurm file with those specificatons
        run_step({
            "name": ("photoMet%d" % viewIter),
            "partition": "batch",
            "qos": "short",
            "time": "00:05:00",
            "nodes": "1",
            "tasks": "1",
            "command": "python run_rps.py" + " " + mask + " " + lights + " " + pmDir + " " + output
        })

        # run the slurm file
        print("sbatch photoMet%d.slurm" % viewIter)
        status, jobId = commands.getstatusoutput("sbatch photoMet%d.slurm" % viewIter)
        jobId = int(re.search(r'\d+', jobId).group())
        pmIdList.append(jobId)

        #increment view counter
        viewIter += 1

    SilentMkdir(baseDir)

    # Camera Initialization
    SilentMkdir(os.path.join(baseDir, "00_CameraInit"))

    # provide the actual command to the slurm file
    binName = os.path.join(binDir, "aliceVision_cameraInit")

    dstDir = os.path.join(baseDir, "00_CameraInit/")
    cmdLine = binName
    cmdLine = cmdLine + " --defaultFieldOfView 45.0 --verboseLevel info --sensorDatabase \"\" --allowSingleView 1"
    cmdLine = cmdLine + " --imageFolder \"" + srcImageDir + "\""
    cmdLine = cmdLine + " --output \"" + os.path.join(dstDir, "cameraInit.sfm") + "\""

    run_step({
        "name": "camInit",
        "partition": "batch",
        "qos": "short",
        "time": "00:05:00",
        "nodes": "1",
        "tasks": "1",
        "command": cmdLine
    })

    # run the slurm file
    print("sbatch camInit.slurm")
    status, jobId = commands.getstatusoutput("sbatch camInit.slurm")
    jobId = int(re.search(r'\d+', jobId).group())

    #Feature Extraction
    SilentMkdir(os.path.join(baseDir, "01_FeatureExtraction"))

    srcSfm = os.path.join(baseDir, "00_CameraInit/cameraInit.sfm")

    binName = os.path.join(binDir, "aliceVision_featureExtraction")

    dstDir = os.path.join(baseDir, "01_FeatureExtraction/")

    cmdLine = "module load CUDA/9.2.148.1 \n" + binName
    cmdLine = cmdLine + " --describerTypes sift --forceCpuExtraction True --verboseLevel info --describerPreset normal"
    cmdLine = cmdLine + " --rangeStart 0 --rangeSize " + str(numImages)
    cmdLine = cmdLine + " --input \"" + srcSfm + "\""
    cmdLine = cmdLine + " --output \"" + dstDir + "\""

    run_step({
        "name": "featExtract",
        "partition": "gpu",
        "qos": "short",
        "time": "00:05:00",
        "nodes": "1",
        "tasks": "1",
        "command": cmdLine
    })

    # run the slurm file
    slurmCmd = ("sbatch --depend=afterany:%d featExtract.slurm" % jobId)
    print(slurmCmd)

    status, jobId = commands.getstatusoutput(slurmCmd)
    jobId = int(re.search(r'\d+', jobId).group())

    #Image Matching
    SilentMkdir(os.path.join(baseDir, "02_ImageMatching"))

    srcSfm = os.path.join(baseDir, "00_CameraInit/cameraInit.sfm")
    srcFeatures = os.path.join(baseDir, "01_FeatureExtraction/")
    dstMatches = os.path.join(baseDir, "02_ImageMatching/imageMatches.txt")

    binName = os.path.join(binDir, "aliceVision_imageMatching")

    cmdLine = binName
    cmdLine = cmdLine + " --minNbImages 200 --tree "" --maxDescriptors 500 --verboseLevel info --weights "" --nbMatches 50"
    cmdLine = cmdLine + " --input \"" + srcSfm + "\""
    cmdLine = cmdLine + " --featuresFolder \"" + srcFeatures + "\""
    cmdLine = cmdLine + " --output \"" + dstMatches + "\""

    run_step({
        "name": "imgMatch",
        "partition": "batch",
        "qos": "short",
        "time": "00:05:00",
        "nodes": "1",
        "tasks": "1",
        "command": cmdLine
    })

    # run the slurm file
    slurmCmd = ("sbatch --depend=afterany:%d imgMatch.slurm" % jobId)
    print(slurmCmd)

    status, jobId = commands.getstatusoutput(slurmCmd)
    jobId = int(re.search(r'\d+', jobId).group())

    #Feature Matching
    SilentMkdir(os.path.join(baseDir, "03_FeatureMatching"))

    srcSfm = os.path.join(baseDir, "00_CameraInit/cameraInit.sfm")
    srcFeatures = os.path.join(baseDir, "01_FeatureExtraction/")
    srcImageMatches = os.path.join(baseDir, "02_ImageMatching/imageMatches.txt")
    dstMatches = os.path.join(baseDir, "03_FeatureMatching")

    binName = os.path.join(binDir, "aliceVision_featureMatching")

    cmdLine = "module load CUDA/9.2.148.1 \n" + binName
    cmdLine = cmdLine + " --verboseLevel info --describerTypes sift --maxMatches 0 --exportDebugFiles False --savePutativeMatches False --guidedMatching False"
    cmdLine = cmdLine + " --geometricEstimator acransac --geometricFilterType fundamental_matrix --maxIteration 2048 --distanceRatio 0.8"
    cmdLine = cmdLine + " --photometricMatchingMethod ANN_L2"
    cmdLine = cmdLine + " --imagePairsList \"" + srcImageMatches + "\""
    cmdLine = cmdLine + " --input \"" + srcSfm + "\""
    cmdLine = cmdLine + " --featuresFolders \"" + srcFeatures + "\""
    cmdLine = cmdLine + " --output \"" + dstMatches + "\""

    run_step({
        "name": "featMatch",
        "partition": "gpu",
        "qos": "short",
        "time": "00:05:00",
        "nodes": "1",
        "tasks": "1",
        "command": cmdLine
    })

    # run the slurm file
    slurmCmd = ("sbatch --depend=afterany:%d featMatch.slurm" % jobId)
    print(slurmCmd)

    status, jobId = commands.getstatusoutput(slurmCmd)
    print(jobId)
    jobId = int(re.search(r'\d+', jobId).group())

    #Structure from Motion
    SilentMkdir(os.path.join(baseDir, "04_StructureFromMotion"))

    srcSfm = os.path.join(baseDir, "00_CameraInit/cameraInit.sfm")
    srcFeatures = os.path.join(baseDir, "01_FeatureExtraction/")
    srcImageMatches = os.path.join(baseDir, "02_ImageMatching/imageMatches.txt")
    srcMatches = os.path.join(baseDir, "03_FeatureMatching")
    dstDir = os.path.join(baseDir, "04_StructureFromMotion")

    binName = os.path.join(binDir, "/aliceVision_incrementalSfM")

    cmdLine = "module load CUDA/9.2.148.1 \n" + binName
    cmdLine = cmdLine + " --minAngleForLandmark 2.0 --minNumberOfObservationsForTriangulation 2 --maxAngleInitialPair 40.0 --maxNumberOfMatches 0 --localizerEstimator acransac --describerTypes sift --lockScenePreviouslyReconstructed False --localBAGraphDistance 1"
    cmdLine = cmdLine + " --initialPairA \"\" --initialPairB \"\" --interFileExtension .ply --useLocalBA True"
    cmdLine = cmdLine + " --minInputTrackLength 2 --useOnlyMatchesFromInputFolder False --verboseLevel info --minAngleForTriangulation 3.0 --maxReprojectionError 4.0 --minAngleInitialPair 5.0"

    cmdLine = cmdLine + " --input \"" + srcSfm + "\""
    cmdLine = cmdLine + " --featuresFolders \"" + srcFeatures + "\""
    cmdLine = cmdLine + " --matchesFolders \"" + srcMatches + "\""
    cmdLine = cmdLine + " --outputViewsAndPoses \"" + os.path.join(dstDir, "cameras.sfm") + "\""
    cmdLine = cmdLine + " --extraInfoFolder \"" + dstDir + "\""
    cmdLine = cmdLine + " --output \"" + os.path.join(dstDir, "sfm.abc") + "\""

    run_step({
        "name": "structMotion",
        "partition": "gpu",
        "qos": "short",
        "time": "00:05:00",
        "nodes": "1",
        "tasks": "1",
        "command": cmdLine
    })

    # run the slurm file
    slurmCmd = ("sbatch --depend=afterany:%d structMotion.slurm" % jobId)
    print(slurmCmd)

    status, jobId = commands.getstatusoutput(slurmCmd)
    jobId = int(re.search(r'\d+', jobId).group())

    #Prepare Dense Scene
    SilentMkdir(os.path.join(baseDir, "05_PrepareDenseScene"))

    srcSfm = os.path.join(baseDir, "04_StructureFromMotion/sfm.abc")
    dstDir = os.path.join(baseDir, "05_PrepareDenseScene")

    binName = os.path.join(binDir, "aliceVision_prepareDenseScene")

    cmdLine = binName
    cmdLine = cmdLine + " --verboseLevel info"
    cmdLine = cmdLine + " --input \"" + srcSfm + "\""
    cmdLine = cmdLine + " --output \"" + dstDir + "\""

    run_step({
        "name": "prepDense",
        "partition": "batch",
        "qos": "short",
        "time": "00:05:00",
        "nodes": "1",
        "tasks": "1",
        "command": cmdLine
    })

    # run the slurm file
    slurmCmd = ("sbatch --depend=afterany:%d prepDense.slurm" % jobId)
    print(slurmCmd)

    status, jobId = commands.getstatusoutput(slurmCmd)
    jobId = int(re.search(r'\d+', jobId).group())

    # Depth Mapping
    SilentMkdir(baseDir + "/06_DepthMap")

    groupSize = 3

    numGroups = (numImages + (groupSize - 1)) / groupSize

    srcIni = os.path.join(baseDir, "04_StructureFromMotion/sfm.abc")
    binName = os.path.join(binDir, "aliceVision_depthMapEstimation")
    imgDir = os.path.join(baseDir, "05_PrepareDenseScene")
    dstDir = os.path.join(baseDir, "06_DepthMap")

    cmdLine = "module load CUDA/9.2.148.1 \n" + binName
    cmdLine = cmdLine + " --sgmGammaC 5.5 --sgmWSH 4 --refineGammaP 8.0 --refineSigma 15 --refineNSamplesHalf 150 --sgmMaxTCams 10 --refineWSH 3 --downscale 2 --refineMaxTCams 6 --verboseLevel info --refineGammaC 15.5 --sgmGammaP 8.0"
    cmdLine = cmdLine + " --refineNiters 100 --refineNDepthsToRefine 31 --refineUseTcOrRcPixSize False"

    cmdLine = cmdLine + " --input \"" + srcIni + "\""
    cmdLine = cmdLine + " --imagesFolder \"" + imgDir + "\""
    cmdLine = cmdLine + " --output \"" + dstDir + "\""

    groupStart = 0
    groupSize = min(groupSize, numImages - groupStart)
    print("DepthMap Group %d/%d: %d, %d" % (0, numGroups, groupStart, groupSize))

    cmd = cmdLine + (" --rangeStart %d --rangeSize %d" % (groupStart, groupSize))

    run_step({
        "name": ("depthMap%d" % 0),
        "partition": "gpu",
        "qos": "short",
        "time": "00:15:00",
        "nodes": "1",
        "tasks": "1",
        "command": cmd
    })

    # run the slurm file
    slurmCmd = ("sbatch --depend=afterany:%d depthMap%d.slurm" % (jobId, 0))
    print(slurmCmd)

    #list for the job ids
    jobIdList = list()

    status, jobId = commands.getstatusoutput(slurmCmd)
    jobId = int(re.search(r'\d+', jobId).group())
    jobIdList.append(jobId)

    # The rest of the rounds of depth mapping
    for groupIter in range(1, numGroups):
        groupStart = groupSize * groupIter
        groupSize = min(groupSize, numImages - groupStart)
        print("DepthMap Group %d/%d: %d, %d" % (groupIter, numGroups, groupStart, groupSize))

        cmd = cmdLine + (" --rangeStart %d --rangeSize %d" % (groupStart, groupSize))

        run_step({
            "name": ("depthMap%d" % groupIter),
            "partition": "gpu",
            "qos": "short",
            "time": "00:15:00",
            "nodes": "1",
            "tasks": "1",
            "command": cmd
        })

        # run the slurm file
        slurmCmd = ("sbatch --depend=after:%d depthMap%d.slurm" % (jobId, groupIter))
        print(slurmCmd)

        status, jobId = commands.getstatusoutput(slurmCmd)
        jobId = int(re.search(r'\d+', jobId).group())
        jobIdList.append(jobId)

    #Depth Map Filtering
    SilentMkdir(os.path.join(baseDir, "07_DepthMapFilter"))

    binName = os.path.join(binDir, "aliceVision_depthMapFiltering")
    dstDir = os.path.join(baseDir, "07_DepthMapFilter")
    srcIni = os.path.join(baseDir, "04_StructureFromMotion/sfm.abc")
    srcDepthDir = os.path.join(baseDir, "06_DepthMap")

    cmdLine = binName
    cmdLine = cmdLine + " --minNumOfConsistentCamsWithLowSimilarity 4"
    cmdLine = cmdLine + " --minNumOfConsistentCams 3 --verboseLevel info --pixSizeBall 0"
    cmdLine = cmdLine + " --pixSizeBallWithLowSimilarity 0 --nNearestCams 10"

    cmdLine = cmdLine + " --input \"" + srcIni + "\""
    cmdLine = cmdLine + " --output \"" + dstDir + "\""
    cmdLine = cmdLine + " --depthMapsFolder \"" + srcDepthDir + "\""

    run_step({
        "name": "depthMapFilter",
        "partition": "batch",
        "qos": "short",
        "time": "00:15:00",
        "nodes": "1",
        "tasks": "1",
        "command": cmd
    })

    # run the slurm file
    slurmCmd = "sbatch --depend=afterany"

    # for loop over all of the jobs in the job id list
    for jid in jobIdList:
        slurmCmd = slurmCmd + (":%d" % jid)

    slurmCmd = slurmCmd + " depthMapFilter.slurm"

    print(slurmCmd)

    status, jobId = commands.getstatusoutput(slurmCmd)
    jobId = int(re.search(r'\d+', jobId).group())

    #Meshing
    SilentMkdir(os.path.join(baseDir, "09_Meshing"))

    binName = os.path.join(binDir, "aliceVision_meshing")
    srcIni = os.path.join(baseDir, "04_StructureFromMotion/sfm.abc")
    srcDepthFilterDir = os.path.join(baseDir, "07_DepthMapFilter")
    srcDepthMapDir = os.path.join(baseDir, "06_DepthMap")

    dstDir = os.path.join(baseDir, "08_Meshing")

    cmdLine = binName
    cmdLine = cmdLine + " --simGaussianSizeInit 10.0 --maxInputPoints 50000000 --repartition multiResolution"
    cmdLine = cmdLine + " --simGaussianSize 10.0 --simFactor 15.0 --voteMarginFactor 4.0 --contributeMarginFactor 2.0 --minStep 2 --pixSizeMarginFinalCoef 4.0 --maxPoints 5000000 --maxPointsPerVoxel 1000000 --angleFactor 15.0 --partitioning singleBlock"
    cmdLine = cmdLine + " --minAngleThreshold 1.0 --pixSizeMarginInitCoef 2.0 --refineFuse True --verboseLevel info"

    cmdLine = cmdLine + " --input \"" + srcIni + "\""
    cmdLine = cmdLine + " --depthMapsFilterFolder \"" + srcDepthFilterDir + "\""
    cmdLine = cmdLine + " --depthMapsFolder \"" + srcDepthMapDir + "\""
    cmdLine = cmdLine + " --output \"" + os.path.join(dstDir, "mesh.obj") + "\""

    run_step({
        "name": "mesh",
        "partition": "batch",
        "qos": "short",
        "time": "00:10:00",
        "nodes": "1",
        "tasks": "1",
        "command": cmdLine
    })

    # run the slurm file
    slurmCmd = ("sbatch --depend=afterany:%d mesh.slurm" % jobId)
    print(slurmCmd)

    status, jobId = commands.getstatusoutput(slurmCmd)
    jobId = int(re.search(r'\d+', jobId).group())

    # Mesh Filtering
    SilentMkdir(os.path.join(baseDir, "09_MeshFiltering"))

    # create a new slurm file
    slurm = open("meshFilt.slurm", "w+")

    # provide sbatch info to the slurm file
    slurm.write("#!/bin/bash \n")
    slurm.write("#SBATCH --job-name=meshFilt \n")
    slurm.write("#SBATCH --output=meshFilt.out \n")
    slurm.write("#SBATCH --error=meshFilt.err \n")
    slurm.write("#SBATCH --partition=batch \n")
    slurm.write("#SBATCH --qos=normal \n")
    slurm.write("#SBATCH --time=00:05:00 \n")
    slurm.write("#SBATCH --nodes=1 \n")
    slurm.write("#SBATCH --ntasks-per-node=1 \n \n")

    binName = os.path.join(binDir, "aliceVision_meshFiltering")

    srcMesh = os.path.join(baseDir, "08_Meshing/mesh.obj")
    dstMesh = os.path.join(baseDir, "09_MeshFiltering/mesh.obj")

    cmdLine = binName
    cmdLine = cmdLine + " --verboseLevel info --removeLargeTrianglesFactor 60.0 --iterations 5 --keepLargestMeshOnly True"
    cmdLine = cmdLine + " --lambda 1.0"

    cmdLine = cmdLine + " --input \"" + srcMesh + "\""
    cmdLine = cmdLine + " --output \"" + dstMesh + "\""

    run_step({
        "name": "meshFilt",
        "partition": "batch",
        "qos": "short",
        "time": "00:05:00",
        "nodes": "1",
        "tasks": "1",
        "command": cmdLine
    })

    # run the slurm file
    slurmCmd = ("sbatch --depend=afterany:%d meshFilt.slurm" % jobId)
    print(slurmCmd)

    status, jobId = commands.getstatusoutput(slurmCmd)
    jobId = int(re.search(r'\d+', jobId).group())

    # Texturing
    SilentMkdir(os.path.join(baseDir, "10_Texturing"))

    binName = os.path.join(binDir, "aliceVision_texturing")

    srcMesh = os.path.join(baseDir, "09_MeshFiltering/mesh.obj")
    srcRecon = os.path.join(baseDir, "08_Meshing/denseReconstruction.bin")
    srcIni = os.path.join(baseDir, "04_StructureFromMotion/sfm.abc")
    dstDir = os.path.join(baseDir, "10_Texturing")
    imgDir = os.path.join(baseDir, "05_PrepareDenseScene")

    cmdLine = binName
    cmdLine = cmdLine + " --textureSide 8192"
    cmdLine = cmdLine + " --downscale 2 --verboseLevel info --padding 15"
    cmdLine = cmdLine + " --unwrapMethod Basic --outputTextureFileType png --flipNormals False --fillHoles False"

    cmdLine = cmdLine + " --inputDenseReconstruction \"" + srcRecon + "\""
    cmdLine = cmdLine + " --inputMesh \"" + srcMesh + "\""
    cmdLine = cmdLine + " --input \"" + srcIni + "\""
    cmdLine = cmdLine + " --imagesFolder \"" + imgDir + "\""
    cmdLine = cmdLine + " --output \"" + dstDir + "\""

    run_step({
        "name": "text",
        "partition": "batch",
        "qos": "short",
        "time": "00:05:00",
        "nodes": "1",
        "tasks": "1",
        "command": cmdLine
    })

    # run the slurm file
    slurmCmd = ("sbatch --depend=afterany:%d text.slurm" % jobId)
    print(slurmCmd)

    # check that the photometric step and all of the photogrammetric steps are finished

    # re run the texturing step with the normals for the images
    SilentMkdir(os.path.join(baseDir, "11_NormalTexturing"))

    binName = os.path.join(binDir, "aliceVision_texturing")

    srcMesh = os.path.join(baseDir, "09_MeshFiltering/mesh.obj")
    srcRecon = os.path.join(baseDir, "08_Meshing/denseReconstruction.bin")
    srcIni = os.path.join(baseDir, "04_StructureFromMotion/sfm.abc")
    dstDir = os.path.join(baseDir, "11_NormalTexturing")
    imgDir = os.path.join(baseDir, "12_NormalMaps")

    cmdLine = binName
    cmdLine = cmdLine + " --textureSide 8192"
    cmdLine = cmdLine + " --downscale 2 --verboseLevel info --padding 15"
    cmdLine = cmdLine + " --unwrapMethod Basic --outputTextureFileType png --flipNormals False --fillHoles False"

    cmdLine = cmdLine + " --inputDenseReconstruction \"" + srcRecon + "\""
    cmdLine = cmdLine + " --inputMesh \"" + srcMesh + "\""
    cmdLine = cmdLine + " --input \"" + srcIni + "\""
    cmdLine = cmdLine + " --imagesFolder \"" + imgDir + "\""
    cmdLine = cmdLine + " --output \"" + dstDir + "\""

    run_step({
        "name": "textNorm",
        "partition": "batch",
        "qos": "short",
        "time": "00:05:00",
        "nodes": "1",
        "tasks": "1",
        "command": cmdLine
    })

    # run the slurm file
    slurmCmd = ("sbatch --depend=afterany:%d" % jobId)
    # for loop over all of the jobs in the photometric job id list
    for jid in pmIdList:
        slurmCmd = slurmCmd + (":%d" % jid)

    slurmCmd = slurmCmd + " textNorm.slurm"
    print(slurmCmd)
    status, jobId = commands.getstatusoutput(slurmCmd)

main()
