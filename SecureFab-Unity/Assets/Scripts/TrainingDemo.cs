using UnityEngine;
using SecureFab.Training;
using Unity.XR.PXR;
using Unity.XR.PXR.SecureMR;
using System.Threading;
using System;
using System.Collections.Generic;

/// <summary>
/// Complete SecureFab training demo with YOLO detection and step validation.
/// Detects bottle, cup, scissors, book and validates placement against training steps.
/// </summary>
public class TrainingDemo : MonoBehaviour
{
    #region Inspector Fields

    [Header("ML Models")]
    [Tooltip("YOLO model for object detection (yolov8n_fp16_qnn229.bytes)")]
    public TextAsset yoloModel;

    [Tooltip("GLTF asset for rendering detection labels")]
    public TextAsset labelGltfAsset;

    [Header("References")]
    [Tooltip("Reference to StepManager in scene")]
    public StepManager stepManager;

    [Header("Zone Configuration")]
    [Tooltip("Zone detection thresholds (normalized 0-1)")]
    [Range(0f, 0.5f)]
    public float leftZoneX = 0.33f;

    [Range(0.5f, 1f)]
    public float rightZoneX = 0.66f;

    [Range(0f, 0.5f)]
    public float topZoneY = 0.33f;

    [Range(0.5f, 1f)]
    public float bottomZoneY = 0.66f;

    [Header("Detection Settings")]
    [Tooltip("YOLO confidence threshold")]
    [Range(0f, 1f)]
    public float confidenceThreshold = 0.5f;

    [Tooltip("NMS IoU threshold")]
    [Range(0f, 1f)]
    public float nmsThreshold = 0.5f;

    [Tooltip("Number of objects to detect")]
    public int numberOfObjects = 4;

    [Tooltip("Minimum frames with matching config before validation")]
    [Range(1, 30)]
    public int stabilityFrames = 10;

    [Header("VST Settings")]
    [Tooltip("VST image width")]
    public int vstWidth = 640;
    
    [Tooltip("VST image height")]
    public int vstHeight = 640;

    [Header("Visual Feedback")]
    [Tooltip("Color when configuration is correct (applied to zone indicators)")]
    public Color validColor = Color.green;

    [Tooltip("Color when configuration is incorrect")]
    public Color invalidColor = Color.red;

    [Tooltip("Color when waiting for objects")]
    public Color waitingColor = Color.yellow;

    [Header("Debug")]
    [Tooltip("Enable debug logging")]
    public bool debugLogging = true;

    [Tooltip("Show zone visualizations")]
    public bool showZoneGizmos = true;

    [Tooltip("Show detected objects in console")]
    public bool logDetections = false;

    #endregion

    #region Private Fields

    private int frameCount = 0;
    private bool isInitialized = false;
    private bool pipelinesInitialized = false;

    // Training validation state
    private ExpectedConfig currentDetectedConfig;
    private ExpectedConfig lastStableConfig;
    private int stableFrameCount = 0;
    private bool lastValidationResult = false;
    private float lastValidationTime = 0f;

    // COCO class IDs for our training objects
    private const int COCO_BOTTLE = 39;
    private const int COCO_CUP = 41;
    private const int COCO_SCISSORS = 76;
    private const int COCO_BOOK = 73;

    // Mapping from COCO class ID to our object names
    private readonly Dictionary<int, string> trainingObjectMap = new Dictionary<int, string>
    {
        { COCO_BOTTLE, "bottle" },
        { COCO_CUP, "cup" },
        { COCO_SCISSORS, "scissors" },
        { COCO_BOOK, "book" }
    };

    // Full COCO class names
    private readonly string[] classNames = new string[]
    {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    };

    // Detection results (updated by inference thread, read by validation)
    private readonly object detectionLock = new object();
    private List<DetectedObject> latestDetections = new List<DetectedObject>();

    // SecureMR components
    private Provider provider;
    private Pipeline vstPipeline;
    private Pipeline inferencePipeline;
    private Pipeline map2dTo3dPipeline;
    private Pipeline renderPipeline;
    
    // Global tensors
    private Tensor vstOutputLeftUint8Global;
    private Tensor vstOutputRightUint8Global;
    private Tensor vstOutputLeftFp32Global;
    private Tensor vstTimestampGlobal;
    private Tensor vstCameraMatrixGlobal;
    private Tensor nmsBoxesGlobal;
    private Tensor nmsScoresGlobal;
    private Tensor classesSelectGlobal;
    private Tensor pointXYZGlobal;
    private Tensor scaleGlobal;
    private Tensor gltfAsset;
    private Tensor gltfAsset1;
    private Tensor gltfAsset2;
    private Tensor gltfAsset3;
    
    // Pipeline placeholders
    private Tensor vstOutputLeftUint8Placeholder;
    private Tensor vstOutputRightUint8Placeholder;
    private Tensor vstOutputLeftFp32Placeholder;
    private Tensor vstTimestampPlaceholder;
    private Tensor vstCameraMatrixPlaceholder;
    private Tensor vstImagePlaceholder;
    private Tensor nmsBoxesPlaceholder;
    private Tensor nmsScoresPlaceholder;
    private Tensor classesSelectPlaceholder;
    private Tensor nmsBoxesPlaceholder1;
    private Tensor timestampPlaceholder1;
    private Tensor cameraMatrixPlaceholder1;
    private Tensor leftImagePlaceholder;
    private Tensor rightImagePlaceholder;
    private Tensor pointXYZPlaceholder;
    private Tensor scalePlaceholder;
    private Tensor pointXYZPlaceholder1;
    private Tensor timestampPlaceholder2;
    private Tensor classesSelectPlaceholder1;
    private Tensor nmsScoresPlaceholder1;
    private Tensor scalePlaceholder1;
    private Tensor gltfPlaceholderTensor;
    private Tensor gltfPlaceholderTensor1;
    private Tensor gltfPlaceholderTensor2;
    private Tensor gltfPlaceholderTensor3;
    
    // Threading
    private Thread pipelineInitThread;
    private Thread vstRunnerThread;
    private Thread inferenceRunnerThread;
    private Thread map2dRunnerThread;
    private Thread renderRunnerThread;
    private bool keepRunning = true;
    private readonly object initLock = new object();

    // Detection result extraction
    private float[] boxesData;
    private float[] scoresData;
    private float[] classesData;

    #endregion

    #region Helper Structures

    private class DetectedObject
    {
        public int classId;
        public float confidence;
        public float centerX;
        public float centerY;
        public float width;
        public float height;
        public string objectName;
        public string zone;

        public override string ToString()
        {
            return $"{objectName}@{zone} ({confidence:F2})";
        }
    }

    #endregion

    #region Unity Lifecycle

    private void Awake()
    {
        PXR_Manager.EnableVideoSeeThrough = true;
    }

    private void Start()
    {
        InitializeDemo();
    }

    private void Update()
    {
        if (!isInitialized || !pipelinesInitialized) return;

        frameCount++;

        // Process detections and validate configuration every frame
        ProcessLatestDetections();
    }

    private void OnDestroy()
    {
        keepRunning = false;
        
        pipelineInitThread?.Join(1000);
        vstRunnerThread?.Join(1000);
        inferenceRunnerThread?.Join(1000);
        map2dRunnerThread?.Join(1000);
        renderRunnerThread?.Join(1000);
    }

    private void OnDrawGizmos()
    {
#if UNITY_EDITOR
        if (!showZoneGizmos) return;
        DrawZoneBoundaries();
#endif
    }

    #endregion

    #region Initialization

    private void InitializeDemo()
    {
        LogDebug("Initializing SecureFab Training Demo...");

        if (stepManager == null)
        {
            Debug.LogError("[TrainingDemo] StepManager reference missing!");
            enabled = false;
            return;
        }

        if (!stepManager.IsInitialized)
        {
            Debug.LogError("[TrainingDemo] StepManager not initialized!");
            enabled = false;
            return;
        }

        if (yoloModel == null)
        {
            Debug.LogError("[TrainingDemo] YOLO model not assigned!");
            enabled = false;
            return;
        }

        if (labelGltfAsset == null)
        {
            Debug.LogError("[TrainingDemo] Label GLTF not assigned!");
            enabled = false;
            return;
        }

        // Subscribe to step manager events
        stepManager.onStepChanged.AddListener(OnStepChanged);
        stepManager.onConfigurationValidated.AddListener(OnConfigValidated);

        // Initialize arrays for detection data
        boxesData = new float[numberOfObjects * 4];
        scoresData = new float[numberOfObjects];
        classesData = new float[numberOfObjects];

        // Initialize SecureMR
        InitializeSecureMRPipeline();

        isInitialized = true;
        LogDebug($"Initialized! Current step: {stepManager.CurrentStep.title}");
    }

    private void InitializeSecureMRPipeline()
    {
        LogDebug("Initializing SecureMR pipelines...");
        
        try
        {
            provider = new Provider(vstWidth, vstHeight);
            
            pipelineInitThread = new Thread(() =>
            {
                try
                {
                    CreateVSTImagePipeline();
                    CreateModelInferencePipeline();
                    CreateMap2dTo3dPipeline();
                    CreateRenderingPipeline();
                    
                    lock (initLock)
                    {
                        pipelinesInitialized = true;
                    }
                    
                    LogDebug("All pipelines initialized!");
                    StartPipelineRunners();
                }
                catch (Exception e)
                {
                    Debug.LogError($"[TrainingDemo] Pipeline init failed: {e.Message}\n{e.StackTrace}");
                }
            });
            
            pipelineInitThread.Start();
        }
        catch (Exception e)
        {
            Debug.LogError($"[TrainingDemo] Failed to initialize: {e.Message}\n{e.StackTrace}");
            enabled = false;
        }
    }

    #endregion

    #region VST Pipeline

    private void CreateVSTImagePipeline()
    {
        LogDebug("Creating VST pipeline...");
        
        vstPipeline = provider.CreatePipeline();
        
        vstOutputLeftUint8Global = provider.CreateTensor<byte, Matrix>(
            3, new TensorShape(new[] { vstHeight, vstWidth }));
        vstOutputRightUint8Global = provider.CreateTensor<byte, Matrix>(
            3, new TensorShape(new[] { vstHeight, vstWidth }));
        vstOutputLeftFp32Global = provider.CreateTensor<float, Matrix>(
            3, new TensorShape(new[] { vstHeight, vstWidth }));
        vstTimestampGlobal = provider.CreateTensor<int, TimeStamp>(
            4, new TensorShape(new[] { 1 }));
        vstCameraMatrixGlobal = provider.CreateTensor<float, Matrix>(
            1, new TensorShape(new[] { 3, 3 }));
        
        vstOutputLeftUint8Placeholder = vstPipeline.CreateTensorReference<byte, Matrix>(
            3, new TensorShape(new[] { vstHeight, vstWidth }));
        vstOutputRightUint8Placeholder = vstPipeline.CreateTensorReference<byte, Matrix>(
            3, new TensorShape(new[] { vstHeight, vstWidth }));
        vstOutputLeftFp32Placeholder = vstPipeline.CreateTensorReference<float, Matrix>(
            3, new TensorShape(new[] { vstHeight, vstWidth }));
        vstTimestampPlaceholder = vstPipeline.CreateTensorReference<int, TimeStamp>(
            4, new TensorShape(new[] { 1 }));
        vstCameraMatrixPlaceholder = vstPipeline.CreateTensorReference<float, Matrix>(
            1, new TensorShape(new[] { 3, 3 }));
        
        var vstOp = vstPipeline.CreateOperator<RectifiedVstAccessOperator>();
        vstOp.SetResult("left image", vstOutputLeftUint8Placeholder);
        vstOp.SetResult("right image", vstOutputRightUint8Placeholder);
        vstOp.SetResult("timestamp", vstTimestampPlaceholder);
        vstOp.SetResult("camera matrix", vstCameraMatrixPlaceholder);
        
        var assignOp = vstPipeline.CreateOperator<AssignmentOperator>();
        assignOp.SetOperand("src", vstOutputLeftUint8Placeholder);
        assignOp.SetResult("dst", vstOutputLeftFp32Placeholder);
        
        var normalizeOp = vstPipeline.CreateOperator<ArithmeticComposeOperator>(
            new ArithmeticComposeOperatorConfiguration("{0} / 255.0"));
        normalizeOp.SetOperand("{0}", vstOutputLeftFp32Placeholder);
        normalizeOp.SetResult("result", vstOutputLeftFp32Placeholder);
        
        LogDebug("VST pipeline created!");
    }

    private void RunVSTImagePipeline()
    {
        var tensorMapping = new TensorMapping();
        tensorMapping.Set(vstOutputLeftUint8Placeholder, vstOutputLeftUint8Global);
        tensorMapping.Set(vstOutputRightUint8Placeholder, vstOutputRightUint8Global);
        tensorMapping.Set(vstTimestampPlaceholder, vstTimestampGlobal);
        tensorMapping.Set(vstCameraMatrixPlaceholder, vstCameraMatrixGlobal);
        tensorMapping.Set(vstOutputLeftFp32Placeholder, vstOutputLeftFp32Global);
        
        vstPipeline.Execute(tensorMapping);
    }

    #endregion

    #region Model Inference Pipeline

    private void CreateModelInferencePipeline()
    {
        LogDebug("Creating YOLO inference pipeline...");
        
        inferencePipeline = provider.CreatePipeline();
        
        vstImagePlaceholder = inferencePipeline.CreateTensorReference<float, Matrix>(
            3, new TensorShape(new[] { vstHeight, vstWidth }));
        
        var yoloOutput = inferencePipeline.CreateTensor<float, Matrix>(
            1, new TensorShape(new[] { 8400, 84 }));
        
        var modelConfig = new ModelOperatorConfiguration(
            yoloModel.bytes,
            SecureMRModelType.QnnContextBinary,
            "yolo"
        );
        modelConfig.AddInputMapping("images", "images", SecureMRModelEncoding.Float32);
        modelConfig.AddOutputMapping("output0", "output0", SecureMRModelEncoding.Float32);
        
        var modelOp = inferencePipeline.CreateOperator<RunModelInferenceOperator>(modelConfig);
        modelOp.SetOperand("images", vstImagePlaceholder);
        modelOp.SetResult("output0", yoloOutput);
        
        var boxes = inferencePipeline.CreateTensor<float, Matrix>(
            1, new TensorShape(new[] { 8400, 4 }));
        var scores = inferencePipeline.CreateTensor<float, Matrix>(
            1, new TensorShape(new[] { 8400, 80 }));
        
        var assignBoxes = inferencePipeline.CreateOperator<AssignmentOperator>();
        assignBoxes.SetOperand("src", yoloOutput);
        assignBoxes.SetOperand("src slices", CreateSliceTensor(inferencePipeline, 0, 8400, 0, 4));
        assignBoxes.SetResult("dst", boxes);
        
        var assignScores = inferencePipeline.CreateOperator<AssignmentOperator>();
        assignScores.SetOperand("src", yoloOutput);
        assignScores.SetOperand("src slices", CreateSliceTensor(inferencePipeline, 0, 8400, 4, 84));
        assignScores.SetResult("dst", scores);
        
        var xcenterycenter = inferencePipeline.CreateTensor<float, Matrix>(
            1, new TensorShape(new[] { 8400, 2 }));
        var widthHeight = inferencePipeline.CreateTensor<float, Matrix>(
            1, new TensorShape(new[] { 8400, 2 }));
        var xminymin = inferencePipeline.CreateTensor<float, Matrix>(
            1, new TensorShape(new[] { 8400, 2 }));
        var xmaxymax = inferencePipeline.CreateTensor<float, Matrix>(
            1, new TensorShape(new[] { 8400, 2 }));
        
        var assignCenter = inferencePipeline.CreateOperator<AssignmentOperator>();
        assignCenter.SetOperand("src", boxes);
        assignCenter.SetOperand("src slices", CreateSliceTensor(inferencePipeline, 0, 8400, 0, 2));
        assignCenter.SetResult("dst", xcenterycenter);
        
        var assignWH = inferencePipeline.CreateOperator<AssignmentOperator>();
        assignWH.SetOperand("src", boxes);
        assignWH.SetOperand("src slices", CreateSliceTensor(inferencePipeline, 0, 8400, 2, 4));
        assignWH.SetResult("dst", widthHeight);
        
        var calcMin = inferencePipeline.CreateOperator<ArithmeticComposeOperator>(
            new ArithmeticComposeOperatorConfiguration("({0} - {1} / 2)"));
        calcMin.SetOperand("{0}", xcenterycenter);
        calcMin.SetOperand("{1}", widthHeight);
        calcMin.SetResult("result", xminymin);
        
        var calcMax = inferencePipeline.CreateOperator<ArithmeticComposeOperator>(
            new ArithmeticComposeOperatorConfiguration("({0} + {1} / 2)"));
        calcMax.SetOperand("{0}", xcenterycenter);
        calcMax.SetOperand("{1}", widthHeight);
        calcMax.SetResult("result", xmaxymax);
        
        var assignMin = inferencePipeline.CreateOperator<AssignmentOperator>();
        assignMin.SetOperand("src", xminymin);
        assignMin.SetOperand("dst slices", CreateSliceTensor(inferencePipeline, 0, 8400, 0, 2));
        assignMin.SetResult("dst", boxes);
        
        var assignMax = inferencePipeline.CreateOperator<AssignmentOperator>();
        assignMax.SetOperand("src", xmaxymax);
        assignMax.SetOperand("dst slices", CreateSliceTensor(inferencePipeline, 0, 8400, 2, 4));
        assignMax.SetResult("dst", boxes);
        
        var sortedScores = inferencePipeline.CreateTensor<float, Matrix>(
            1, new TensorShape(new[] { 8400, 80 }));
        var sortedIndices = inferencePipeline.CreateTensor<int, Matrix>(
            1, new TensorShape(new[] { 8400, 80 }));
        
        var sortOp = inferencePipeline.CreateOperator<SortMatrixByRowOperator>();
        sortOp.SetOperand("input", scores);
        sortOp.SetResult("sorted values", sortedScores);
        sortOp.SetResult("sorted indices", sortedIndices);
        
        var bestScores = inferencePipeline.CreateTensor<float, Matrix>(
            1, new TensorShape(new[] { 8400, 1 }));
        var bestIndices = inferencePipeline.CreateTensor<int, Matrix>(
            1, new TensorShape(new[] { 8400, 1 }));
        
        var assignBestScores = inferencePipeline.CreateOperator<AssignmentOperator>();
        assignBestScores.SetOperand("src", sortedScores);
        assignBestScores.SetOperand("src slices", CreateSliceTensor(inferencePipeline, 0, 8400, 0, 1));
        assignBestScores.SetResult("dst", bestScores);
        
        var assignBestIndices = inferencePipeline.CreateOperator<AssignmentOperator>();
        assignBestIndices.SetOperand("src", sortedIndices);
        assignBestIndices.SetOperand("src slices", CreateSliceTensor(inferencePipeline, 0, 8400, 0, 1));
        assignBestIndices.SetResult("dst", bestIndices);
        
        nmsBoxesGlobal = provider.CreateTensor<float, Matrix>(
            1, new TensorShape(new[] { numberOfObjects, 4 }));
        nmsScoresGlobal = provider.CreateTensor<float, Matrix>(
            1, new TensorShape(new[] { numberOfObjects, 1 }));
        
        nmsBoxesPlaceholder = inferencePipeline.CreateTensorReference<float, Matrix>(
            1, new TensorShape(new[] { numberOfObjects, 4 }));
        nmsScoresPlaceholder = inferencePipeline.CreateTensorReference<float, Matrix>(
            1, new TensorShape(new[] { numberOfObjects, 1 }));
        
        var nmsIndices = inferencePipeline.CreateTensor<int, Matrix>(
            1, new TensorShape(new[] { numberOfObjects, 1 }));
        
        var nmsOp = inferencePipeline.CreateOperator<NonMaximumSuppressionOperator>();
        nmsOp.SetOperand("scores", bestScores);
        nmsOp.SetOperand("boxes", boxes);
        nmsOp.SetOperand("iou threshold", CreateScalarTensor(inferencePipeline, nmsThreshold));
        nmsOp.SetResult("output scores", nmsScoresPlaceholder);
        nmsOp.SetResult("output boxes", nmsBoxesPlaceholder);
        nmsOp.SetResult("output indices", nmsIndices);
        
        classesSelectGlobal = provider.CreateTensor<float, Matrix>(
            1, new TensorShape(new[] { numberOfObjects, 1 }));
        classesSelectPlaceholder = inferencePipeline.CreateTensorReference<float, Matrix>(
            1, new TensorShape(new[] { numberOfObjects, 1 }));
        
        CopyTensorBySlice(inferencePipeline, bestIndices, classesSelectPlaceholder, nmsIndices, numberOfObjects);
        
        LogDebug("YOLO inference pipeline created!");
    }

    private void RunModelInferencePipeline()
    {
        var tensorMapping = new TensorMapping();
        tensorMapping.Set(vstImagePlaceholder, vstOutputLeftFp32Global);
        tensorMapping.Set(nmsBoxesPlaceholder, nmsBoxesGlobal);
        tensorMapping.Set(nmsScoresPlaceholder, nmsScoresGlobal);
        tensorMapping.Set(classesSelectPlaceholder, classesSelectGlobal);
        
        inferencePipeline.Execute(tensorMapping);

        // Extract detection results for validation
        ExtractDetectionResults();
    }

    private void ExtractDetectionResults()
    {
        try
        {
            // Read tensor data (this is safe as these are output tensors)
            nmsBoxesGlobal.GetData(boxesData);
            nmsScoresGlobal.GetData(scoresData);
            classesSelectGlobal.GetData(classesData);

            var detections = new List<DetectedObject>();

            for (int i = 0; i < numberOfObjects; i++)
            {
                float confidence = scoresData[i];
                if (confidence < confidenceThreshold)
                    continue;

                int classId = Mathf.RoundToInt(classesData[i]);
                
                // Only process training objects
                if (!trainingObjectMap.ContainsKey(classId))
                    continue;

                float xmin = boxesData[i * 4 + 0];
                float ymin = boxesData[i * 4 + 1];
                float xmax = boxesData[i * 4 + 2];
                float ymax = boxesData[i * 4 + 3];

                float centerX = (xmin + xmax) / 2.0f;
                float centerY = (ymin + ymax) / 2.0f;
                float width = xmax - xmin;
                float height = ymax - ymin;

                // Normalize to 0-1 range
                centerX /= vstWidth;
                centerY /= vstHeight;
                width /= vstWidth;
                height /= vstHeight;

                string zone = DetermineZone(centerX, centerY);
                
                var detection = new DetectedObject
                {
                    classId = classId,
                    confidence = confidence,
                    centerX = centerX,
                    centerY = centerY,
                    width = width,
                    height = height,
                    objectName = trainingObjectMap[classId],
                    zone = zone
                };

                detections.Add(detection);

                if (logDetections)
                {
                    LogDebug($"Detected: {detection}");
                }
            }

            lock (detectionLock)
            {
                latestDetections = detections;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[TrainingDemo] Failed to extract detections: {e.Message}");
        }
    }

    #endregion

    #region Map 2D to 3D Pipeline

    private void CreateMap2dTo3dPipeline()
    {
        LogDebug("Creating Map2D to 3D pipeline...");
        
        map2dTo3dPipeline = provider.CreatePipeline();
        
        nmsBoxesPlaceholder1 = map2dTo3dPipeline.CreateTensorReference<float, Matrix>(
            1, new TensorShape(new[] { numberOfObjects, 4 }));
        timestampPlaceholder1 = map2dTo3dPipeline.CreateTensorReference<int, TimeStamp>(
            4, new TensorShape(new[] { 1 }));
        cameraMatrixPlaceholder1 = map2dTo3dPipeline.CreateTensorReference<float, Matrix>(
            1, new TensorShape(new[] { 3, 3 }));
        leftImagePlaceholder = map2dTo3dPipeline.CreateTensorReference<byte, Matrix>(
            3, new TensorShape(new[] { vstHeight, vstWidth }));
        rightImagePlaceholder = map2dTo3dPipeline.CreateTensorReference<byte, Matrix>(
            3, new TensorShape(new[] { vstHeight, vstWidth }));
        
        var imagePoint = map2dTo3dPipeline.CreateTensor<int, Point>(
            2, new TensorShape(new[] { numberOfObjects }));
        
        var xminymin = map2dTo3dPipeline.CreateTensor<float, Matrix>(
            1, new TensorShape(new[] { numberOfObjects, 2 }));
        var xmaxymax = map2dTo3dPipeline.CreateTensor<float, Matrix>(
            1, new TensorShape(new[] { numberOfObjects, 2 }));
        var imagePointMat = map2dTo3dPipeline.CreateTensor<float, Matrix>(
            1, new TensorShape(new[] { numberOfObjects, 2 }));
        
        var assignMin = map2dTo3dPipeline.CreateOperator<AssignmentOperator>();
        assignMin.SetOperand("src", nmsBoxesPlaceholder1);
        assignMin.SetOperand("src slices", CreateSliceTensor(map2dTo3dPipeline, 0, numberOfObjects, 0, 2));
        assignMin.SetResult("dst", xminymin);
        
        var assignMax = map2dTo3dPipeline.CreateOperator<AssignmentOperator>();
        assignMax.SetOperand("src", nmsBoxesPlaceholder1);
        assignMax.SetOperand("src slices", CreateSliceTensor(map2dTo3dPipeline, 0, numberOfObjects, 2, 4));
        assignMax.SetResult("dst", xmaxymax);
        
        var calcCenter = map2dTo3dPipeline.CreateOperator<ArithmeticComposeOperator>(
            new ArithmeticComposeOperatorConfiguration("{0} * 0.5 + {1} * 0.5"));
        calcCenter.SetOperand("{0}", xminymin);
        calcCenter.SetOperand("{1}", xmaxymax);
        calcCenter.SetResult("result", imagePointMat);
        
        var assignPoint = map2dTo3dPipeline.CreateOperator<AssignmentOperator>();
        assignPoint.SetOperand("src", imagePointMat);
        assignPoint.SetResult("dst", imagePoint);
        
        pointXYZGlobal = provider.CreateTensor<float, Point>(
            3, new TensorShape(new[] { numberOfObjects }));
        pointXYZPlaceholder = map2dTo3dPipeline.CreateTensorReference<float, Point>(
            3, new TensorShape(new[] { numberOfObjects }));
        
        var uv2CamOp = map2dTo3dPipeline.CreateOperator<UvTo3DInCameraSpaceOperator>();
        uv2CamOp.SetOperand("uv", imagePoint);
        uv2CamOp.SetOperand("timestamp", timestampPlaceholder1);
        uv2CamOp.SetOperand("camera intrinsic", cameraMatrixPlaceholder1);
        uv2CamOp.SetOperand("left image", leftImagePlaceholder);
        uv2CamOp.SetOperand("right image", rightImagePlaceholder);
        uv2CamOp.SetResult("point_xyz", pointXYZPlaceholder);
        
        scaleGlobal = provider.CreateTensor<float, Matrix>(
            1, new TensorShape(new[] { numberOfObjects, 3 }));
        scalePlaceholder = map2dTo3dPipeline.CreateTensorReference<float, Matrix>(
            1, new TensorShape(new[] { numberOfObjects, 3 }));
        
        var ratio = map2dTo3dPipeline.CreateTensor<float, Matrix>(
            1, new TensorShape(new[] { numberOfObjects, 2 }));
        var divider = CreateConstantTensor(map2dTo3dPipeline, numberOfObjects, 2, 2000.0f);
        
        var calcRatio = map2dTo3dPipeline.CreateOperator<ArithmeticComposeOperator>(
            new ArithmeticComposeOperatorConfiguration("{0} - {1}"));
        calcRatio.SetOperand("{0}", xmaxymax);
        calcRatio.SetOperand("{1}", xminymin);
        calcRatio.SetResult("result", ratio);
        
        var divideRatio = map2dTo3dPipeline.CreateOperator<ArithmeticComposeOperator>(
            new ArithmeticComposeOperatorConfiguration("{0} / {1}"));
        divideRatio.SetOperand("{0}", ratio);
        divideRatio.SetOperand("{1}", divider);
        divideRatio.SetResult("result", ratio);
        
        var assignScale = map2dTo3dPipeline.CreateOperator<AssignmentOperator>();
        assignScale.SetOperand("src", ratio);
        assignScale.SetOperand("dst slices", CreateSliceTensor(map2dTo3dPipeline, 0, numberOfObjects, 0, 2));
        assignScale.SetResult("dst", scalePlaceholder);
        
        var zScale = CreateConstantTensor(map2dTo3dPipeline, numberOfObjects, 1, 0.05f);
        var assignZScale = map2dTo3dPipeline.CreateOperator<AssignmentOperator>();
        assignZScale.SetOperand("src", zScale);
        assignZScale.SetOperand("dst slices", CreateSliceTensor(map2dTo3dPipeline, 0, numberOfObjects, 2, 3));
        assignZScale.SetResult("dst", scalePlaceholder);
        
        LogDebug("Map2D to 3D pipeline created!");
    }

    private void RunMap2dTo3dPipeline()
    {
        var tensorMapping = new TensorMapping();
        tensorMapping.Set(nmsBoxesPlaceholder1, nmsBoxesGlobal);
        tensorMapping.Set(timestampPlaceholder1, vstTimestampGlobal);
        tensorMapping.Set(cameraMatrixPlaceholder1, vstCameraMatrixGlobal);
        tensorMapping.Set(leftImagePlaceholder, vstOutputLeftUint8Global);
        tensorMapping.Set(rightImagePlaceholder, vstOutputRightUint8Global);
        tensorMapping.Set(pointXYZPlaceholder, pointXYZGlobal);
        tensorMapping.Set(scalePlaceholder, scaleGlobal);
        
        map2dTo3dPipeline.Execute(tensorMapping);
    }

    #endregion

    #region Rendering Pipeline

    private void CreateRenderingPipeline()
    {
        LogDebug("Creating rendering pipeline...");
        
        renderPipeline = provider.CreatePipeline();
        
        pointXYZPlaceholder1 = renderPipeline.CreateTensorReference<float, Point>(
            3, new TensorShape(new[] { numberOfObjects }));
        timestampPlaceholder2 = renderPipeline.CreateTensorReference<int, TimeStamp>(
            4, new TensorShape(new[] { 1 }));
        classesSelectPlaceholder1 = renderPipeline.CreateTensorReference<float, Matrix>(
            1, new TensorShape(new[] { numberOfObjects, 1 }));
        nmsScoresPlaceholder1 = renderPipeline.CreateTensorReference<float, Matrix>(
            1, new TensorShape(new[] { numberOfObjects, 1 }));
        scalePlaceholder1 = renderPipeline.CreateTensorReference<float, Matrix>(
            1, new TensorShape(new[] { numberOfObjects, 3 }));
        
        var classesSelectInt = renderPipeline.CreateTensor<int, Matrix>(
            1, new TensorShape(new[] { numberOfObjects, 1 }));
        var assignClassInt = renderPipeline.CreateOperator<AssignmentOperator>();
        assignClassInt.SetOperand("src", classesSelectPlaceholder1);
        assignClassInt.SetResult("dst", classesSelectInt);
        
        var textArrayTensor = renderPipeline.CreateTensor<byte, Scalar>(
            1, new TensorShape(new[] { 80, 13 }));
        CopyTextArray(renderPipeline, classNames, textArrayTensor, 80);
        
        var textToPrintTensor = renderPipeline.CreateTensor<byte, Scalar>(
            1, new TensorShape(new[] { numberOfObjects, 13 }));
        CopyTensorBySlice(renderPipeline, textArrayTensor, textToPrintTensor, classesSelectInt, numberOfObjects);
        
        gltfAsset = provider.CreateTensor<Gltf>(labelGltfAsset.bytes);
        gltfAsset1 = provider.CreateTensor<Gltf>(labelGltfAsset.bytes);
        gltfAsset2 = provider.CreateTensor<Gltf>(labelGltfAsset.bytes);
        gltfAsset3 = provider.CreateTensor<Gltf>(labelGltfAsset.bytes);
        
        gltfPlaceholderTensor = renderPipeline.CreateTensorReference<Gltf>();
        gltfPlaceholderTensor1 = renderPipeline.CreateTensorReference<Gltf>();
        gltfPlaceholderTensor2 = renderPipeline.CreateTensorReference<Gltf>();
        gltfPlaceholderTensor3 = renderPipeline.CreateTensorReference<Gltf>();
        
        var textArray0 = renderPipeline.CreateTensor<byte, Scalar>(1, new TensorShape(new[] { 1, 13 }));
        var textArray1 = renderPipeline.CreateTensor<byte, Scalar>(1, new TensorShape(new[] { 1, 13 }));
        var textArray2 = renderPipeline.CreateTensor<byte, Scalar>(1, new TensorShape(new[] { 1, 13 }));
        var textArray3 = renderPipeline.CreateTensor<byte, Scalar>(1, new TensorShape(new[] { 1, 13 }));
        
        var pointXYZ0 = renderPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 1, 3 }));
        var pointXYZ1 = renderPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 1, 3 }));
        var pointXYZ2 = renderPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 1, 3 }));
        var pointXYZ3 = renderPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 1, 3 }));
        
        var scale0 = renderPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 3, 1 }));
        var scale1 = renderPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 3, 1 }));
        var scale2 = renderPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 3, 1 }));
        var scale3 = renderPipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 3, 1 }));
        
        var score0 = renderPipeline.CreateTensor<float, Scalar>(1, new TensorShape(new[] { 1 }));
        var score1 = renderPipeline.CreateTensor<float, Scalar>(1, new TensorShape(new[] { 1 }));
        var score2 = renderPipeline.CreateTensor<float, Scalar>(1, new TensorShape(new[] { 1 }));
        var score3 = renderPipeline.CreateTensor<float, Scalar>(1, new TensorShape(new[] { 1 }));
        
        // Extract per-object data
        ExtractObjectData(renderPipeline, textToPrintTensor, pointXYZPlaceholder1, scalePlaceholder1, nmsScoresPlaceholder1,
            0, textArray0, pointXYZ0, scale0, score0);
        ExtractObjectData(renderPipeline, textToPrintTensor, pointXYZPlaceholder1, scalePlaceholder1, nmsScoresPlaceholder1,
            1, textArray1, pointXYZ1, scale1, score1);
        ExtractObjectData(renderPipeline, textToPrintTensor, pointXYZPlaceholder1, scalePlaceholder1, nmsScoresPlaceholder1,
            2, textArray2, pointXYZ2, scale2, score2);
        ExtractObjectData(renderPipeline, textToPrintTensor, pointXYZPlaceholder1, scalePlaceholder1, nmsScoresPlaceholder1,
            3, textArray3, pointXYZ3, scale3, score3);
        
        RenderText(renderPipeline, textArray0, pointXYZ0, gltfPlaceholderTensor, scale0, score0, timestampPlaceholder2);
        RenderText(renderPipeline, textArray1, pointXYZ1, gltfPlaceholderTensor1, scale1, score1, timestampPlaceholder2);
        RenderText(renderPipeline, textArray2, pointXYZ2, gltfPlaceholderTensor2, scale2, score2, timestampPlaceholder2);
        RenderText(renderPipeline, textArray3, pointXYZ3, gltfPlaceholderTensor3, scale3, score3, timestampPlaceholder2);
        
        LogDebug("Rendering pipeline created!");
    }

    private void ExtractObjectData(Pipeline pipeline, Tensor textSource, Tensor pointSource, 
        Tensor scaleSource, Tensor scoreSource, int index, Tensor textDst, Tensor pointDst, 
        Tensor scaleDst, Tensor scoreDst)
    {
        var assignText = pipeline.CreateOperator<AssignmentOperator>();
        assignText.SetOperand("src", textSource);
        assignText.SetOperand("src slices", CreateSliceTensor(pipeline, index, index + 1, 0, 13));
        assignText.SetResult("dst", textDst);
        
        var assignPoint = pipeline.CreateOperator<AssignmentOperator>();
        assignPoint.SetOperand("src", pointSource);
        assignPoint.SetOperand("src slices", CreateSliceTensor(pipeline, index, index + 1, 0, 3));
        assignPoint.SetResult("dst", pointDst);
        
        var assignScale = pipeline.CreateOperator<AssignmentOperator>();
        assignScale.SetOperand("src", scaleSource);
        assignScale.SetOperand("src slices", CreateSliceTensor(pipeline, index, index + 1, 0, 3));
        assignScale.SetResult("dst", scaleDst);
        
        var assignScore = pipeline.CreateOperator<AssignmentOperator>();
        assignScore.SetOperand("src", scoreSource);
        assignScore.SetOperand("src slices", CreateSliceTensor(pipeline, index, index + 1, 0, 1));
        assignScore.SetResult("dst", scoreDst);
    }

    private void RenderText(Pipeline pipeline, Tensor textTensor, Tensor pointXYZ, Tensor gltfPlaceholder,
                           Tensor scale, Tensor score, Tensor timestamp)
    {
        var rvec = CreateConstantTensor(pipeline, 3, 1, 0.0f);
        var leftEyeTransform = pipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 4, 4 }));
        var currentPosition = pipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 4, 4 }));
        
        var multiplier = pipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 1, 3 }));
        multiplier.Reset(new float[] { 1.0f, -1.0f, 1.0f });
        
        var pointXYZAdj = pipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 1, 3 }));
        
        var multOp = pipeline.CreateOperator<ElementwiseMultiplyOperator>();
        multOp.SetOperand("operand0", pointXYZ);
        multOp.SetOperand("operand1", multiplier);
        multOp.SetResult("result", pointXYZAdj);
        
        var offset = pipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 1, 3 }));
        offset.Reset(new float[] { 0.1f, 0.0f, 0.0f });
        
        var addOp = pipeline.CreateOperator<ArithmeticComposeOperator>(
            new ArithmeticComposeOperatorConfiguration("({0} + {1})"));
        addOp.SetOperand("{0}", pointXYZAdj);
        addOp.SetOperand("{1}", offset);
        addOp.SetResult("result", pointXYZAdj);
        
        var transformOp = pipeline.CreateOperator<GetTransformMatrixOperator>();
        transformOp.SetOperand("rotation", rvec);
        transformOp.SetOperand("translation", pointXYZAdj);
        transformOp.SetOperand("scale", scale);
        transformOp.SetResult("result", currentPosition);
        
        var cam2WorldOp = pipeline.CreateOperator<CameraSpaceToWorldOperator>();
        cam2WorldOp.SetOperand("timestamp", timestamp);
        cam2WorldOp.SetResult("left", leftEyeTransform);
        
        var finalTransform = pipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { 4, 4 }));
        var finalMulOp = pipeline.CreateOperator<ArithmeticComposeOperator>(
            new ArithmeticComposeOperatorConfiguration("({0} * {1})"));
        finalMulOp.SetOperand("{0}", leftEyeTransform);
        finalMulOp.SetOperand("{1}", currentPosition);
        finalMulOp.SetResult("result", finalTransform);
        
        var startPosition = pipeline.CreateTensor<float, Point>(2, new TensorShape(new[] { 1 }));
        startPosition.Reset(new float[] { 0.1f, 0.5f });
        
        var colors = pipeline.CreateTensor<byte, Color>(4, new TensorShape(new[] { 2 }));
        colors.Reset(new byte[] { 255, 255, 255, 255, 128, 128, 128, 128 });
        
        var textureId = pipeline.CreateTensor<ushort, Scalar>(1, new TensorShape(new[] { 1 }));
        textureId.Reset(new ushort[] { 0 });
        
        var fontSize = pipeline.CreateTensor<float, Scalar>(1, new TensorShape(new[] { 1 }));
        fontSize.Reset(new float[] { 255.0f });
        
        var renderTextOp = pipeline.CreateOperator<RenderTextOperator>(
            new RenderTextOperatorConfiguration(SecureMRFontTypeface.SansSerif, "en-US", 1024, 1024));
        renderTextOp.SetOperand("text", textTensor);
        renderTextOp.SetOperand("start", startPosition);
        renderTextOp.SetOperand("colors", colors);
        renderTextOp.SetOperand("texture ID", textureId);
        renderTextOp.SetOperand("font size", fontSize);
        renderTextOp.SetOperand("gltf", gltfPlaceholder);
        
        var isVisible = pipeline.CreateTensor<int, Scalar>(1, new TensorShape(new[] { 1 }));
        var threshold = pipeline.CreateTensor<float, Scalar>(1, new TensorShape(new[] { 1 }));
        threshold.Reset(new float[] { confidenceThreshold });
        
        var compareOp = pipeline.CreateOperator<CompareToOperator>();
        compareOp.SetOperand("operand", score);
        compareOp.SetOperand("threshold", threshold);
        compareOp.SetOperand("comparison", ">");
        compareOp.SetResult("result", isVisible);
        
        var renderGltfOp = pipeline.CreateOperator<SwitchGltfRenderStatusOperator>();
        renderGltfOp.SetOperand("gltf", gltfPlaceholder);
        renderGltfOp.SetOperand("world pose", finalTransform);
        renderGltfOp.SetOperand("visible", isVisible);
    }

    private void RunRenderingPipeline()
    {
        var tensorMapping = new TensorMapping();
        tensorMapping.Set(gltfPlaceholderTensor, gltfAsset);
        tensorMapping.Set(gltfPlaceholderTensor1, gltfAsset1);
        tensorMapping.Set(gltfPlaceholderTensor2, gltfAsset2);
        tensorMapping.Set(gltfPlaceholderTensor3, gltfAsset3);
        tensorMapping.Set(pointXYZPlaceholder1, pointXYZGlobal);
        tensorMapping.Set(timestampPlaceholder2, vstTimestampGlobal);
        tensorMapping.Set(classesSelectPlaceholder1, classesSelectGlobal);
        tensorMapping.Set(nmsScoresPlaceholder1, nmsScoresGlobal);
        tensorMapping.Set(scalePlaceholder1, scaleGlobal);
        
        renderPipeline.Execute(tensorMapping);
    }

    #endregion

    #region Zone Detection and Validation

    private void ProcessLatestDetections()
    {
        List<DetectedObject> detections;
        lock (detectionLock)
        {
            detections = new List<DetectedObject>(latestDetections);
        }

        // Build configuration from detections
        var config = BuildConfigurationFromDetections(detections);

        // Update current detected config
        currentDetectedConfig = config;

        // Check stability
        if (lastStableConfig != null && lastStableConfig.Matches(config))
        {
            stableFrameCount++;
        }
        else
        {
            stableFrameCount = 0;
            lastStableConfig = config;
        }

        // Validate if stable
        if (stableFrameCount >= stabilityFrames)
        {
            ValidateConfiguration(config);
        }
    }

    private ExpectedConfig BuildConfigurationFromDetections(List<DetectedObject> detections)
    {
        var config = new ExpectedConfig();

        foreach (var detection in detections)
        {
            if (string.IsNullOrEmpty(detection.zone))
                continue;

            // Assign first detection per zone (ignore duplicates)
            switch (detection.zone.ToLower())
            {
                case "left":
                    if (string.IsNullOrEmpty(config.left))
                        config.left = detection.objectName;
                    break;
                case "right":
                    if (string.IsNullOrEmpty(config.right))
                        config.right = detection.objectName;
                    break;
                case "top":
                    if (string.IsNullOrEmpty(config.top))
                        config.top = detection.objectName;
                    break;
                case "bottom":
                    if (string.IsNullOrEmpty(config.bottom))
                        config.bottom = detection.objectName;
                    break;
            }
        }

        return config;
    }

    private string DetermineZone(float normalizedX, float normalizedY)
    {
        // Check horizontal zones first (LEFT/RIGHT have priority)
        if (normalizedX < leftZoneX)
            return "left";
        
        if (normalizedX > rightZoneX)
            return "right";

        // Check vertical zones (TOP/BOTTOM)
        if (normalizedY < topZoneY)
            return "top";
        
        if (normalizedY > bottomZoneY)
            return "bottom";

        // Object in center/undefined zone
        return null;
    }

    private void ValidateConfiguration(ExpectedConfig detected)
    {
        // Throttle validation to avoid spam
        if (Time.time - lastValidationTime < 0.5f)
            return;

        lastValidationTime = Time.time;

        bool isValid = stepManager.ValidateConfiguration(detected);
        
        if (isValid != lastValidationResult)
        {
            lastValidationResult = isValid;
            
            if (isValid)
            {
                LogDebug($"✓ Configuration CORRECT: {detected}");
            }
            else
            {
                LogDebug($"✗ Configuration INCORRECT: {detected}");
                LogDebug($"  Expected: {stepManager.CurrentStep.expected_config}");
            }
        }
    }

    #endregion

    #region Pipeline Runners

    private void StartPipelineRunners()
    {
        LogDebug("Starting pipeline runners...");
        
        vstRunnerThread = new Thread(() =>
        {
            while (keepRunning)
            {
                try
                {
                    RunVSTImagePipeline();
                    Thread.Sleep(33); // ~30 FPS
                }
                catch (Exception e)
                {
                    Debug.LogError($"[TrainingDemo] VST error: {e.Message}");
                }
            }
        });
        vstRunnerThread.Start();
        
        inferenceRunnerThread = new Thread(() =>
        {
            while (keepRunning)
            {
                try
                {
                    lock (initLock)
                    {
                        if (!pipelinesInitialized) continue;
                    }
                    RunModelInferencePipeline();
                    Thread.Sleep(200); // 5 FPS
                }
                catch (Exception e)
                {
                    Debug.LogError($"[TrainingDemo] Inference error: {e.Message}");
                }
            }
        });
        inferenceRunnerThread.Start();
        
        map2dRunnerThread = new Thread(() =>
        {
            while (keepRunning)
            {
                try
                {
                    lock (initLock)
                    {
                        if (!pipelinesInitialized) continue;
                    }
                    RunMap2dTo3dPipeline();
                    Thread.Sleep(200);
                }
                catch (Exception e)
                {
                    Debug.LogError($"[TrainingDemo] Map2D3D error: {e.Message}");
                }
            }
        });
        map2dRunnerThread.Start();
        
        renderRunnerThread = new Thread(() =>
        {
            while (keepRunning)
            {
                try
                {
                    lock (initLock)
                    {
                        if (!pipelinesInitialized) continue;
                    }
                    RunRenderingPipeline();
                    Thread.Sleep(200);
                }
                catch (Exception e)
                {
                    Debug.LogError($"[TrainingDemo] Render error: {e.Message}");
                }
            }
        });
        renderRunnerThread.Start();
        
        LogDebug("All pipeline runners started!");
    }

    #endregion

    #region Helper Methods

    private Tensor CreateSliceTensor(Pipeline pipeline, int startRow, int endRow, int startCol, int endCol)
    {
        var slice = pipeline.CreateTensor<int, Slice>(2, new TensorShape(new[] { 2 }));
        slice.Reset(new int[] { startRow, endRow, startCol, endCol });
        return slice;
    }

    private Tensor CreateScalarTensor(Pipeline pipeline, float value)
    {
        var tensor = pipeline.CreateTensor<float, Scalar>(1, new TensorShape(new[] { 1 }));
        tensor.Reset(new float[] { value });
        return tensor;
    }

    private Tensor CreateConstantTensor(Pipeline pipeline, int rows, int cols, float value)
    {
        var tensor = pipeline.CreateTensor<float, Matrix>(1, new TensorShape(new[] { rows, cols }));
        float[] data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
            data[i] = value;
        tensor.Reset(data);
        return tensor;
    }

    private void CopyTensorBySlice(Pipeline pipeline, Tensor src, Tensor dst, Tensor indices, int size)
    {
        var indicesPlusOne = pipeline.CreateTensor<int, Matrix>(1, new TensorShape(new[] { size, 1 }));
        
        var addOp = pipeline.CreateOperator<ArithmeticComposeOperator>(
            new ArithmeticComposeOperatorConfiguration("({0} + 1)"));
        addOp.SetOperand("{0}", indices);
        addOp.SetResult("result", indicesPlusOne);
        
        for (int i = 0; i < size; i++)
        {
            var srcSlice = pipeline.CreateTensor<int, Slice>(2, new TensorShape(new[] { 2 }));
            
            var assignStart = pipeline.CreateOperator<AssignmentOperator>();
            assignStart.SetOperand("src", indices);
            assignStart.SetOperand("src slices", CreateSliceTensor(pipeline, i, i + 1, 0, 1));
            assignStart.SetOperand("dst slices", CreateSliceTensor(pipeline, 0, 1, 0, 1));
            assignStart.SetResult("dst", srcSlice);
            
            var assignEnd = pipeline.CreateOperator<AssignmentOperator>();
            assignEnd.SetOperand("src", indicesPlusOne);
            assignEnd.SetOperand("src slices", CreateSliceTensor(pipeline, i, i + 1, 0, 1));
            assignEnd.SetOperand("dst slices", CreateSliceTensor(pipeline, 0, 1, 1, 2));
            assignEnd.SetResult("dst", srcSlice);
            
            var copyOp = pipeline.CreateOperator<AssignmentOperator>();
            copyOp.SetOperand("src", src);
            copyOp.SetOperand("src slices", srcSlice);
            copyOp.SetOperand("dst slices", CreateSliceTensor(pipeline, i, i + 1, 0, -1));
            copyOp.SetResult("dst", dst);
        }
    }

    private void CopyTextArray(Pipeline pipeline, string[] textArray, Tensor dstTensor, int maxClasses)
    {
        int textLength = 13;
        
        for (int i = 0; i < Math.Min(textArray.Length, maxClasses); i++)
        {
            string text = textArray[i];
            
            if (text.Length > textLength)
                text = text.Substring(0, textLength);
            else if (text.Length < textLength)
                text = text.PadRight(textLength, ' ');
            
            var textTensor = pipeline.CreateTensor<byte, Scalar>(1, new TensorShape(new[] { textLength }));
            byte[] bytes = System.Text.Encoding.UTF8.GetBytes(text);
            textTensor.Reset(bytes);
            
            var assignOp = pipeline.CreateOperator<AssignmentOperator>();
            assignOp.SetOperand("src", textTensor);
            assignOp.SetOperand("dst slices", CreateSliceTensor(pipeline, i, i + 1, 0, textLength));
            assignOp.SetResult("dst", dstTensor);
        }
    }

    #endregion

    #region Event Handlers

    private void OnStepChanged(Step newStep)
    {
        LogDebug($"=== Step Changed: {newStep.GetSummary()} ===");
        
        // Reset validation state
        stableFrameCount = 0;
        lastStableConfig = null;
        lastValidationResult = false;
        lastValidationTime = 0f;
    }

    private void OnConfigValidated(bool isValid)
    {
        // Optional: Add visual/audio feedback
        // You could trigger haptics, change UI colors, play sounds, etc.
    }

    #endregion

    #region Debug & Visualization

    private void LogDebug(string message)
    {
        if (debugLogging)
        {
            Debug.Log($"[TrainingDemo] {message}");
        }
    }

#if UNITY_EDITOR
    private void DrawZoneBoundaries()
    {
        Vector3 center = transform.position + Vector3.forward * 2f;
        float size = 2f;

        // LEFT zone (red)
        Gizmos.color = Color.red;
        Vector3 leftCenter = center + Vector3.left * size * 0.33f;
        Gizmos.DrawWireCube(leftCenter, Vector3.one * 0.5f);
        UnityEditor.Handles.Label(leftCenter, "LEFT\n(bottle)");

        // RIGHT zone (blue)
        Gizmos.color = Color.blue;
        Vector3 rightCenter = center + Vector3.right * size * 0.33f;
        Gizmos.DrawWireCube(rightCenter, Vector3.one * 0.5f);
        UnityEditor.Handles.Label(rightCenter, "RIGHT\n(cup)");

        // TOP zone (green)
        Gizmos.color = Color.green;
        Vector3 topCenter = center + Vector3.up * size * 0.33f;
        Gizmos.DrawWireCube(topCenter, Vector3.one * 0.5f);
        UnityEditor.Handles.Label(topCenter, "TOP\n(scissors)");

        // BOTTOM zone (yellow)
        Gizmos.color = Color.yellow;
        Vector3 bottomCenter = center + Vector3.down * size * 0.33f;
        Gizmos.DrawWireCube(bottomCenter, Vector3.one * 0.5f);
        UnityEditor.Handles.Label(bottomCenter, "BOTTOM\n(book)");
    }
#endif

    #endregion
}