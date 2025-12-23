using UnityEngine;
using SecureFab.Training;

/// <summary>
/// Example integration script showing how to connect StepManager with SecureMR.
/// This bridges Unity C# with the SecureMR YOLO pipeline for object detection.
/// 
/// IMPORTANT: This is a TEMPLATE showing the integration pattern.
/// You'll need to replace pseudo-code sections with actual SecureMR API calls.
/// </summary>
public class SecureMRIntegration : MonoBehaviour
{
    #region Inspector Fields

    [Header("References")]
    [SerializeField] 
    [Tooltip("Reference to the StepManager in the scene")]
    private StepManager stepManager;

    [Header("SecureMR Settings")]
    [Tooltip("How often to poll YOLO detection results (seconds)")]
    [Range(0.1f, 2.0f)]
    public float detectionPollInterval = 0.5f;

    [Tooltip("Confidence threshold for YOLO detections (0-1)")]
    [Range(0.0f, 1.0f)]
    public float confidenceThreshold = 0.5f;

    [Header("Zone Mapping")]
    [Tooltip("X threshold for LEFT zone (normalized 0-1)")]
    [Range(0.0f, 0.5f)]
    public float leftZoneThresholdX = 0.33f;

    [Tooltip("X threshold for RIGHT zone (normalized 0-1)")]
    [Range(0.5f, 1.0f)]
    public float rightZoneThresholdX = 0.66f;

    [Tooltip("Y threshold for TOP zone (normalized 0-1)")]
    [Range(0.0f, 0.5f)]
    public float topZoneThresholdY = 0.33f;

    [Tooltip("Y threshold for BOTTOM zone (normalized 0-1)")]
    [Range(0.5f, 1.0f)]
    public float bottomZoneThresholdY = 0.66f;

    [Header("Visual Feedback")]
    [SerializeField]
    [Tooltip("Material to apply when config is correct")]
    private Material validMaterial;

    [SerializeField]
    [Tooltip("Material to apply when config is incorrect")]
    private Material invalidMaterial;

    #endregion

    #region Private Fields

    private float _pollTimer = 0f;
    private ExpectedConfig _lastDetectedConfig;
    private bool _lastValidationResult = false;

    // Class ID to object name mapping (from YOLO model)
    private readonly string[] _classNames = new string[]
    {
        "bottle",   // class 0
        "cup",      // class 1
        "scissors", // class 2
        "book"      // class 3
    };

    #endregion

    #region Unity Lifecycle

    private void Start()
    {
        if (stepManager == null)
        {
            Debug.LogError("[SecureMRIntegration] StepManager reference is missing!");
            enabled = false;
            return;
        }

        // Subscribe to step changes to update SecureMR instruction text
        stepManager.onStepChanged.AddListener(OnStepChanged);
        stepManager.onConfigurationValidated.AddListener(OnConfigValidated);

        // Initialize with current step
        if (stepManager.IsInitialized)
        {
            OnStepChanged(stepManager.CurrentStep);
        }

        Debug.Log("[SecureMRIntegration] Initialized successfully");
    }

    private void Update()
    {
        // Poll YOLO detections at specified interval
        _pollTimer += Time.deltaTime;
        if (_pollTimer >= detectionPollInterval)
        {
            _pollTimer = 0f;
            PollYOLODetections();
        }
    }

    #endregion

    #region SecureMR Integration

    /// <summary>
    /// Called when the current step changes.
    /// Updates the instruction text shown in SecureMR's GLTF panel.
    /// </summary>
    private void OnStepChanged(Step newStep)
    {
        if (newStep == null) return;

        // Format instruction string
        string instruction = FormatInstructionText(newStep);

        // PSEUDO-CODE: Update SecureMR text tensor
        // You need to replace this with actual SecureMR API calls
        UpdateSecureMRInstructionText(instruction);

        Debug.Log($"[SecureMRIntegration] Updated instructions for {newStep.title}");
    }

    /// <summary>
    /// Format step information for display in MR.
    /// </summary>
    private string FormatInstructionText(Step step)
    {
        // Simple format: Title on first line, body on following lines
        return $"{step.title}\n\n{step.body}";
    }

    /// <summary>
    /// PSEUDO-CODE: Update the text displayed in SecureMR.
    /// Replace this with actual SecureMR API calls to write to the text tensor.
    /// </summary>
    private void UpdateSecureMRInstructionText(string text)
    {
        // EXAMPLE PSEUDO-CODE - Replace with actual implementation:
        //
        // if (SecureMRManager.Instance != null)
        // {
        //     SecureMRManager.Instance.SetTextTensor("InstructionPanel", text);
        //     SecureMRManager.Instance.RequestPipelineUpdate();
        // }

        Debug.Log($"[SecureMRIntegration] Would update SecureMR text to:\n{text}");
    }

    #endregion

    #region YOLO Detection Processing

    /// <summary>
    /// Poll YOLO detection results and map to zone configuration.
    /// This is called every detectionPollInterval seconds.
    /// </summary>
    private void PollYOLODetections()
    {
        // PSEUDO-CODE: Get YOLO detections from SecureMR
        // You need to replace this with actual SecureMR API calls
        var detections = GetYOLODetections();

        if (detections == null || detections.Length == 0)
        {
            // No objects detected - all zones empty
            UpdateDetectedConfiguration(new ExpectedConfig());
            return;
        }

        // Map detections to zones
        ExpectedConfig detectedConfig = MapDetectionsToZones(detections);
        
        // Update and validate
        UpdateDetectedConfiguration(detectedConfig);
    }

    /// <summary>
    /// PSEUDO-CODE: Get YOLO detections from SecureMR pipeline.
    /// Replace with actual SecureMR API calls.
    /// </summary>
    private YOLODetection[] GetYOLODetections()
    {
        // EXAMPLE PSEUDO-CODE - Replace with actual implementation:
        //
        // if (SecureMRManager.Instance != null)
        // {
        //     var tensorData = SecureMRManager.Instance.GetOutputTensor("yolo_detections");
        //     return ParseYOLOTensor(tensorData);
        // }
        // return null;

        // For now, return empty array (will show all zones as empty)
        return new YOLODetection[0];
    }

    /// <summary>
    /// Map YOLO bounding box detections to the four board zones.
    /// </summary>
    private ExpectedConfig MapDetectionsToZones(YOLODetection[] detections)
    {
        ExpectedConfig config = new ExpectedConfig();

        foreach (var detection in detections)
        {
            // Skip low-confidence detections
            if (detection.confidence < confidenceThreshold)
                continue;

            // Get class name from ID
            string objectName = GetClassName(detection.classId);
            if (string.IsNullOrEmpty(objectName))
                continue;

            // Determine zone based on bounding box center
            string zone = DetermineZone(detection.centerX, detection.centerY);

            // Assign to config (first detection wins per zone)
            AssignObjectToZone(config, zone, objectName);
        }

        return config;
    }

    /// <summary>
    /// Determine which zone a point belongs to based on normalized coordinates.
    /// </summary>
    private string DetermineZone(float normalizedX, float normalizedY)
    {
        // Check horizontal zones first (LEFT/RIGHT)
        if (normalizedX < leftZoneThresholdX)
        {
            return "left";
        }
        else if (normalizedX > rightZoneThresholdX)
        {
            return "right";
        }

        // Check vertical zones (TOP/BOTTOM)
        if (normalizedY < topZoneThresholdY)
        {
            return "top";
        }
        else if (normalizedY > bottomZoneThresholdY)
        {
            return "bottom";
        }

        // Object in center/undefined zone - ignore
        return null;
    }

    /// <summary>
    /// Assign an object to a zone in the configuration.
    /// </summary>
    private void AssignObjectToZone(ExpectedConfig config, string zone, string objectName)
    {
        if (string.IsNullOrEmpty(zone)) return;

        switch (zone.ToLower())
        {
            case "left":
                if (string.IsNullOrEmpty(config.left))
                    config.left = objectName;
                break;
            case "right":
                if (string.IsNullOrEmpty(config.right))
                    config.right = objectName;
                break;
            case "top":
                if (string.IsNullOrEmpty(config.top))
                    config.top = objectName;
                break;
            case "bottom":
                if (string.IsNullOrEmpty(config.bottom))
                    config.bottom = objectName;
                break;
        }
    }

    /// <summary>
    /// Get human-readable class name from YOLO class ID.
    /// </summary>
    private string GetClassName(int classId)
    {
        if (classId >= 0 && classId < _classNames.Length)
            return _classNames[classId];
        return null;
    }

    #endregion

    #region Configuration Validation

    /// <summary>
    /// Update the detected configuration and validate against current step.
    /// </summary>
    private void UpdateDetectedConfiguration(ExpectedConfig detected)
    {
        // Check if configuration actually changed
        if (_lastDetectedConfig != null && _lastDetectedConfig.Matches(detected))
            return; // No change, skip validation

        _lastDetectedConfig = detected;

        // Validate against current step
        bool isValid = stepManager.ValidateConfiguration(detected);
        _lastValidationResult = isValid;
    }

    /// <summary>
    /// Called when configuration is validated.
    /// </summary>
    private void OnConfigValidated(bool isValid)
    {
        // Provide visual feedback
        UpdateVisualFeedback(isValid);

        if (isValid)
        {
            Debug.Log("[SecureMRIntegration] ✓ Configuration correct!");
            // Could trigger haptic feedback, sound effects, etc.
        }
        else
        {
            Debug.Log("[SecureMRIntegration] ✗ Configuration incorrect");
        }
    }

    /// <summary>
    /// Update visual feedback based on validation result.
    /// </summary>
    private void UpdateVisualFeedback(bool isValid)
    {
        // PSEUDO-CODE: Update SecureMR GLTF materials or overlays
        // Example: Change panel border color based on validation
        
        Material feedbackMaterial = isValid ? validMaterial : invalidMaterial;
        
        // You would apply this to SecureMR rendered objects
        // SecureMRManager.Instance.SetPanelMaterial(feedbackMaterial);
    }

    #endregion

    #region Helper Structures

    /// <summary>
    /// Simple structure to represent a YOLO detection result.
    /// Adjust fields to match your actual YOLO model output format.
    /// </summary>
    public struct YOLODetection
    {
        public float centerX;      // Normalized x coordinate (0-1)
        public float centerY;      // Normalized y coordinate (0-1)
        public float width;        // Normalized width (0-1)
        public float height;       // Normalized height (0-1)
        public int classId;        // Object class ID
        public float confidence;   // Detection confidence (0-1)

        public YOLODetection(float x, float y, float w, float h, int cls, float conf)
        {
            centerX = x;
            centerY = y;
            width = w;
            height = h;
            classId = cls;
            confidence = conf;
        }
    }

    #endregion

    #region Debug Helpers

#if UNITY_EDITOR
    [ContextMenu("Simulate Correct Config")]
    private void SimulateCorrectConfig()
    {
        if (stepManager.CurrentStep != null)
        {
            stepManager.ValidateConfiguration(stepManager.CurrentStep.expected_config);
        }
    }

    [ContextMenu("Simulate Wrong Config")]
    private void SimulateWrongConfig()
    {
        ExpectedConfig wrongConfig = new ExpectedConfig
        {
            left = "scissors",  // Wrong objects
            right = "book",
            top = "bottle",
            bottom = "cup"
        };
        stepManager.ValidateConfiguration(wrongConfig);
    }

    [ContextMenu("Simulate Empty Config")]
    private void SimulateEmptyConfig()
    {
        stepManager.ValidateConfiguration(new ExpectedConfig());
    }
#endif

    #endregion
}
