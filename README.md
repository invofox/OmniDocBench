# OmniDocBench Layout Benchmark Pipeline

This repository documents the full evaluation pipeline we use to benchmark layout detection models on OmniDocBench (public split) and the internal publicBench derivative. The goal of this document is to let any engineer reproduce the process end to end: from raw model outputs, through our custom evaluation metrics, to the final comparative tables. Unless stated otherwise, all commands assume the current working directory is the repository root (`OmniDocBench/`).

The workflow is organised in four stages:

1. **Prediction normalisation** (`pred_raw` → `pred_omni`) with the conversion utilities in `tools/data_conversion/`.
2. **Custom metric computation** using `tools/eval/run_detection_custom_metrics.py`.
3. **Visual debugging** of individual pages with the same script in inspection mode.
4. **Result consolidation** with `tools/eval/create_result_tables.py`.

Each section provides command examples, a description of the inputs and outputs, and implementation details where they matter for reproducibility.

---

## 1. OCR Inference Pipelines Notebooks

[MinerU](https://console.cloud.google.com/vertex-ai/colab/notebooks;source=shared?authuser=0&hl=en&project=data-sandbox-408714&activeNb=projects%2Fdata-sandbox-408714%2Flocations%2Fus-central1%2Frepositories%2F688aaa15-fef3-4d33-964d-9b43c63b6792)

[DeepSeekOCR](https://console.cloud.google.com/vertex-ai/colab/notebooks;source=owned?authuser=0&hl=en&project=data-sandbox-408714&activeNb=projects%2Fdata-sandbox-408714%2Flocations%2Fus-central1%2Frepositories%2F18c0985b-963b-4a5b-9f68-2befdd379e38)

[DotsOCR](https://console.cloud.google.com/vertex-ai/colab/notebooks;source=owned?authuser=0&hl=en&project=data-sandbox-408714&activeNb=projects%2Fdata-sandbox-408714%2Flocations%2Fus-central1%2Frepositories%2F7c1c8d57-564a-4b8f-ac59-9fb4c404b3e5)

[PPStructureV3](https://console.cloud.google.com/vertex-ai/colab/notebooks;source=shared?authuser=0&hl=en&project=data-sandbox-408714&activeNb=projects%2Fdata-sandbox-408714%2Flocations%2Fus-central1%2Frepositories%2Fef03eedd-425d-4425-8b26-8579afcb47b2)

[MonkeyOCR](https://console.cloud.google.com/vertex-ai/colab/notebooks;source=shared?authuser=0&hl=en&project=data-sandbox-408714&activeNb=projects%2Fdata-sandbox-408714%2Flocations%2Fus-central1%2Frepositories%2Fb1ed9ea1-81c8-47de-bb2b-2bb1bcc9699b)


These notebooks represent development-ready solutions for large-scale document OCR using state-of-the-art vision-language models. Each one of them implements their own detailed strategies, but we can summarize the following ones to be common among them:

**Implementations**:
1. **Atomic checkpointing** prevents data loss between executions or inestable environments
2. **Page-aware resume** enables granular progress tracking for PDFs
3. **Multi-stage coordinate transformations** ensure pixel-accurate bounding boxes
4. **Comprehensive error handling** isolates failures without crashing pipelines
5. **vLLM integration** provides 4x throughput improvements over standard backends

**Production Deployment**: These notebooks can be adapted for on-premise deployment by:
- Replacing Colab-specific code (`files.download()`, widget UI)
- Adding database persistence for results
- Implementing distributed processing across GPU clusters
- Adding webhook callbacks for completion notifications

### 1.1. Shared Infrastructure Components

All the notebooks follow a consistent architectural pattern optimized for Google Colab's GPU runtime environment:

#### **GPU Runtime Management**
- **Target Hardware**: NVIDIA A100, T4, or equivalent CUDA-capable GPUs
- **CUDA Version**: 11.8 (cu118) for maximum compatibility across Colab runtimes
- **PyTorch Version**: 2.5.1 with CUDA 11.8 bindings
- **Verification Strategy**: Explicit version assertions to prevent CUDA/PyTorch mismatches
  
#### **Dependency Installation Strategy**

All notebooks implement a **clean slate installation** approach to avoid dependency conflicts:

1. **Complete Uninstallation** of existing PyTorch/vLLM installations
2. **Pinned Version Installation** with specific CUDA wheels
3. **Verification Step** to confirm compatibility
4. **Protobuf Pinning** to avoid MessageFactory errors (common in TensorFlow/vLLM interop)

### 1.2 Batch Processing Architecture

All notebooks implement production-grade batch processing with:

1. **Incremental Checkpointing**: Results saved after each batch to prevent data loss
2. **Resume Capability**: Detection of previously processed images to avoid redundant work
3. **Atomic Writes**: Temporary file + rename pattern to prevent corruption
4. **Error Isolation**: Batch-level error handling that doesn't crash entire pipeline
5. **Progress Tracking**: Real-time ETA calculation and progress reporting

### 1.3 vLLM Integration Strategy

**vLLM** (Versatile Large Language Model inference) is a high-throughput, memory-efficient inference engine for LLMs. All three notebooks use vLLM because:

1. **PagedAttention**: Reduces GPU memory waste by 2-4x via virtual memory paging
2. **Continuous Batching**: Dynamically batches requests for maximum throughput
3. **Optimized CUDA Kernels**: Fused operations for attention, RoPE, and layer norms
4. **Multi-GPU Support**: Tensor parallelism for models exceeding single GPU memory

**Memory Management:**
- `gpu_memory_utilization=0.8`: Reserves 20% GPU RAM for CUDA operations
- `max_model_len`: Balances context capacity vs. batch size
- Dynamic batching: vLLM automatically batches concurrent requests

### 1.4 Image Preprocessing Pipeline

All notebooks implement specialized preprocessing for document images:

1. **Standard Image Loading**: through the Image module.
2. **PDF Handling**: suited for cases which require using PyMuPDF (fitz) for High-Quality PDF Rendering.
3. **DPI Normalization**: suited for cases which require an improvement over the quality of the samples: Balances text clarity with GPU memory constraints.

### 1.5. Model Implementations

This section provides detailed technical documentation for each of the five Colab notebook implementations used in the benchmark pipeline. Each model has unique characteristics, preprocessing strategies, and output formats that are critical for understanding the evaluation results.

---

#### **1.5.1 MonkeyOCR Batch Inference**

**Purpose**: Production-grade batch OCR for document understanding with multi-stage pipeline  
**Model**: MonkeyOCR-pro-1.2B (1.2 billion parameters)  
**Key Feature**: Multi-backend support with automatic fallback (vLLM → transformers)

**Architecture**: Multi-Stage Pipeline
1. **Layout Detection** → YOLO-based structure extraction
2. **Reading Order** → LayoutReader ordering  
3. **Text Recognition** → VLM-based OCR with vLLM acceleration

##### **Model Components**

**1. Layout Detection Stage**
- **Model**: `doclayout_yolo_docstructbench_imgsz1280_2501.pt`
- **Input**: Full document image at 1280px
- **Output**: Bounding boxes + class labels (Title, Text, Table, Figure, etc.)
- **Framework**: Ultralytics YOLO architecture optimized for document layouts

**2. Reading Order Stage**
- **Model**: LayoutReader (Relation module)
- **Input**: Layout bounding boxes from YOLO
- **Output**: Reading sequence (directed graph of elements)
- **Purpose**: Ensures text flows logically (not just left-to-right, top-to-bottom)

**3. Text Recognition Stage**
- **Model**: MonkeyOCR-pro-1.2B VLM
- **Backend Options**:
  - `vllm_async`: Asynchronous vLLM with queue (fastest)
  - `transformers`: HuggingFace standard (fallback)
  - `lmdeploy`: Alternative high-performance backend
- **Context Window**: 16,384 tokens
- **Output Format**: Markdown with structured layout preservation

##### **Configuration System**

**YAML-Based Configuration** (`model_configs.yaml`):
```yaml
device: cuda
models_dir: model_weight

weights:
  doclayout_yolo: Structure/doclayout_yolo_docstructbench_imgsz1280_2501.pt
  PP-DocLayout_plus-L: Structure/PP-DocLayout_plus-L
  layoutreader: Relation

chat_config:
  weight_path: model_weight/Recognition
  backend: vllm_async
  data_parallelism: 1
  model_parallelism: 1
  queue_config:
    max_batch_size: 256
    queue_timeout: 1
    max_queue_size: 2000
```

**Key Parameters:**
- `backend: vllm_async`: Enables asynchronous batching for 2-3x speedup
- `max_batch_size: 256`: vLLM internal micro-batching limit
- `queue_timeout: 1`: Wait 1 second to accumulate batch before processing

##### **Multi-Backend Fallback System**

**Unique Feature**: Automatic backend switching on failure

**Implementation Flow**:
1. Attempt inference with `vllm_async` (primary, fastest)
2. On failure, patch config to use `transformers` backend
3. Retry batch with fallback backend
4. Restore original backend for next batch
5. If both fail, record detailed error for all images in batch

**Failure Modes Handled**:
- vLLM CUDA OOM errors
- Protobuf version conflicts
- Model loading failures

##### **Output Format: *_middle.json**

**Structure**:
```json
{
  "blocks": [
    {
      "bbox": [x1, y1, x2, y2],
      "category": "Title",
      "text": "Document Title",
      "reading_order": 0
    },
    {
      "bbox": [x1, y1, x2, y2],
      "category": "Text",
      "text": "Body paragraph content...",
      "reading_order": 1
    }
  ]
}
```

**JSONL Results Format**:
```json
{
  "image_path": "/path/to/image.png",
  "status": "ok",
  "error": null,
  "middle_json": "/path/to/output_middle.json",
  "blocks": [...],
  "batch_index": 5,
  "duration_ms": 1234
}
```

##### **Batch Processing Workflow**

**Step-by-Step Execution**:
1. **Image Discovery**: Recursively scan input folder for images
2. **Resume Check**: Load existing JSONL, skip processed images
3. **Batch Creation**: Create temporary symlinks to avoid path issues
4. **Inference**: Run `parse.py` on batch directory
5. **Output Parsing**: Extract `*_middle.json` files
6. **JSONL Append**: Atomic append of batch results
7. **Cleanup**: Remove temporary symlinks

**Symlink Strategy**: Ensures consistent naming for output file discovery while preserving original paths in results.

---

#### **1.5.2 DeepSeek-OCR vLLM Batch Inference**

**Purpose**: Structured layout extraction with precise bounding box coordinates  
**Model**: deepseek-ai/DeepSeek-OCR (VLM)  
**Key Feature**: Advanced coordinate transformation system for pixel-accurate bounding boxes

**Unique Capabilities**:
- Pixel-level bounding box accuracy
- Multi-format output (JSON, CSV, debug images)
- Sophisticated coordinate space transformations

##### **Model Architecture**

**Technical Specifications**:
- **Model Family**: Vision-Language Model (VLM)
- **Vision Encoder**: High-resolution image encoder (undisclosed architecture)
- **Language Model**: Transformer decoder (parameter count not public)
- **Input Resolution**: Dynamic (smart_resize algorithm)
- **Output Format**: Structured JSON with bboxes + text
- **Supported Categories**: 11 types (Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title)

##### **Prompt Engineering System**

**Three Built-in Prompt Templates**:

1. **OCR Mode** (`ocr`):
   - Simple text extraction without structure
   - Fastest inference mode
   - Use case: Pure text extraction

2. **Layout-Only Mode** (`layout-only`):
   - Structure detection without OCR
   - Outputs bbox and category only
   - Use case: Layout analysis pipelines

3. **Layout-All Mode** (`layout-all`, default):
   - Complete document understanding
   - Outputs bbox, category, and text content
   - Text formatting rules:
     - Picture: omit text field
     - Formula: format as LaTeX
     - Table: format as HTML
     - Others: format as Markdown
   - Most comprehensive output

##### **Coordinate Transformation System**

**Challenge**: The model outputs bounding boxes in a normalized coordinate space that differs from the original image dimensions due to:
1. Model's internal `smart_resize` algorithm
2. Optional fitz preprocessing (DPI normalization)

**Solution**: Multi-stage coordinate transformation pipeline

**Stage 1: Understanding smart_resize**

**Algorithm** (from Qwen2-VL image processor):
```python
def smart_resize(original_width, original_height, 
                 factor=28, min_pixels=3136, max_pixels=11289600):
    """
    Resize image to satisfy:
    - Dimensions divisible by factor (28)
    - Total pixels in [min_pixels, max_pixels]
    """
    # Round to nearest multiple of factor
    h_bar = max(factor, round(original_height / factor) * factor)
    w_bar = max(factor, round(original_width / factor) * factor)
    
    # Enforce max_pixels constraint
    if h_bar * w_bar > max_pixels:
        beta = sqrt((original_height * original_width) / max_pixels)
        h_bar = max(factor, floor((original_height / beta) / factor) * factor)
        w_bar = max(factor, floor((original_width / beta) / factor) * factor)
    
    # Enforce min_pixels constraint with nested check
    elif h_bar * w_bar < min_pixels:
        beta = sqrt(min_pixels / (original_height * original_width))
        h_bar = ceil((original_height * beta) / factor) * factor
        w_bar = ceil((original_width * beta) / factor) * factor
        
        # Ensure we didn't exceed max_pixels
        if h_bar * w_bar > max_pixels:
            beta_new = sqrt((h_bar * w_bar) / max_pixels)
            h_bar = max(factor, floor((h_bar / beta_new) / factor) * factor)
            w_bar = max(factor, floor((w_bar / beta_new) / factor) * factor)
    
    return w_bar, h_bar
```

**Why This Matters**:
- Model outputs bboxes relative to `(w_bar, h_bar)`, not original size
- Factor=28: Required by model's patch-based architecture
- Nested constraints: Tricky edge cases require exact replication

**Stage 2: Reverse smart_resize Transformation**

Converts model bbox to original image coordinates by replicating the exact smart_resize logic and calculating separate scale factors for width and height.

**Stage 3: Fitz Preprocessing Reversal** (if enabled)

If DPI normalization was applied, an additional scaling step maps from the preprocessed image dimensions back to the original dimensions.

##### **JSON Repair System**

**Problem**: Long documents may cause model output truncation, breaking JSON syntax

**Solution**: Intelligent JSON extraction and repair

**Effectiveness**:
- Recovers 90%+ of truncated outputs
- Preserves bbox and category (critical for layout analysis)
- Marks truncated text explicitly with "TRUNCATED" label

##### **Multi-Format Output Generation**

**1. Structured JSON**:
```json
[
  {
    "image_name": "doc001.png",
    "width": 2480,
    "height": 3508,
    "results": [
      {
        "category": "Title",
        "text": "Annual Report 2024",
        "bbox_norm": [120, 80, 880, 140],
        "bbox_abs": [298, 198, 2182, 347]
      }
    ]
  }
]
```

**2. Regions CSV Table**: For spreadsheet analysis and data science workflows

**3. Debug Visualization Images**: Overlay bounding boxes with category labels for visual validation

**Use Cases**:
- JSON: Structured pipelines, databases
- CSV: Spreadsheet analysis, data science workflows
- Debug images: Visual validation, error diagnosis

##### **Configuration System**

**Widget-Based UI** (ipywidgets):
- Runtime configuration without editing code
- Interactive parameter adjustment
- User-friendly for non-programmers

**Key Parameters**:
- `input_dir`: Path to image folder
- `output_path`: JSON output file
- `fitz_preprocess`: Enable DPI normalization
- `batch_size`: Images per batch (1-64)
- `prompt_mode`: OCR/layout-all/layout-only/custom

---

#### **1.5.3 DoTS.ocr Colab Batch Inference**

**Purpose**: Multilingual OCR with advanced DPI handling and PDF support  
**Model**: rednote-hilab/dots.ocr (1.7B parameters)  
**Key Feature**: Comprehensive DPI normalization via fitz preprocessing

**Unique Capabilities**:
- Page-aware PDF processing (tracks individual pages)
- Production-grade resume with page-level granularity
- Extensive error handling with full tracebacks
- Multilingual support (80+ languages including CJK)

##### **Model Architecture**

**Technical Specifications**:
- **Model Size**: 1.7 billion parameters
- **Languages**: Multilingual (80+ languages including CJK)
- **Architecture**: Vision-encoder + decoder LLM
- **Input Resolution**: Dynamic via smart_resize
- **Context Window**: 16,384 tokens
- **Special Features**: Qwen2-VL image processor integration

**Image Processing Constants**:
- `IMAGE_FACTOR = 28`: Patch size divisibility requirement
- `MIN_PIXELS = 3136`: 56×56 minimum resolution
- `MAX_PIXELS = 11289600`: 3360×3360 maximum resolution

**Rationale**:
- `factor=28`: Model uses 28×28 pixel patches (ViT architecture)
- `min_pixels`: Ensures readable text (< 56px → too blurry)
- `max_pixels`: GPU memory limit (higher → OOM errors)

##### **Advanced PDF Processing**

**Challenge**: PDFs contain multiple pages that must be tracked individually

**Solution**: Page-Aware Task Expansion

**Task Structure**:
```python
# Each task tracks both file path and specific page
{
  "path": "/path/to/document.pdf",
  "page": 2  # Zero-based page index
}
```

**Resume System** (page-granular):
- Builds processed set with page awareness: `{image_path}|{page_number}`
- Filters remaining tasks to exclude already-processed page/file combinations
- Enables resume even if PDF processing was interrupted mid-document

**Benefits**:
- Accurate progress tracking (pages, not just files)
- Enables partial PDF reprocessing
- Fault tolerance for long-running PDF batches

##### **DPI Normalization Deep Dive**

**Why DPI Matters for OCR**:

**Problem**: Document images from diverse sources have inconsistent DPI:
- Screenshots: 72 DPI (low quality)
- Scans: 150-600 DPI (high quality, large files)
- Web images: 96 DPI (standard monitor resolution)
- Phone photos: Variable DPI metadata

**Impact on OCR**:
- Low DPI: Small text becomes unreadable → poor accuracy
- High DPI: Exceeds model input limits → cropping or excessive downsampling
- Inconsistent DPI: Model sees vastly different effective resolutions → unstable performance

**DoTS.ocr Fitz Preprocessing Solution**:

**Step-by-Step Algorithm**:
1. Load original image and extract DPI metadata
2. Convert image to temporary PDF in memory
3. Render PDF page at target DPI (200) using transformation matrix
4. Handle oversized images (>4500px) with fallback to 72 DPI
5. Return normalized image

**Performance Impact**:
- +25-35% preprocessing time
- +5-15% OCR accuracy on mixed-quality datasets
- Negligible impact on high-quality scans

**When to Enable**:
- ✅ Mixed-source datasets (web scraping, user uploads)
- ✅ Unknown provenance images
- ❌ Controlled scans from single source
- ❌ Already normalized images

##### **Comprehensive Error Handling**

**Multi-Level Error Capture**:

**Level 1: Image-Level Errors**
- File not found → Skip with warning
- Image loading failure → Record error in results

**Level 2: Parsing Errors**
- JSON decode error → Save raw text output
- Bbox transformation error → Mark as "[PROCESSING ERROR]"

**Level 3: Batch-Level Errors**
- vLLM engine crash → Record full traceback
- Continue to next batch without stopping pipeline

**Level 4: System-Level Errors**
- CUDA OOM → Caught by vLLM, may retry with smaller batch

**Benefits**:
- **No Silent Failures**: Every error is logged with context
- **Partial Results**: Completed images saved even if batch fails
- **Debuggability**: Full tracebacks enable root cause analysis
- **Resilience**: Pipeline continues despite individual failures

##### **Progress Tracking & ETA**

**Real-Time Progress Metrics**:
- Processing rate (seconds per image)
- Estimated time to completion
- Per-batch timing
- Cumulative elapsed time

**Console Output Example**:
```
[12/50] Saved partial: 384/1600 (batch 32 imgs, 45.2s). ETA ~ 2730s
```

**Metadata Tracking**:
```json
{
  "meta": {
    "model": "rednote-hilab/dots.ocr",
    "start_time_utc": "2025-10-27T14:32:18.123Z",
    "elapsed_sec": 540.2,
    "eta_sec": 2730.0,
    "processed_so_far": 384,
    "total_target": 1600,
    "errors": 3,
    "resumed": true,
    "already_done": 0,
    "last_batch_sec": 45.2
  }
}
```

---

#### **1.5.4 MinerU VLM Batch Inference**

**Purpose**: High-performance layout detection using vision-language models  
**Model**: opendatalab/MinerU2.5-2509-1.2B  
**Key Feature**: Dual backend support (vLLM for production, Transformers for development)

**Unique Capabilities**:
- Native batch processing via `batch_layout_detect()`
- Normalized bbox coordinates (0-1 range)
- Simple, clean API through `MinerUClient`
- Efficient memory utilization

##### **Model Architecture**

**Technical Specifications**:
- **Model Size**: 1.2 billion parameters
- **Model Family**: Qwen2VL-based vision-language model
- **Input Processing**: Multi-scale image encoding
- **Output Format**: Normalized bounding boxes with category labels
- **Supported Categories**: Standard document layout elements (header, text, title, image, table, list, footer, page_number)

##### **Backend Options**

**1. vLLM Backend (Production)**:
```python
from vllm import LLM
from mineru_vl_utils import MinerUClient

llm = LLM(
    model="opendatalab/MinerU2.5-2509-1.2B",
    gpu_memory_utilization=0.9
)

client = MinerUClient(
    backend="vllm-engine",
    vllm_llm=llm
)
```

**Advantages**:
- High throughput for batch processing
- Efficient GPU memory usage
- Faster inference on large datasets

**2. Transformers Backend (Development)**:
```python
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from mineru_vl_utils import MinerUClient

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "opendatalab/MinerU2.5-2509-1.2B",
    dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    "opendatalab/MinerU2.5-2509-1.2B",
    use_fast=True
)

client = MinerUClient(
    backend="transformers",
    model=model,
    processor=processor
)
```

**Advantages**:
- Easier debugging
- More control over inference parameters
- Better for single-image testing

##### **Batch Processing Implementation**

**Core Features**:
1. **Automatic Resume**: Loads existing results and skips processed images
2. **Error Handling**: Per-image error tracking without pipeline failure
3. **Incremental Saving**: Results saved after each batch
4. **Progress Tracking**: tqdm progress bars for batch processing

**Processing Workflow**:
```python
# Batch configuration
batch_size = 50

# Load images
batch_images = [Image.open(path).convert("RGB") for path in batch_paths]

# Batch inference
batch_results = client.batch_layout_detect(batch_images)

# Results structure per image
for filename, blocks in zip(batch_filenames, batch_results):
    all_results[filename] = blocks
```

##### **Output Format**

**Normalized Bbox Structure**:
```json
{
  "image_name.jpg": [
    {
      "bbox": [x1_norm, y1_norm, x2_norm, y2_norm],  // 0-1 range
      "type": "text",
      "score": 0.97
    },
    {
      "bbox": [x1_norm, y1_norm, x2_norm, y2_norm],
      "type": "table",
      "score": 0.95
    }
  ]
}
```

**Coordinate System**:
- Bounding boxes are normalized to [0, 1] range
- Convert to absolute coordinates: `x_abs = x_norm * image_width`
- No complex coordinate transformations required

##### **Visualization Support**

**Built-in Visualization**:
```python
from PIL import ImageDraw

# Convert normalized to pixel coordinates
x1 = bbox[0] * img_width
y1 = bbox[1] * img_height
x2 = bbox[2] * img_width
y2 = bbox[3] * img_height

# Draw with color-coded categories
draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
draw.text((x1 + 5, y1 + 5), block_type, fill=color)
```

**Color Palette by Category**:
- header: red
- text: blue
- title: green
- image: purple
- table: orange
- list: brown
- footer: gray
- page_number: cyan

##### **Error Handling Strategy**

**Per-Image Error Tracking**:
```python
try:
    img = Image.open(img_path).convert("RGB")
    batch_images.append(img)
except Exception as e:
    # Record error without stopping batch
    all_results[filename] = {"error": str(e)}
```

**Batch-Level Error Recovery**:
```python
try:
    batch_results = client.batch_layout_detect(batch_images)
except Exception as e:
    # Mark all images in failed batch
    for filename in batch_filenames:
        all_results[filename] = {"processing_error": str(e)}
```

##### **Performance Characteristics**

**Batch Size Recommendations**:
- **vLLM Backend**: 32-64 images per batch
- **Transformers Backend**: 8-16 images per batch
- **Memory Consideration**: Adjust based on GPU VRAM

**Throughput Metrics** (A100 GPU):
- vLLM: ~6-8 images/second
- Transformers: ~2-3 images/second

---

#### **1.5.5 PP-StructureV3 Layout Detection Benchmark**

**Purpose**: Production-grade layout detection using PaddlePaddle's latest models  
**Models**: Multiple PP-DocLayout variants and RT-DETR models  
**Key Feature**: Robust multi-level fallback system for HPI and TensorRT acceleration

**Unique Capabilities**:
- Multi-model benchmarking in single notebook
- Automatic acceleration fallback: (HPI+TRT) → (TRT) → (HPI) → (GPU fp32) → (CPU)
- Integrated evaluation with OmniDocBench metrics
- Per-image JSON outputs for detailed analysis

##### **Supported Models**

**Available Model Options**:
1. **PP-DocLayout_plus-L**: Large variant with enhanced accuracy
2. **PP-DocLayout-L**: Standard large model
3. **PP-DocLayout-M**: Medium model (balanced speed/accuracy)
4. **PP-DocLayout-S**: Small model (fastest inference)
5. **PP-DocBlockLayout**: Specialized block-level detection
6. **RT-DETR-H_layout_17cls**: Real-time DETR with 17 categories
7. **RT-DETR-H_layout_3cls**: Real-time DETR with 3 categories
8. **PicoDet variants**: Lightweight detection models

##### **Acceleration Strategy**

**Multi-Level Fallback System**:

**Attempt 1: HPI + TensorRT on GPU**
- Highest performance when available
- Requires High-Performance Inference plugin
- TensorRT FP16 precision by default

**Attempt 2: TensorRT Only on GPU**
- Falls back if HPI unavailable
- Still provides acceleration via TensorRT
- Maintains good performance

**Attempt 3: HPI Only on GPU**
- Falls back if TensorRT unavailable
- Uses High-Performance Inference without TensorRT
- Slower than TRT but faster than baseline

**Attempt 4: Plain GPU (FP32)**
- Standard GPU inference
- No special acceleration
- Reliable fallback

**Attempt 5: CPU Fallback**
- Last resort when GPU unavailable
- Significantly slower but ensures execution
- Useful for testing/debugging

**Implementation**:
```python
def _make_predictor(model_name, enable_hpi, use_trt, trt_precision, device='gpu'):
    return LayoutDetection(
        model_name=model_name,
        device=device,
        enable_hpi=enable_hpi,
        use_tensorrt=use_trt,
        precision=trt_precision,
    )

# Automatic fallback logic
attempts = [
    (enable_hpi, use_trt, trt_precision, 'gpu', 'HPI+TRT on GPU'),
    (False, use_trt, trt_precision, 'gpu', 'TRT only on GPU'),
    (enable_hpi, False, 'fp32', 'gpu', 'HPI only on GPU'),
    (False, False, 'fp32', 'gpu', 'Plain GPU (fp32)'),
    (False, False, 'fp32', 'cpu', 'CPU fallback'),
]
```

##### **Configuration System**

**Key Parameters**:
```python
IMAGE_DIR = Path('/content/OmniDocBench/data/images')
GT_JSON = Path('/content/OmniDocBench/data/OmniDocBench.json')
OUTPUT_BASE = Path('/content/PPLayoutBenchmark')

BATCH_SIZE = 16
CONF_THRESHOLD = None  # Use model defaults
USE_TENSORRT = True
TRT_PRECISION = 'fp16'
ENABLE_HPI = False  # Default False for Colab compatibility
```

**Batch Processing Configuration**:
- Resumable: Skips already-processed images
- Atomic writes: Per-image JSON files
- Progress tracking: tqdm progress bars
- Error isolation: Failed batches don't stop pipeline

##### **Label Mapping System**

**Canonical Category Mapping**:
```python
LABEL_ALIAS_TO_EVAL = {
    'title': 'title',
    'document_title': 'title',
    'paragraph_title': 'title',
    'section_title': 'title',
    
    'text': 'text',
    'plain text': 'text',
    'references': 'text',
    
    'figure': 'figure',
    'image': 'figure',
    'code_txt': 'figure',
    
    'table': 'table',
    'table_caption': 'table_caption',
    'table_footnote': 'table_footnote',
    
    'formula': 'isolate_formula',
    'equation': 'isolate_formula',
    'inline_formula': 'isolate_formula',
    
    'header': 'abandon',
    'footer': 'abandon',
    'page_number': 'abandon',
}
```

**Purpose**: Normalizes diverse model outputs to OmniDocBench evaluation categories

##### **Output Format**

**Per-Image JSON Structure**:
```json
{
  "boxes": [
    {
      "label": "text",
      "score": 0.95,
      "coordinate": [x1, y1, x2, y2]
    },
    {
      "label": "table",
      "score": 0.92,
      "coordinate": [x1, y1, x2, y2]
    }
  ]
}
```

**OmniDocBench Detection Format**:
```json
{
  "results": [
    {
      "image_name": "doc001_0",
      "bbox": [x1, y1, x2, y2],
      "category_id": 0,
      "score": 0.95
    }
  ],
  "categories": {
    "0": "text",
    "1": "table",
    "2": "figure"
  }
}
```

##### **Evaluation Integration**

**Automatic OmniDocBench Evaluation**:
1. **Config Generation**: Auto-generates YAML config for each model
2. **Metric Computation**: Runs OmniDocBench evaluation pipeline
3. **mAP Extraction**: Extracts COCO-style mAP from metrics
4. **Leaderboard Creation**: Generates comparative CSV and table

**Evaluation Config Structure**:
```yaml
detection_eval:
  metrics: ['COCODet']
  dataset:
    dataset_name: detection_dataset_simple_format
    ground_truth:
      data_path: /path/to/ground_truth.json
    prediction:
      data_path: /path/to/predictions.json
  categories:
    eval_cat:
      block_level: [title, text, figure, table, ...]
    gt_cat_mapping: {...}
    pred_cat_mapping: {...}
```

##### **Leaderboard Output**

**Generated Metrics**:
```
=== Layout Detection Leaderboard (higher mAP is better) ===
| model                    | mAP   | metrics_path |
|--------------------------|-------|--------------|
| PP-DocLayout_plus-L      | 0.876 | result/      |
| RT-DETR-H_layout_3cls    | 0.854 | result/      |
| PP-DocLayout-S           | 0.841 | result/      |
```

**Saved Outputs**:
- `leaderboard_layout_detection.csv`: Comparative metrics table
- `<model>_aggregate_metrics.json`: Detailed COCO metrics per model
- `predictions_for_eval/<model>/detection_prediction.json`: OmniDocBench-format predictions

##### **Visualization System**

**Debug Visualization Features**:
```python
def visualize_sample(model_name: str, image_name: str):
    """Overlays predicted boxes on original image"""
    # Color-coded by category
    # Score labels for each detection
    # Bounding box visualization
```

**Color Palette**:
- Rotates through: red, blue, green, yellow, purple, orange
- Consistent colors per label within same image
- 3px outline width for visibility

**Use Cases**:
- Visual validation of predictions
- Error diagnosis
- Model comparison
- Presentation materials

##### **Batch Processing Workflow**

**Step-by-Step Execution**:
1. **Model Initialization**: Load predictor with fallback attempts
2. **Image Discovery**: Scan input directory for images
3. **Resume Check**: Skip already-processed images
4. **Batch Inference**: Process images in configurable batches
5. **Per-Image Save**: Atomic write of each result JSON
6. **Aggregation**: Combine per-image results into evaluation format
7. **Evaluation**: Run OmniDocBench metrics
8. **Leaderboard**: Generate comparative table

**Error Handling**:
- Predictor initialization failures → Try all fallback options
- Batch inference failures → Record error, continue pipeline
- Individual image failures → Skip with warning
- Evaluation failures → Report but don't crash

##### **Performance Optimization**

**Memory Management**:
- Batch size adjustment based on model size
- GPU memory monitoring
- Automatic cleanup between batches

**Disk I/O Optimization**:
- Per-image JSON files for resume capability
- Atomic writes to prevent corruption
- ZIP export for easy download

**Throughput Metrics** (A100 GPU, batch=16):
- **PP-DocLayout_plus-L**: ~12-15 images/second
- **PP-DocLayout-S**: ~20-25 images/second
- **RT-DETR-H**: ~15-18 images/second

##### **Export Functionality**

**Raw Predictions Export**:
```python
# Aggregate all per-image JSONs
aggregate_model_dir(model_dir) 
# → /content/PPLayoutBenchmark/exports/<model>_all_predictions.json

# ZIP entire predictions folder
shutil.make_archive("preds_per_image", "zip", root_dir, base_dir)
# → /content/preds_per_image.zip

# Trigger browser download (Colab)
files.download(zip_file)
```

**Benefits**:
- Easy transfer of results
- Backup of intermediate outputs
- Sharing predictions with team
- Integration with external tools


### 1.6 Debugging Common Issues

#### **Issue 1: CUDA Out of Memory**

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions**:
1. Reduce batch size: `BATCH_SIZE = BATCH_SIZE // 2`
2. Lower GPU utilization: `gpu_memory_utilization=0.7`
3. Reduce max_model_len: `max_model_len=8192`
4. Clear cache: `torch.cuda.empty_cache()`

#### **Issue 2: vLLM Engine Crashes**

**Symptoms**:
```
[CRITICAL] Batch 12 failed. vLLM engine may be dead.
```

**Solutions**:
1. Check logs for protobuf version conflicts
2. Check out for memory issues, reduce batch_size or GPU Memory Utilization
3. Restart runtime and resume
4. Update vLLM: `pip install -U vllm`

#### **Issue 3: Incorrect Bounding Boxes**

**Symptoms**: Boxes don't align with visual elements

**Debugging**:
1. Enable debug image generation
2. Print intermediate coordinate transformations:
   ```python
   print(f"Original dims: {original_dims}")
   print(f"Model input dims: {model_input_dims}")
   print(f"Resized dims (calculated): {resized_dims}")
   print(f"Raw bbox: {model_bbox}")
   print(f"Final bbox: {final_bbox}")
   ```
3. Verify preprocessing steps coordinates and behaviour and check if transformation is correctly applied
4. Check for model output in unexpected format
5. Visually debug with the original images

#### **Issue 4: JSON Parsing Failures**

**Symptoms**:
```
JSONDecodeError: Expecting ',' delimiter: line 1 column 1234
```

**Solutions**:
1. Use `extract_and_repair_json()` function
2. Increase `max_tokens` to avoid truncation
3. Validate model output format against expected schema
4. Fall back to raw text output if repair fails


## 2. Converting raw predictions into OmniDocBench format

All conversion scripts expose a CLI that reads the raw prediction artefacts produced by each model and emits an OmniDocBench-style JSON object. The output has a stable structure:

```json
[
  {
    "page_info": { "image_path": "...", "width": 1654, "height": 2339, ... },
    "layout_dets": [
      {
        "category_type": "text",
        "poly": [x1, y1, x2, y1, x2, y2, x1, y2],
        "score": 0.97,
        ...
      },
      ...
    ]
  }
]
```

Whenever possible, the converters reuse metadata from `data/OmniDocBench_data/OmniDocBench.json` to align coordinate systems and image paths with the benchmark reference.

### 2.1 Generic multi-model converter

**Script**: `tools/data_conversion/convert_predictions_to_omnidoc.py`

Supports PP-DocLayout-S/L, RT-DETR-H, DotsOCR JSON, Dolphin Stage 1 and DocLayNet-DETR.

```bash
python tools/data_conversion/convert_predictions_to_omnidoc.py \
  --pred-type dotsocr \
  --input data/publicBench_data/predictions/raw/dotsocr_results.json \
  --output data/publicBench_data/predictions/dotsocr.json \
  --gt data/OmniDocBench_data/OmniDocBench.json
```

- `--pred-type`: selects the normalisation branch (`pp_doclayout_s`, `pp_doclayout_plus_l`, `rt_detr_h`, `dotsocr`, `dolphin`, `doclaynet_detr`).
- `--input`: raw inference file or directory (model-dependent).
- `--output`: OmniDocBench-compatible JSON destination.
- `--gt`: optional manifest providing page metadata; defaults to the public OmniDocBench release.

### 2.2 Model-specific converters

| Script | Purpose | Key arguments |
| --- | --- | --- |
| `tools/data_conversion/mineru_to_omnidocbench.py` | MinerU VLM detection dumps (normalised boxes per page). | `--mineru-json`, `--gt`, `--images-dir`, `--output` |
| `tools/data_conversion/deepseek_to_omnidoc.py` | DeepSeek OCR page-wise results. | `--input`, `--output` |
| `tools/data_conversion/docling_to_omnidocbench.py` | DocTags TXT exports from Docling/Granite. | `--manifest`, `--docling-dir`, `--output`, `--default-score`, `--loc-grid` |
| `tools/data_conversion/dotsocr_to_omnidocbench.py` | DotsOCR dataset JSON (supports PDF mode). | `input_json`, `output_path`, `--reference-json`, `--md-text-to-table`, `--pdf`, `--images-dir`, `--indent`, `--quiet` |
| `tools/data_conversion/landingai_to_omnidocbench.py` | LandingAI document-level JSON responses. | `input_dir`, `output_path`, `--reference-json`, `--pattern`, `--indent`, `--quiet` |
| `tools/data_conversion/marker_to_omnidoc.py` | Marker OCR HTML/JSON responses per page. | `--input-dir`, `--image-dir`, `--output` |
| `tools/data_conversion/monkeyocr_to_omnidoc.py` | Monkey OCR Pro outputs. | `--input-dir`, `--output`, `--images-dir`, `--reference-json`, `--score-field` |
| `tools/data_conversion/paddlepaddle_to_omnidocbench.py` | PaddlePaddle Layout results. | `input_dir`, `output_path`, `--reference-json`, `--pattern`, `--indent` |
| `tools/data_conversion/reducto_to_omnidocbench.py` | Reducto document JSONs. | `input_dir`, `output_path`, `--reference-json`, `--pattern`, `--indent`, `--quiet` |
| `tools/data_conversion/aws_to_omnidocbench.py` | AWS Textract GroundTruth manifests. | `--input-dir`, `--reference-json`, `--output`, `--pattern`, `--indent` |

### 2.3 Tooling directory layout
- `tools/data_conversion/`: one-off converters that transform raw model outputs or third-party manifests into OmniDocBench JSON.
- `tools/model_infer/`: reference inference drivers for each model (DocLayout-YOLO, DocLayNet, Dolphin, Marker, MonkeyOCR, etc.).
- `tools/eval/`: evaluation CLIs, metric exporters and reporting utilities.
- `tools/eval/examples/`: small reproducible examples and diagnostics.
- `tools/load_test/`: scripts to stress-test remote inference endpoints and render sanity previews.

All converters apply label remapping to our canonical taxonomy (`title`, `text`, `figure`, `table`, etc.), clamp boxes to the page dimensions, and create canonical quadrilateral polygons.

---

## 3. Custom metric engine

**Script**: `tools/eval/run_detection_custom_metrics.py`

```bash
python tools/eval/run_detection_custom_metrics.py \
  --config configs/PublicBench/layout_detection_dotsocr.yaml \
  --output-dir data/publicBench_data/results/new_metrics \
  --merge \
  --limit-merge \
  --max-merge 20 \
  --adjust-boxes
```

### 3.1 Inputs
- **YAML config** (`--config`): points to the ground truth JSON, the converted prediction JSON, the set of categories to evaluate (`eval_cat`), and any label mapping required for the model.
- **Output directory** (`--output-dir`): root under which per-model folders are created (e.g. `data/publicBench_data/results/new_metrics/dotsocr/detection_eval/`).

### 3.2 Algorithmic pipeline
1. **Candidate clustering & merge search**  
   - Builds a bipartite graph between ground truth boxes and predictions that intersect above zero IoU.  
   - For each connected component it enumerates unions of overlapping boxes up to `--max-merge`. Two modes are supported:  
     - **Full merge** (`--merge`): considers combinations on both sides (GT and predictions) to maximise IoU.  
     - **Limited merge** (`--limit-merge`): restricts combinations to 1→N or N→1 (one side merges, the other remains atomic). This bounds the search and prevents exponential blow-up.

2. **Matching**  
   - Within each component, selects the GT–prediction union pair that yields the highest IoU and locks both sides.  
   - Remaining GT boxes become false negatives; remaining predictions become false positives.

3. **Post-merge absorption**  
   - After the initial assignment, any unmatched box with >90% overlap against an existing union is absorbed to eliminate spurious FP/FN stemming from minor misalignments (e.g. multi-line text split across rows).

4. **Optional box adjustment** (`--adjust-boxes`)  
   - Loads the Azure OCR words for each page (`data/publicBench_data/azure-ocr/<doc>_ocr_result.json`).  
   - Crops the matched unions to the tightest rectangle covering all words whose area lies ≥90% inside the union. This is skipped if any of the merged elements is a `figure`.  
   - Recomputes IoU based on the adjusted boxes and overlays word polygons in the debug visualisation (semi-transparent grey).

5. **Metric aggregation**  
   - `total_matches`: number of matched unions.  
   - `total_false_negatives` / `total_false_positives`: unmatched GT/pred boxes after absorption.  
   - `mean_iou`: average IoU across matched unions.  
   - `per_class.failures` & `failure_rate`: weighted sum of errors per category (matches to other classes or missing), normalised by the GT count.  
   - `confusion`: dense confusion matrix detailing where GT boxes end up (e.g. `table→text`, `figure→missing`).  
   - Per-document CSV mirrors the same fields allowing micro-analysis at page level.

### 3.3 CLI flags
- `--merge`, `--limit-merge`, `--max-merge`: configure the search strategy as described above.
- `--adjust-boxes`: enables OCR-guided box shrink and visual overlays.
- `--image <page>`: restricts processing to a single page and produces a debug PNG only.
- `--max-pages`, `--start-page`: additional filters for sampling (see script help for full reference).

### 3.4 Dependencies
- `utils/match_utils.py`: IoU computation, merge combinatorics, Azure OCR loader.
- `utils/ocr_utils.py`: helpers for polygon/bbox conversions and OCR alignment.
- Standard Python stack: `numpy`, `Pillow`, `yaml`, `typer/argparse`. All dependencies are listed in `OmniDocBench/requirements.txt`.

### 3.5 Outputs
- `<model>/detection_eval/<model>_aggregate_metrics.json`: aggregate metrics and confusion matrix.
- `<model>/detection_eval/<model>_per_document_metrics.csv`: per-page metrics.
- `<model>/detection_eval/<model>_debug_*.png`: generated automatically for pages with FP/FN or when `--image` is provided.

---

## 4. Visual debugging

Use the same script with `--image` to inspect any page:

```bash
python tools/eval/run_detection_custom_metrics.py \
  --config configs/PublicBench/layout_detection_dotsocr.yaml \
  --output-dir data/publicBench_data/results/new_metrics \
  --merge --limit-merge --max-merge 20 --adjust-boxes \
  --image 68f0f279af342e1497f85730_0.jpg
```

The resulting PNG (stored alongside the metrics) overlays:
- Solid lines for GT unions and dashed lines for prediction unions.
- IoU and label assignment in the top-right corner of each merged cluster.
- Colour palette by class; red is reserved for FP, FN or label mismatches.
- Azure OCR words (semi-transparent grey) when `--adjust-boxes` is active.
- Summary banner with FP, FN, average IoU, and off-diagonal confusion mass.

This mode is essential when tuning merge thresholds or diagnosing label confusions.

---

## 5. Result consolidation

**Script**: `tools/eval/create_result_tables.py`

```bash
python tools/eval/create_result_tables.py \
  --base-dir data/publicBench_data/results/new_metrics \
  --models landingai dotsocr_pdf dotsocr dotsocr_dpi marker reducto minerU monkey_pro_3B deepseek pp yolo
```

The utility reads each `<model>_aggregate_metrics.json`, formats numeric columns (IoU, counts, failure rates) and prints two ASCII tables:

1. **Metric summary** – mean IoU, matches, FN/FP, failure rates per class.
2. **Confusion breakdown** – percentage flow of ground-truth categories into predicted classes (`text→table`, `table→missing`, etc.).

If `--models` is omitted, the default order matches the command above. Use this script after every evaluation run to refresh the tables in the README and provide stakeholders with a quick comparison.

---

## 6. Current benchmark snapshot

Generated with `python tools/eval/create_result_tables.py` on the latest runs.

### 6.1 Aggregate metrics

| Model | Mean IoU | Matches | FN | FP | Text fail % | Table fail % | Figure fail % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| landingai | 0.935 | 1153 | 3 | 7 | 22.3 | 9.0 | 25.8 |
| dotsocr_pdf | 0.961 | 1431 | 37 | 7 | 10.9 | 17.3 | 43.6 |
| dotsocr | 0.958 | 1485 | 4 | 2 | 7.6 | 18.3 | 59.1 |
| dotsocr_dpi | 0.959 | 1447 | 31 | 15 | 9.1 | 17.6 | 57.5 |
| marker | 0.941 | 1424 | 19 | 4 | 5.1 | 22.8 | 41.7 |
| reducto | 0.939 | 1283 | 9 | 22 | 16.7 | 24.8 | 32.4 |
| minerU | 0.965 | 1391 | 7 | 9 | 15.7 | 27.2 | 79.4 |
| monkey_pro_3B | 0.924 | 1036 | 184 | 2 | 34.6 | 24.2 | 79.7 |
| deepseek | 0.941 | 1099 | 345 | 21 | 29.5 | 26.0 | 74.8 |
| pp | 0.931 | 1026 | 160 | 14 | 33.1 | 26.9 | 79.0 |
| yolo | 0.935 | 938 | 351 | 28 | 38.2 | 44.1 | 83.2 |

### 6.2 Confusion flows (percent of GT instances)

| Model | Text→Table % | Text→Missing % | Text→Figure % | Table→Text % | Table→Missing % | Table→Figure % | Figure→Text % | Figure→Missing % | Figure→Table % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| landingai | 18.9 | 0.2 | 2.9 | 8.1 | 0.0 | 1.0 | 20.2 | 0.0 | 5.6 |
| dotsocr_pdf | 7.9 | 2.8 | 0.2 | 17.3 | 0.0 | 0.0 | 38.9 | 1.2 | 3.5 |
| dotsocr | 7.3 | 0.2 | 0.1 | 18.2 | 0.0 | 0.2 | 57.2 | 0.6 | 1.4 |
| dotsocr_dpi | 5.3 | 2.2 | 1.6 | 16.2 | 0.5 | 1.0 | 54.8 | 1.8 | 0.9 |
| marker | 4.3 | 0.7 | 0.0 | 22.6 | 0.0 | 0.2 | 32.8 | 6.0 | 3.0 |
| reducto | 15.7 | 0.2 | 0.8 | 24.3 | 0.0 | 0.5 | 27.1 | 3.5 | 1.8 |
| minerU | 15.3 | 0.3 | 0.0 | 27.0 | 0.0 | 0.2 | 73.8 | 1.3 | 4.3 |
| monkey_pro_3B | 22.9 | 11.2 | 0.5 | 13.6 | 9.1 | 1.4 | 56.5 | 15.3 | 7.9 |
| deepseek | 9.5 | 19.7 | 0.3 | 23.8 | 1.9 | 0.2 | 13.2 | 59.6 | 2.0 |
| pp | 21.9 | 10.1 | 1.1 | 17.4 | 7.7 | 1.8 | 60.2 | 12.6 | 6.2 |
| yolo | 15.2 | 21.2 | 1.8 | 38.4 | 4.8 | 1.0 | 25.6 | 51.7 | 5.9 |

Use these tables as the canonical reference when comparing model revisions or preparing reports.

---

## 7. Quick reference

- **Converted predictions**: `data/publicBench_data/predictions/<model>.json`
- **Metric outputs**: `data/publicBench_data/results/new_metrics/<model>/detection_eval/`
- **Azure OCR words**: `data/publicBench_data/azure-ocr/<doc_id>_ocr_result.json`
- **Evaluation configs**: `OmniDocBench/configs/PublicBench/`

Follow the four stages above for any new model: convert predictions, compute metrics, inspect errors, and regenerate the comparison tables. The entire process is reproducible and does not rely on hidden state beyond the assets stored in this repository.
