# ***Geological-knowledge-guided information extraction from raster geological map using Multimodal Large Language Models***

 

![img](images/sample.png) 

## *Overview*

Geo-IGM is a geological map interpretation framework that integrates domain knowledge with Multimodal Large Language Models (MLLMs). The system extracts the legend metadata library utilizing MLLMs, segments various geological regions in the main map using an optimized Felzenszwalb algorithm, and matches the segmented regions of the main map with the legend metadata through a color and texture feature fusion mechanism. Finally, it constructs a closed-loop workflow of “intelligent interpretation - manual review” via an interactive visualization panel. 

## *Installation*

### *Prerequisites*

```python
Python 3.10+

PyTorch 2.0+
```

### *Dependencies*

Install the required packages using pip:

```python
pip install -r requirements.txt
```

## *API Keys Configuration*

Create a file in the project root directory with the following environment variables: .env

```python
# Gemini API Key Configuration

GEMINI_API_KEY="Enter your actual API Key here"
```



## *Data Preparation*

### *Required Data*

#### *1．Geological Maps*

Format: PNG or JPG images

Location: data/maps

***Note*:** Images should be high-resolution and include a legend. The legend must be located in a standardized position.

 

#### *2.Ground Truth Regions for Main Map Segmentation (for evaluation)*

This experiment utilizes a combination of automated tools and manual annotation to construct the ground truth file gt_regions.json, which is used to evaluate the accuracy of both segmentation and matching.

***Format*:** JSON

***Location*:** eval/gt

***Note*:** The file should contain the boundary coordinates of each region within the main map, along with the specific legend information matched to those regions.

 

### *Directory Structure*

```c++
data/

├── maps/        # Geological map images

│  ├── map1.jpg

│  ├── map2.jpg

│  └── ...

 

demo/

├── output/

├── eval_demo.py

├── main_demo.py

├── quick_test.py

├── viewer_demo.py

 

eval/

├── gt/         # Geological map ground truth

│  ├── gt1_regions.json

│  ├── gt2_regions.json

│  └── ...

├── metrics/       # Evaluation metrics

├── output/       # Evaluation output files

├── eval.py    # Evaluation script

 

legendParser/      # Legend metadata extraction

├── models/

├── dependencies/

├── toolpool/

 

main/

├── segment.py      # Main map segmentation

├── match.py       # Main map-legend matching

├── run.py        # Main program entry point

├── tools.py       # Utility tools

├── viewer.py      # Visualization viewer
```

  

## *Project Execution*

### *Step 1: Execute Data Extraction*

```python
python run.py
```

***Outputs*:** Extracted prediction result data, large model legend parsing cache, and intermediate process images. (regions_ui.json, legend_gemini_cache.json, color_img, text_img, vis_felz_regions_before_merge.png, vis_felz_regions_after_merge.png)

 

### *Step 2: Visual Inspection*

```python
python viewer.py --img "map.jpg" --regions "regions_ui.json" --cache "legend_gemini_cache.json"
```

***Outputs:*** This step does not directly generate or save new files. Its purpose is to read the image, region JSON, and cache JSON, and then open a visualization window on your screen. You can drag, zoom, and click on specific polygons within this window to view their lithological attributes and model parsing results.

 

### *Step 3: Accuracy Evaluation*

This stage is divided into two main sub-steps: first, generating ground truth labels via an annotation tool, and then comparing the predicted results with the ground truth to calculate accuracy metrics.

***Ground Truth Annotation:***

```python
python eval.py annotate --img "map.jpg" --pred_json "regions_ui.json" --save_json "gt_regions.json"
```

***Outputs:*** Generates and saves the reviewed ground truth file: gt_regions.json.

 

***Metrics Evaluation:***

```python
python eval.py evaluate --img "map.jpg" --pred_json "pred_regions.json" --gt_json "gt_regions.json" --out_dir "./eval_results" --tolerance 3
```

***Outputs:*** Core metrics file, boundary overlay comparison image, and predicted label rendering image. (evaluation_metrics.json, boundary_overlay.png, pred_label_map.png)

 

## *Quick Start*

The “Demo” folder includes a simple test file. By adding your own OpenAI API key in the “quick_test.py” file, you can run it directly. The results will be saved in the “output” directory.

```python
cd demo

python quick_test.py
```

## *Dataset & Language Notes*

- ***Original Data & Chinese Output:*** The dataset currently used for evaluation in this system is primarily based on ***Chinese geological maps***. Therefore, the metadata (such as geological descriptions, lithology descriptions, stratigraphic eras, etc.) extracted by the large model, cached in legend_gemini_cache.json, and finally outputted in the regions_ui.json and CSV files are ***all in Chinese*** (e.g., “中粒斑状二长花岗岩”, “二叠纪三叠纪”).
- ***Special Note on Paper Prompts:*** Although the large model prompts presented in the related academic paper are in English, ***the actual prompts used during system execution are in Chinese***. The English versions presented in the paper are specialized translations made to meet publication requirements. To ensure academic rigor, we paid special attention to the accurate mapping of geological terminology during the translation process. By cross-referencing authoritative geological dictionaries and comparing multiple translation tools, we have ensured the highest possible consistency between the English descriptions and the original Chinese meanings.