# Aligning Knowledge Concepts to Whole Slide Images for Precise Histopathology Image Analysis

## 1. Usage Tutorial:

Here is a colab tutorial for model testing, please **[click here](https://colab.research.google.com/drive/1HEStFLSfR1DHx5s_valMEZpyT6D__2hl?usp=sharing)**.

**[Training Tutorial.ipynb](https://github.com/HKU-MedAI/ConcepPath/blob/main/Tutorial.ipynb)**

## 2. Descriptive Prompt Generation

Generate descriptive prompt on two levels: slide-level and patch-level. Templates for questioning on LLM (GPT-4) are:

### Patch-level + Postive(EBV):

```txt
Q1: Please provide a summary of the factors found in primary tumor whole slide images that may indicate Epstein-Barr Virus (EBV)-positive subtype of Gastric cancer, along with a description of their image features in short terms, separated by semicolons. Please avoid using subtype names in your response.
Q*: Please avoid using the word "EBV" in your response.
Q*: Please give more.
```
### Patch-level + Negative(others):

```
Q2: Suppose we let Gastric cancer subtypes other than Epstein-Barr Virus (EBV)-positive subtype into one group. Please provide a summary of the factors found in primary tumor whole slide images that may indicate this group of subtypes, along with a description of their image features in short terms, separated by semicolons.
Q*: Please avoid using the word "EBV" in your response.
Q*: Please give more.
```
### Slide-level + Postive(EBV):

```txt
Q3: Summary the appearance of whole slide images of Epstein-Barr Virus (EBV)-positive subtype of Gastric cancer in short terms, separated by semicolons.
Q*: Please avoid using the word "EBV" in your response.
Q*: Please give more.
```
### Slide-level + Negative(others):

```
Q4: Suppose we let Gastric cancer subtypes other than Epstein-Barr Virus (EBV)-positive subtype into one group. Summary the appearance of whole slide images of this group in short terms, separated by semicolons. 
Q*: Please avoid using the word "EBV" in your response.
Q*: Please give more.
```

Write these answers with format as the following: (columns="prompt_level,label,descriptive_prompt")
```csv
patch,ebv,Lymphoid stroma: Dense lymphoid stroma with infiltrating lymphocytes; lymphocytes surrounding tumor cells.
patch,ebv,Lymphoepithelioma-like appearance: Undifferentiated tumor cells with abundant lymphocytic infiltrate; tumor cells interspersed with lymphocytes.
patch,ebv,Epstein-Barr Virus (EBV)-encoded RNA (EBER) expression: Strong and diffuse nuclear staining for EBER; intense staining in the nuclei of tumor cells.
...
slide,ebv,Lymphocyte-rich infiltrate; Well-defined glandular structures; Epithelial cell apoptosis; Nuclear atypia; Syncytial growth pattern; Presence of viral-associated features; Intense lymphoid reaction; Absence of signet ring cells; Expression of viral RNA.
```
Next, run the helper script to parse these prompts into JSON format. The primary functions included within the helper scripts are:
 - split prompts into two file based on its level (slide vs.patch)
 - avoid using class label(or associated keywords) 
 - filter the top-N concepts of patch-level prompts

Note: `EXPERIMENT_NAME` is the same as created in `main.py`.
```bash
EXPERIMENT_NAME=molecular_ebv_others_full_train python tools/parse_prompt.py
```

## 3. Reference:

 - ### HER2 Summary Prompts

Shang, Jiuyan et al. “Evolution and clinical significance of HER2-low status after neoadjuvant therapy for breast cancer.” Frontiers in oncology vol. 13 1086480. 22 Feb. 2023, doi:10.3389/fonc.2023.1086480

Venetis, Konstantinos et al. “HER2 Low, Ultra-low, and Novel Complementary Biomarkers: Expanding the Spectrum of HER2 Positivity in Breast Cancer.” Frontiers in molecular biosciences vol. 9 834651. 15 Mar. 2022, doi:10.3389/fmolb.2022.834651

Zhang, Huina et al. “HER2-low breast cancers: Current insights and future directions.” Seminars in diagnostic pathology vol. 39,5 (2022): 305-312. doi:10.1053/j.semdp.2022.07.003

An, Junsha et al. “New Advances in Targeted Therapy of HER2-Negative Breast Cancer.” Frontiers in oncology vol. 12 828438. 4 Mar. 2022, doi:10.3389/fonc.2022.828438

Lee, Hyo-Jae et al. “HER2-Positive Breast Cancer: Association of MRI and Clinicopathologic Features With Tumor-Infiltrating Lymphocytes.” AJR. American journal of roentgenology vol. 218,2 (2022): 258-269. doi:10.2214/AJR.21.26400

den Hollander, Petra et al. “Targeted therapy for breast cancer prevention.” Frontiers in oncology vol. 3 250. 23 Sep. 2013, doi:10.3389/fonc.2013.00250

Patrizio, Armando et al. “Thyroid Metastasis from Primary Breast Cancer.” Journal of clinical medicine vol. 12,7 2709. 4 Apr. 2023, doi:10.3390/jcm12072709


 - ### Lung (LUAD vs. LUSC) Summary Prompts

Song, Xiaojie et al. “Construction of a Novel Ferroptosis-Related Gene Signature for Predicting Survival of Patients With Lung Adenocarcinoma.” Frontiers in oncology vol. 12 810526. 3 Mar. 2022, doi:10.3389/fonc.2022.810526

Qiu, Wang-Ren et al. “Predicting the Lung Adenocarcinoma and Its Biomarkers by Integrating Gene Expression and DNA Methylation Data.” Frontiers in genetics vol. 13 926927. 30 Jun. 2022, doi:10.3389/fgene.2022.926927

Wang, Wen et al. “What's the difference between lung adenocarcinoma and lung squamous cell carcinoma? Evidence from a retrospective analysis in a cohort of Chinese patients.” Frontiers in endocrinology vol. 13 947443. 29 Aug. 2022, doi:10.3389/fendo.2022.947443

Hu, Xiaoshan et al. “Novel cellular senescence-related risk model identified as the prognostic biomarkers for lung squamous cell carcinoma.” Frontiers in oncology vol. 12 997702. 17 Nov. 2022, doi:10.3389/fonc.2022.997702


 - ### Molecular (EBV,MSI,CIN,GS) Summary Prompts

Zhu, Chunrong et al. “Genomic Profiling Reveals the Molecular Landscape of Gastrointestinal Tract Cancers in Chinese Patients.” Frontiers in genetics vol. 12 608742. 14 Sep. 2021, doi:10.3389/fgene.2021.608742

Dedieu, Stéphane, and Olivier Bouché. “Clinical, Pathological, and Molecular Characteristics in Colorectal Cancer.” Cancers vol. 14,23 5958. 2 Dec. 2022, doi:10.3390/cancers14235958

Hinata, Munetoshi, and Tetsuo Ushiku. “Detecting immunotherapy-sensitive subtype in gastric cancer using histologic image-based deep learning.” Scientific reports vol. 11,1 22636. 22 Nov. 2021, doi:10.1038/s41598-021-02168-4

Han, Shuting et al. “Epstein-Barr Virus Epithelial Cancers-A Comprehensive Understanding to Drive Novel Therapies.” Frontiers in immunology vol. 12 734293. 10 Dec. 2021, doi:10.3389/fimmu.2021.734293

Sun, Keran et al. “EBV-Positive Gastric Cancer: Current Knowledge and Future Perspectives.” Frontiers in oncology vol. 10 583463. 14 Dec. 2020, doi:10.3389/fonc.2020.583463

Genitsch, Vera et al. “Epstein-barr virus in gastro-esophageal adenocarcinomas - single center experiences in the context of current literature.” Frontiers in oncology vol. 5 73. 26 Mar. 2015, doi:10.3389/fonc.2015.00073

Saito, Motonobu, and Koji Kono. “Landscape of EBV-positive gastric cancer.” Gastric cancer : official journal of the International Gastric Cancer Association and the Japanese Gastric Cancer Association vol. 24,5 (2021): 983-989. doi:10.1007/s10120-021-01215-3

Joshi, Smita S, and Brian D Badgwell. “Current treatment and recent progress in gastric cancer.” CA: a cancer journal for clinicians vol. 71,3 (2021): 264-279. doi:10.3322/caac.21657

Amato, Martina et al. “Microsatellite Instability: From the Implementation of the Detection to a Prognostic and Predictive Role in Cancers.” International journal of molecular sciences vol. 23,15 8726. 5 Aug. 2022, doi:10.3390/ijms23158726

Ratti, Margherita et al. “Microsatellite instability in gastric cancer: molecular bases, clinical perspectives, and new treatment approaches.” Cellular and molecular life sciences : CMLS vol. 75,22 (2018): 4151-4162. doi:10.1007/s00018-018-2906-9

Salnikov, Mikhail et al. “Tumor-Infiltrating T Cells in EBV-Associated Gastric Carcinomas Exhibit High Levels of Multiple Markers of Activation, Effector Gene Expression, and Exhaustion.” Viruses vol. 15,1 176. 7 Jan. 2023, doi:10.3390/v15010176

